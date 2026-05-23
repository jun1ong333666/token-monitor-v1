#!/usr/bin/env python3
"""Token Monitor Backend — Flask server with SQLite storage and API proxy."""

import sys
import os
import json
import time
import sqlite3
import threading
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# PyInstaller 打包后 __file__ 指向临时目录，需要用 sys.executable 定位
if getattr(sys, 'frozen', False):
    ROOT_DIR = Path(sys.executable).resolve().parent.parent
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / 'data'
CONFIG_PATH = ROOT_DIR / 'config.json'

# Overridable via CLI for packaged Electron app
_override_data_dir = None
_override_config = None


def get_data_dir():
    return Path(_override_data_dir) if _override_data_dir else DATA_DIR


def get_config_path():
    return Path(_override_config) if _override_config else CONFIG_PATH

DB_PATH = get_data_dir() / 'usage.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app)

DEFAULT_CONFIG = {
    'openai_api_key': '',
    'api_base': '',
    'top_up_url': '',
    'balance_warning': 5,
    'api_key_2': '',
    'api_base_2': '',
    'top_up_url_2': '',
    'balance_warning_2': 5,
    'api_key_3': '',
    'api_base_3': '',
    'top_up_url_3': '',
    'balance_warning_3': 5,
    'budget': 0,
    'auto_launch': True,
    'notifications': {'desktop': True},
}

# DeepSeek 定价 ($/1M tokens)
PRICING = {
    'deepseek-v4-pro':    {'input': 0.55,  'output': 2.19},
    'deepseek-v4-flash':  {'input': 0.14,  'output': 0.28},
    'deepseek-chat':      {'input': 0.27,  'output': 1.10},
    'deepseek-reasoner':  {'input': 0.55,  'output': 2.19},
}
PRICING_DEFAULT = {'input': 0.27, 'output': 1.10}


def calc_cost(model, prompt_tokens, completion_tokens):
    price = PRICING.get(model, PRICING_DEFAULT)
    input_cost = (prompt_tokens / 1_000_000) * price['input']
    output_cost = (completion_tokens / 1_000_000) * price['output']
    return round(input_cost + output_cost, 6)


# ── Database ────────────────────────────────────────────

def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS usage_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        request_count INTEGER DEFAULT 1,
        cost REAL DEFAULT 0.0
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS daily_summary (
        date TEXT PRIMARY KEY,
        total_tokens INTEGER DEFAULT 0,
        request_count INTEGER DEFAULT 0,
        cost REAL DEFAULT 0.0
    )''')
    db.execute('''CREATE TABLE IF NOT EXISTS balance_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        total_balance REAL DEFAULT 0,
        topped_up REAL DEFAULT 0,
        granted REAL DEFAULT 0
    )''')
    db.commit()
    return db


def normalize_model(name):
    """统一模型名称，防止 API 返回不同格式导致数据库重复"""
    if not name:
        return 'unknown'
    n = name.lower().strip()
    if 'deepseek' in n:
        if 'flash' in n:
            return 'deepseek-v4-flash'
        if 'pro' in n:
            return 'deepseek-v4-pro'
    return name


def record_usage(model, prompt_tokens, completion_tokens, cost=None):
    model = normalize_model(model)
    if cost is None:
        cost = calc_cost(model, prompt_tokens, completion_tokens)
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    total = prompt_tokens + completion_tokens

    db.execute('''INSERT INTO usage_log (date, model, prompt_tokens, completion_tokens, total_tokens, cost)
                  VALUES (?, ?, ?, ?, ?, ?)''',
               (today, model, prompt_tokens, completion_tokens, total, cost))

    db.execute('''INSERT INTO daily_summary (date, total_tokens, request_count, cost)
                  VALUES (?, ?, 1, ?)
                  ON CONFLICT(date) DO UPDATE SET
                  total_tokens = total_tokens + ?,
                  request_count = request_count + 1,
                  cost = cost + ?''',
               (today, total, cost, total, cost))
    db.commit()
    db.close()


# ── Config ──────────────────────────────────────────────

def load_config():
    cfg_path = get_config_path()
    if cfg_path.exists():
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(get_config_path(), 'w') as f:
        json.dump(cfg, f, indent=2)


# ── DeepSeek Balance ────────────────────────────────────

def _ssl_context():
    import ssl
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        pass
    return ctx

def fetch_deepseek_balance(api_key):
    import urllib.request
    ctx = _ssl_context()
    ctx.check_hostname = False
    ctx.verify_mode = 0  # ssl.CERT_NONE
    req = urllib.request.Request('https://api.deepseek.com/user/balance')
    req.add_header('Authorization', f'Bearer {api_key}')
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            raw = resp.read().decode()
            data = json.loads(raw)
            return data, None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


# ── API Routes ──────────────────────────────────────────

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'proxy_url': 'http://127.0.0.1:5099/v1'})


@app.route('/api/proxy-status')
def proxy_status():
    db = get_db()
    today = datetime.now().strftime('%Y-%m-%d')
    row = db.execute(
        'SELECT total_tokens, request_count FROM daily_summary WHERE date = ?',
        [today]
    ).fetchone()
    db.close()

    cfg = load_config()
    return jsonify({
        'proxy_active': True,
        'proxy_url': 'http://127.0.0.1:5099/v1',
        'api_base': cfg.get('api_base', ''),
        'has_api_key': bool(cfg.get('openai_api_key', '')),
        'today_tokens': row['total_tokens'] if row else 0,
        'today_requests': row['request_count'] if row else 0,
    })


def provider_name(base_url):
    if not base_url:
        return None
    u = base_url.lower()
    if 'deepseek' in u:
        return 'DeepSeek'
    if 'openai' in u:
        return 'OpenAI'
    if 'anthropic' in u:
        return 'Anthropic'
    try:
        from urllib.parse import urlparse
        return urlparse(base_url).hostname or 'Unknown'
    except Exception:
        return 'Unknown'


def provider_patterns(name):
    m = {
        'DeepSeek': ['deepseek'],
        'OpenAI': ['gpt-', 'o1-', 'o3-', 'openai'],
        'Anthropic': ['claude', 'anthropic'],
    }
    return m.get(name, [name.lower()])


@app.route('/api/usage')
def usage():
    db = get_db()
    cfg = load_config()
    budget = cfg.get('budget', 0)
    now_local = datetime.now()
    now_iso = now_local.strftime('%Y-%m-%dT%H:%M:%S')

    # Monthly totals
    month_start = now_local.replace(day=1).strftime('%Y-%m-%d')
    row = db.execute(
        'SELECT COALESCE(SUM(total_tokens),0) as tokens, COALESCE(SUM(cost),0) as cost,'
        ' COALESCE(SUM(request_count),0) as reqs FROM daily_summary WHERE date >= ?',
        [month_start]
    ).fetchone()

    # Fetch live balance
    balance_data = None
    balance_error = None
    api_key = cfg.get('openai_api_key', '')
    if api_key:
        balance_data, balance_error = fetch_deepseek_balance(api_key)

    result = {
        'total_tokens': row[0],
        'total_cost': round(row[1], 4),
        'request_count': row[2] if len(row) > 2 else 0,
        'budget': float(budget),
        'balance': None,
        'daily_spend': 0,
        'weekly_cost': 0,
        'monthly_cost': 0,
    }

    if balance_error:
        result['balance_error'] = balance_error

    if balance_data and 'balance_infos' in balance_data:
        bi = balance_data['balance_infos'][0] if balance_data['balance_infos'] else {}
        total_balance = float(bi.get('total_balance', 0))
        topped_up = float(bi.get('topped_up_balance', 0))
        granted = float(bi.get('granted_balance', 0))

        result['balance'] = total_balance
        result['topped_up'] = topped_up
        result['granted'] = granted

        # Record balance snapshot
        db.execute(
            'INSERT INTO balance_snapshots (timestamp, total_balance, topped_up, granted) VALUES (?, ?, ?, ?)',
            [now_iso, total_balance, topped_up, granted]
        )

        def calc_spend(from_hour):
            """Calculate spend since a specific datetime by comparing balance snapshots."""
            first = db.execute(
                'SELECT total_balance, topped_up FROM balance_snapshots WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1',
                [from_hour]
            ).fetchone()
            if first:
                balance_decrease = first['total_balance'] - total_balance
                topped_up_increase = max(topped_up - first['topped_up'], 0)
                return round(max(balance_decrease + topped_up_increase, 0), 2)
            return 0

        # Daily spend: since today 00:00
        today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:%S')
        result['daily_spend'] = calc_spend(today_start)

        # Weekly spend: since Monday 00:00
        weekday = now_local.weekday()
        week_start = (now_local - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:%S')
        result['weekly_cost'] = calc_spend(week_start)

        # Monthly spend: since 1st of month 00:00
        month_start_dt = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M:%S')
        result['monthly_cost'] = calc_spend(month_start_dt)

        # Clean old snapshots (keep 60 days)
        cutoff = (now_local - timedelta(days=60)).strftime('%Y-%m-%dT%H:%M:%S')
        db.execute('DELETE FROM balance_snapshots WHERE timestamp < ?', [cutoff])

    # Providers: list configured API keys with model spend
    month_start_str = now_local.replace(day=1).strftime('%Y-%m-%d')
    today_str = now_local.replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d')
    weekday = now_local.weekday()
    week_start_str = (now_local - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d')

    keys_cfg = [
        (cfg.get('openai_api_key'), cfg.get('api_base', ''), cfg.get('top_up_url', ''), cfg.get('balance_warning', 5)),
        (cfg.get('api_key_2'), cfg.get('api_base_2', ''), cfg.get('top_up_url_2', ''), cfg.get('balance_warning_2', 5)),
        (cfg.get('api_key_3'), cfg.get('api_base_3', ''), cfg.get('top_up_url_3', ''), cfg.get('balance_warning_3', 5)),
    ]

    providers = []
    for ki, (key, base, top_up, bw) in enumerate(keys_cfg):
        if not key:
            continue
        name = provider_name(base) or f'Key {ki + 1}'
        patterns = provider_patterns(name)
        conds = ' OR '.join(['model LIKE ?' for _ in patterns])
        params = [f'%{p}%' for p in patterns]

        p_models = db.execute(
            f'SELECT model, SUM(total_tokens) as tokens, SUM(cost) as cost '
            f'FROM usage_log WHERE date >= ? AND ({conds}) GROUP BY model ORDER BY cost DESC',
            [month_start_str] + params
        ).fetchall()

        daily = db.execute(
            f'SELECT COALESCE(SUM(cost),0) as c FROM usage_log WHERE date >= ? AND ({conds})',
            [today_str] + params
        ).fetchone()

        weekly = db.execute(
            f'SELECT COALESCE(SUM(cost),0) as c FROM usage_log WHERE date >= ? AND ({conds})',
            [week_start_str] + params
        ).fetchone()

        monthly = db.execute(
            f'SELECT COALESCE(SUM(cost),0) as c FROM usage_log WHERE date >= ? AND ({conds})',
            [month_start_str] + params
        ).fetchone()

        providers.append({
            'name': name,
            'index': ki,
            'top_up_url': top_up or '',
            'balance_warning': bw,
            'models': [{'model_name': m['model'], 'cost': round(m['cost'] or 0, 4)} for m in p_models],
            'daily_spend': result.get('daily_spend', 0) if ki == 0 else round(daily['c'] or 0, 2),
            'weekly_cost': result.get('weekly_cost', 0) if ki == 0 else round(weekly['c'] or 0, 2),
            'monthly_cost': result.get('monthly_cost', 0) if ki == 0 else round(monthly['c'] or 0, 2),
        })

    result['providers'] = providers

    db.commit()
    db.close()
    return jsonify(result)


@app.route('/api/models')
def models():
    db = get_db()
    month_start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
    rows = db.execute(
        'SELECT model, SUM(total_tokens) as tokens, SUM(request_count) as reqs, SUM(cost) as cost '
        'FROM usage_log WHERE date >= ? GROUP BY model ORDER BY tokens DESC',
        [month_start]
    ).fetchall()

    total = sum(r['tokens'] for r in rows) or 1
    result = []
    for r in rows:
        result.append({
            'model_name': r['model'],
            'tokens': r['tokens'],
            'percentage': round(r['tokens'] / total * 100, 1),
            'requests': r['reqs'],
            'cost': round(r['cost'], 4),
        })
    db.close()
    return jsonify(result)


@app.route('/api/daily')
def daily():
    db = get_db()
    mode = request.args.get('mode', 'month')

    now = datetime.now()

    if mode == 'week':
        # Current week: Monday 00:00 to Sunday 23:59
        weekday = now.weekday()  # 0=Mon, 6=Sun
        monday = (now - timedelta(days=weekday)).replace(hour=0, minute=0, second=0, microsecond=0)
        sunday = monday + timedelta(days=6)
        start = monday.strftime('%Y-%m-%d')
        end = sunday.strftime('%Y-%m-%d')
        days = 7
        first = monday
    elif mode == 'year':
        # Current year: Jan–Dec monthly aggregates
        year = now.year
        rows = db.execute(
            'SELECT SUBSTR(date,1,7) as month, SUM(total_tokens) as tokens, SUM(request_count) as reqs, SUM(cost) as cost '
            'FROM daily_summary WHERE date >= ? AND date <= ? GROUP BY month ORDER BY month',
            [f'{year}-01-01', f'{year}-12-31']
        ).fetchall()
        month_map = {r['month']: r for r in rows}
        result = []
        for m in range(1, 13):
            key = f'{year}-{m:02d}'
            if key in month_map:
                r = month_map[key]
                result.append({'date': key, 'tokens': r['tokens'], 'requests': r['reqs'], 'cost': round(r['cost'], 4)})
            else:
                result.append({'date': key, 'tokens': 0, 'requests': 0, 'cost': 0})
        db.close()
        return jsonify(result)
    else:
        # Current calendar month: 1st to last day
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            month_end = now.replace(year=now.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            month_end = now.replace(month=now.month + 1, day=1) - timedelta(days=1)
        start = month_start.strftime('%Y-%m-%d')
        end = month_end.strftime('%Y-%m-%d')
        days = month_end.day
        first = month_start

    rows = db.execute(
        'SELECT date, total_tokens, request_count, cost FROM daily_summary WHERE date >= ? AND date <= ? ORDER BY date',
        [start, end]
    ).fetchall()

    result = []
    date_set = {r['date'] for r in rows}
    for i in range(days):
        d = (first + timedelta(days=i)).strftime('%Y-%m-%d')
        if d in date_set:
            r = next(r for r in rows if r['date'] == d)
            result.append({'date': d, 'tokens': r['total_tokens'], 'requests': r['request_count'], 'cost': round(r['cost'], 4)})
        else:
            result.append({'date': d, 'tokens': 0, 'requests': 0, 'cost': 0})

    db.close()
    return jsonify(result)


@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        cfg = load_config()
        safe = {}
        for k, v in cfg.items():
            if 'api_key' in k or 'openai_api_key' in k:
                safe[k] = '***' if v else ''
            else:
                safe[k] = v
        safe['has_api_key'] = bool(cfg.get('openai_api_key'))
        safe['has_api_key_2'] = bool(cfg.get('api_key_2'))
        safe['has_api_key_3'] = bool(cfg.get('api_key_3'))
        safe['balance_warning'] = float(cfg.get('balance_warning', 5))
        safe['balance_warning_2'] = float(cfg.get('balance_warning_2', 5))
        safe['balance_warning_3'] = float(cfg.get('balance_warning_3', 5))
        safe['auto_launch'] = bool(cfg.get('auto_launch', True))
        return jsonify(safe)

    body = request.get_json(silent=True) or {}
    cfg = load_config()

    key_fields = ['openai_api_key', 'api_key_2', 'api_key_3']
    for k in key_fields:
        if k in body and body[k]:
            cfg[k] = body[k]

    str_fields = ['api_base', 'top_up_url', 'api_base_2', 'top_up_url_2', 'api_base_3', 'top_up_url_3']
    for k in str_fields:
        if k in body:
            cfg[k] = body[k]

    if 'api_key' in body and body['api_key']:
        cfg['openai_api_key'] = body['api_key']
    if 'api_base' in body:
        cfg['api_base'] = body['api_base']
    if 'top_up_url' in body:
        cfg['top_up_url'] = body['top_up_url']
    if 'budget' in body:
        cfg['budget'] = float(body['budget'])
    if 'balance_warning' in body:
        cfg['balance_warning'] = float(body['balance_warning'])
    if 'balance_warning_2' in body:
        cfg['balance_warning_2'] = float(body['balance_warning_2'])
    if 'balance_warning_3' in body:
        cfg['balance_warning_3'] = float(body['balance_warning_3'])
    if 'auto_launch' in body:
        cfg['auto_launch'] = bool(body['auto_launch'])

    save_config(cfg)
    return jsonify({'status': 'saved'})
    return jsonify({'status': 'saved'})


@app.route('/api/refresh', methods=['POST'])
def refresh():
    return jsonify({'status': 'ok'})


# ── Proxy: intercept API calls to count tokens ──────────

@app.route('/v1/chat/completions', methods=['POST'])
def proxy_chat():
    import urllib.request
    import urllib.error

    cfg = load_config()
    api_key = cfg.get('openai_api_key', '')
    api_base = cfg.get('api_base', '').rstrip('/')

    if not api_key:
        return jsonify({'error': 'API key not configured'}), 500
    if not api_base:
        return jsonify({'error': 'API base not configured'}), 500

    body = request.get_data()
    target_url = f'{api_base}/v1/chat/completions'

    req = urllib.request.Request(target_url, data=body)
    req.add_header('Authorization', f'Bearer {api_key}')
    req.add_header('Content-Type', 'application/json')

    try:
        ctx = _ssl_context()
        ctx.check_hostname = False
        ctx.verify_mode = 0  # ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            resp_data = resp.read()
            resp_json = json.loads(resp_data.decode())

            # Extract usage
            usage = resp_json.get('usage', {})
            model = resp_json.get('model', 'unknown')
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)

            record_usage(model, prompt_tokens, completion_tokens)
            print(f'[proxy] chat: model={model} prompt={prompt_tokens} comp={completion_tokens} total={prompt_tokens + completion_tokens}')

            return resp_data, resp.status

    except urllib.error.HTTPError as e:
        error_body = e.read()
        print(f'[proxy] chat HTTP error: {e.code} {e.reason}')
        return error_body, e.code
    except Exception as e:
        print(f'[proxy] chat error: {type(e).__name__}: {e}')
        return jsonify({'error': f'Proxy error: {e}'}), 502


# ── Proxy: Anthropic Messages API (Claude Code) ──────────

@app.route('/v1/messages', methods=['POST'])
def proxy_messages():
    import urllib.request
    import urllib.error

    cfg = load_config()
    api_key = cfg.get('openai_api_key', '')
    api_base = cfg.get('api_base', '').rstrip('/')

    if not api_key:
        return jsonify({'error': 'API key not configured'}), 500
    if not api_base:
        return jsonify({'error': 'API base not configured'}), 500

    body = request.get_data()
    body_json = json.loads(body.decode())
    is_stream = body_json.get('stream', False)
    target_url = f'{api_base}/anthropic/v1/messages'

    req = urllib.request.Request(target_url, data=body)
    req.add_header('Authorization', f'Bearer {api_key}')
    req.add_header('Content-Type', 'application/json')

    try:
        ctx = _ssl_context()
        ctx.check_hostname = False
        ctx.verify_mode = 0  # ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
            if not is_stream:
                resp_data = resp.read()
                resp_json = json.loads(resp_data.decode())
                usage = resp_json.get('usage', {})
                model = resp_json.get('model', 'unknown')
                prompt_tokens = usage.get('input_tokens', 0)
                completion_tokens = usage.get('output_tokens', 0)
                record_usage(model, prompt_tokens, completion_tokens)
                print(f'[proxy] messages: model={model} prompt={prompt_tokens} comp={completion_tokens} total={prompt_tokens + completion_tokens}')
                return resp_data, resp.status

            # Streaming: forward SSE chunks and extract usage
            from flask import Response
            def generate():
                model = 'unknown'
                prompt_tokens = 0
                completion_tokens = 0
                for line in resp:
                    chunk = line if isinstance(line, bytes) else line.encode()
                    yield chunk
                    try:
                        text = chunk.decode().strip()
                        if text.startswith('data: '):
                            evt = json.loads(text[6:])
                            if evt.get('type') == 'message_start':
                                if 'message' in evt:
                                    model = evt['message'].get('model', model)
                                    u = evt['message'].get('usage', {})
                                    prompt_tokens = u.get('input_tokens', 0)
                            elif evt.get('type') == 'message_delta':
                                u = evt.get('usage', {})
                                completion_tokens = u.get('output_tokens', 0)
                    except Exception:
                        pass
                if prompt_tokens or completion_tokens:
                    record_usage(model, prompt_tokens, completion_tokens)
                    print(f'[proxy] messages(stream): model={model} prompt={prompt_tokens} comp={completion_tokens} total={prompt_tokens + completion_tokens}')
            return Response(generate(), content_type='text/event-stream')

    except urllib.error.HTTPError as e:
        error_body = e.read()
        print(f'[proxy] messages HTTP error: {e.code} {e.reason}')
        return error_body, e.code
    except Exception as e:
        print(f'[proxy] messages error: {type(e).__name__}: {e}')
        return jsonify({'error': f'Proxy error: {e}'}), 502


def main():
    global _override_data_dir, _override_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5099)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    if args.data_dir:
        _override_data_dir = args.data_dir
    if args.config:
        _override_config = args.config

    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Re-init DB_PATH after overrides are set
    global DB_PATH
    DB_PATH = data_dir / 'usage.db'
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f'Token Monitor backend on port {args.port}')
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
