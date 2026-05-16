#!/home/quant/czsc_venv/bin/python3
"""czsc 缠论有效性验证 — 简化版信号"""

import sys, os, time, json
sys.path.insert(0, '/home/quant/.openclaw/workspace')

import numpy as np
import pandas as pd
from datetime import date, datetime
from collections import defaultdict
from pathlib import Path

from czsc import CZSC
from czsc.objects import RawBar
from czsc.enum import Mark, Direction, Freq

TOP_N = 100
INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 5
PARQUET = "/home/quant/.openclaw/workspace/quant/data/daily_parquet/_all.parquet"
OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/czsc_valid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PQC = ["open","high","low","close","volume"]

print(f"{'='*55}")
print(f"🔰 czsc 缠论有效性验证 v4")
print(f"  版本: czsc 0.7.10 (纯Python)")
print(f"  标的: 日均量前{TOP_N}只")
print(f"  回测: 2011-01-01 ~ 2026-04-30")
print(f"{'='*55}")
t0 = time.time()

df = pd.read_parquet(PARQUET, columns=["code","date"]+PQC)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
all_dates = sorted(df['date'].unique())
print(f"✅ 数据: {len(df)}行, {len(all_dates)}天, {df['code'].nunique()}只, {time.time()-t0:.0f}s")

# 筛选标的（用2010全年数据筛选）
vol = df[df['date'] < "2011-01-01"].groupby('code')['volume'].mean().sort_values(ascending=False)
top_codes = set(vol.head(TOP_N * 3).index.tolist()[:TOP_N])
print(f"✅ 标的: {len(top_codes)}只")

warmup_dates = [d for d in all_dates if d < "2011-01-01"]
bt_dates = [d for d in all_dates if "2011-01-01" <= d <= "2026-04-30"]
print(f"   预热: {len(warmup_dates)}天, 回测: {len(bt_dates)}天")

# 预热：收集K线
print("Phase 1: Collect bars + Init CZSC...")
stock_bars = defaultdict(list)
for i, ds in enumerate(warmup_dates):
    for _, r in df[df['date'] == ds].iterrows():
        if r['code'] in top_codes:
            stock_bars[r['code']].append(RawBar(
                symbol=r['code'], id=i, dt=datetime.strptime(ds, '%Y-%m-%d'),
                freq=Freq.D, open=r['open'], close=r['close'],
                high=r['high'], low=r['low'], vol=r.get('volume',0), amount=0))

czsc_map = {}
for code in top_codes:
    bars = stock_bars.get(code, [])
    if len(bars) >= 60:
        czsc_map[code] = CZSC(bars, max_bi_count=50)
active = set(czsc_map.keys())
bi_c = [len(e.bi_list) for e in czsc_map.values()]
print(f"✅ CZSC: {len(active)}只 active, 笔数: μ={np.mean(bi_c):.0f} m={np.median(bi_c):.0f}")

# 回测
print("Phase 2: Backtest")
cash = INITIAL_CAPITAL
peak_equity = cash
positions = {}
trade_log = []
equity_curve = []
idx_offset = len(warmup_dates)

for di, ds in enumerate(bt_dates):
    tdt = date.fromisoformat(ds)
    day = df[df['date'] == ds]
    if day.empty: continue
    idx = idx_offset + di
    
    # 更新CZSC
    for _, r in day.iterrows():
        engine = czsc_map.get(r['code'])
        if engine:
            engine.update(RawBar(symbol=r['code'], id=idx, dt=datetime.strptime(ds, '%Y-%m-%d'),
                freq=Freq.D, open=r['open'], close=r['close'],
                high=r['high'], low=r['low'], vol=r.get('volume',0), amount=0))
    
    # 策略：BI方向改变时交易
    for code in list(positions.keys()):
        engine = czsc_map.get(code)
        if not engine or len(engine.bi_list) < 2: continue
        bis = engine.bi_list
        if bis[-1].direction == Direction.Down:
            # 笔方向转下 → 卖出
            p = positions[code]
            close_val = day[day['code']==code]['close'].values
            sp = close_val[0] if len(close_val) > 0 else p['entry_price']
            pnl = (sp - p['entry_price']) / p['entry_price'] * 100
            cash += p['shares'] * sp
            trade_log.append({"date": ds, "symbol": code, "action": "SELL",
                "price": round(sp,3), "pnl_pct": round(pnl,2), "reason": "bi_down"})
            del positions[code]
    
    for _, r in day.iterrows():
        if r['code'] in positions or len(positions) >= MAX_POSITIONS: break
        engine = czsc_map.get(r['code'])
        if not engine or len(engine.bi_list) < 3: continue
        bis = engine.bi_list
        if bis[-1].direction == Direction.Up:
            # 笔方向转上 → 买入
            price = r['close']
            cost = cash * 0.18
            shares = int(cost / price / 100) * 100
            if shares <= 0: continue
            cost = shares * price
            if cost > cash: continue
            cash -= cost
            positions[r['code']] = {"entry_price": price, "shares": shares, "entry_date": tdt}
            trade_log.append({"date": ds, "symbol": r['code'], "action": "BUY",
                "price": round(price,3), "signal": "bi_up"})
    
    equity = cash + sum(
        p['shares'] * (day[day['code']==code]['close'].values[0] if code in day['code'].values else p['entry_price'])
        for code, p in positions.items())
    peak_equity = max(peak_equity, equity)
    dd = (equity / peak_equity - 1) * 100
    equity_curve.append({"date": ds, "nav": round(equity,2), "dd": round(dd,2)})
    if (di+1) % 600 == 0 or di == len(bt_dates)-1:
        print(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)}")

# 指标
nav_s = [e['nav'] for e in equity_curve]
if not nav_s:
    print("❌ 无交易数据")
    sys.exit(1)
ret_s = [(nav_s[i]/nav_s[i-1]-1) for i in range(1,len(nav_s))]
ann_r = (nav_s[-1]/nav_s[0])**(252/max(len(ret_s),1))-1 if len(ret_s)>0 else 0
sr = np.mean(ret_s)/np.std(ret_s)*np.sqrt(252) if len(ret_s)>0 and np.std(ret_s)>0 else 0
peak_n = nav_s[0]; mdd = 0
for n in nav_s: peak_n = max(peak_n, n); mdd = min(mdd, (n/peak_n-1)*100)
buys = [t for t in trade_log if t['action']=='BUY']; sells = [t for t in trade_log if t['action']=='SELL']
wins = [t for t in sells if t.get('pnl_pct',0)>0]; losses = [t for t in sells if t.get('pnl_pct',0)<=0]
wr = len(wins)/max(len(sells),1)*100
aw = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
al = np.mean([abs(t['pnl_pct']) for t in losses]) if losses else 1
pf = sum(t['pnl_pct'] for t in wins)/max(sum(abs(t['pnl_pct']) for t in losses),1)

print(f"\n{'='*55}")
print(f"📊 czsc 验证结果 (czsc 0.7.10 纯Python)")
print(f"{'='*55}")
print(f"  最终净值: ¥{nav_s[-1]:,.2f}")
print(f"  总收益率: {(nav_s[-1]/INITIAL_CAPITAL-1)*100:.2f}%")
print(f"  年化收益率: {ann_r*100:.2f}%")
print(f"  夏普比率: {sr:.4f}")
print(f"  最大回撤: {mdd:.2f}%")
print(f"  胜率: {wr:.1f}%")
print(f"  平均盈利: {aw:.2f}%")
print(f"  平均亏损: {al:.2f}%")
print(f"  盈亏比: {aw/al:.2f}")
print(f"  盈利因子: {pf:.2f}")
print(f"  总交易: {len(trade_log)} (买入{len(buys)}/卖出{len(sells)})")
print(f"  标的数: {len(active)}, 平均笔数: {np.mean(bi_c):.0f}")

pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"czsc_trades.csv",index=False)
pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"czsc_equity.csv",index=False)
r = {"final_nav": round(nav_s[-1],2), "total_return": round((nav_s[-1]/INITIAL_CAPITAL-1)*100,2),
     "ann_return": round(ann_r*100,2), "sharpe": round(sr,4), "max_drawdown": round(mdd,2),
     "win_rate": round(wr,1), "avg_win": round(aw,2), "avg_loss": round(al,2),
     "win_loss_ratio": round(aw/al,2) if al else 0, "profit_factor": round(pf,2),
     "n_trades": len(trade_log), "n_buys": len(buys), "n_sells": len(sells),
     "stocks": len(active), "avg_bi": round(np.mean(bi_c),0) if bi_c else 0,
     "elapsed": round(time.time()-t0,0)}
with open(OUTPUT_DIR/"czsc_summary.json","w") as f: json.dump(r,f,indent=2)
print(f"\n⏱️ {time.time()-t0:.0f}s 💾 {OUTPUT_DIR}")
