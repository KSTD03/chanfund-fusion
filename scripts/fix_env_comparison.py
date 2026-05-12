#!/usr/bin/env python3
"""快速修复：只提取2024年环境对照，复用已有v30环境分类结果"""
import pandas as pd

# 从v30_run.log提取全量环境分布
# 新分类器（bt_dates范围，1532天）：BULL=525, BEAR=996, OSCILLATE=11
# 旧分类器（R1全量，1532天）：BULL=815, OSCILLATE=141, BEAR=576

# 需要找到2024年在bt_dates中的具体索引
import sys
sys.path.insert(0, '/home/quant/.openclaw/workspace')
from scripts.task3_v21_backtest import load_data

print("加载日期列表...")
data = load_data(max_stocks=0)
all_dates = sorted(data.keys())

# 找到2024年在bt_dates中的索引
bt_dates = [d for d in all_dates if "2020-01-01" <= d <= "2026-12-31"]
warmup_dates = [d for d in all_dates if d < "2020-01-01"]

# 构建指数（只用前800只）
import numpy as np
from collections import defaultdict

n_stocks = 800
vol_ranking = defaultdict(list)
for ds in all_dates:
    if ds >= "2020-01-01": break
    day = data.get(ds, {})
    for sym, bar in day.items():
        v = bar.get("volume", 0)
        if v > 0: vol_ranking[sym].append(v)

avg_vols = {sym: np.mean(vs[-250:]) for sym, vs in vol_ranking.items() if len(vs) >= 20}
top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:n_stocks])
print(f"前{n_stocks}只已选定")

index_c = []
for ds in all_dates:
    day = data.get(ds, {})
    cs = [v["close"] for sym,v in day.items() if sym in top_stocks and v.get("close",0) > 0]
    index_c.append(np.mean(cs) if cs else (index_c[-1] if index_c else 10000))

# R1旧分类器
class MAEnvClassifier:
    def __init__(self):
        self.prices = []
        self.current_env = "OSCILLATE"
    def update(self, price: float):
        if price <= 0: return
        self.prices.append(price)
        if len(self.prices) < 60: return
        if len(self.prices) >= 21:
            rets = [(self.prices[i]-self.prices[i-1])/self.prices[i-1] for i in range(-20,0)]
            vol = np.std(rets) * np.sqrt(252)
            if vol > 0.35:
                self.current_env = "OSCILLATE"
                return
        ma20 = sum(self.prices[-20:])/20
        ma60 = sum(self.prices[-60:])/60
        self.current_env = "BULL" if ma20 > ma60 else "BEAR"

old_cls = MAEnvClassifier()
old_envs = []
for p in index_c:
    old_cls.update(p)
    old_envs.append(old_cls.current_env)

# 新分类器
class AdvancedEnvClassifier:
    def __init__(self):
        self.closes, self.highs, self.lows = [], [], []
        self.current_env = "OSCILLATE"
    def update(self, high, low, close):
        if close <= 0: return
        self.closes.append(close); self.highs.append(high); self.lows.append(low)
        if len(self.closes) < 120: return
        ma20 = sum(self.closes[-20:])/20
        ma60 = sum(self.closes[-60:])/60
        ma120 = sum(self.closes[-120:])/120
        bull = ma20 > ma60 > ma120
        bear = ma20 < ma60 < ma120
        if len(self.closes) >= 60:
            tr = [max(self.highs[i]-self.lows[i], abs(self.highs[i]-self.closes[i-1]), abs(self.lows[i]-self.closes[i-1]))
                  for i in range(-42, 0)]
            atr = np.mean(tr[-20:])
            atr_ratio = atr / self.closes[-1]
            tr_flat = [abs(self.closes[i]-self.closes[i-1])/self.closes[i-1] for i in range(-42, 0)]
            avg_vol = np.mean(tr_flat)
            if bull and atr_ratio < avg_vol * 1.5:
                self.current_env = "BULL"
            elif bear and atr_ratio < avg_vol * 1.5:
                self.current_env = "BEAR"
            elif atr_ratio > avg_vol * 2.0:
                self.current_env = "OSCILLATE"
            else:
                self.current_env = "BULL" if bull else "BEAR"

new_cls = AdvancedEnvClassifier()
new_envs = []
# 简化：复用已有的index_c，估算 high/low
for i, ds in enumerate(all_dates):
    day = data.get(ds, {})
    cand = [v for sym,v in day.items() if sym in top_stocks and v.get("close",0) > 0]
    if cand:
        h = np.mean([v["high"] for v in cand])
        l = np.mean([v["low"] for v in cand])
        c = np.mean([v["close"] for v in cand])
    else:
        c = index_c[i]
        h = c * 1.01; l = c * 0.99
    new_cls.update(h, l, c)
    new_envs.append(new_cls.current_env)

# 提取2024年
records = []
for i, ds in enumerate(all_dates):
    if ds >= "2024-01-01" and ds <= "2024-12-31":
        records.append({"date": ds, "new_env": new_envs[i], "old_env": old_envs[i]})

df = pd.DataFrame(records)
df.to_csv("/home/quant/.openclaw/workspace/output/v30/v30_env_comparison_2024.csv", index=False)

new_bull = sum(1 for r in records if r["new_env"]=="BULL")
new_bear = sum(1 for r in records if r["new_env"]=="BEAR")
new_osc = sum(1 for r in records if r["new_env"]=="OSCILLATE")
old_bull = sum(1 for r in records if r["old_env"]=="BULL")
old_bear = sum(1 for r in records if r["old_env"]=="BEAR")
old_osc = sum(1 for r in records if r["old_env"]=="OSCILLATE")

print(f"2024年总交易日: {len(records)}")
print(f"新分类器: BULL={new_bull}, OSCILLATE={new_osc}, BEAR={new_bear}")
print(f"旧分类器: BULL={old_bull}, OSCILLATE={old_osc}, BEAR={old_bear}")

# 微盘股危机
mc = [r for r in records if "2024-01-01" <= r["date"] <= "2024-02-08"]
mc_new = sum(1 for r in mc if r["new_env"]=="BEAR")
mc_old = sum(1 for r in mc if r["old_env"]=="BEAR")
print(f"\n微盘股危机(1/2~2/8, {len(mc)}天): 新BEAR={mc_new}/{len(mc)}, 旧BEAR={mc_old}/{len(mc)}")

# 924前后(9/15~10/15)
sep = [r for r in records if "2024-09-15" <= r["date"] <= "2024-10-15"]
print(f"\n924行情前后切换(9/15~10/15):")
for r in sep:
    print(f"  {r['date']}: 新={r['new_env']} 旧={r['old_env']}")

# 全量（含所有日期）分布
all_new = {"BULL": sum(1 for e in new_envs if e=="BULL"),
           "BEAR": sum(1 for e in new_envs if e=="BEAR"),
           "OSCILLATE": sum(1 for e in new_envs if e=="OSCILLATE")}
all_old = {"BULL": sum(1 for e in old_envs if e=="BULL"),
           "BEAR": sum(1 for e in old_envs if e=="BEAR"),
           "OSCILLATE": sum(1 for e in old_envs if e=="OSCILLATE")}
print(f"\n全量分布({len(all_dates)}天):")
print(f"  新: BULL={all_new['BULL']}, OSC={all_new['OSCILLATE']}, BEAR={all_new['BEAR']}")
print(f"  旧: BULL={all_old['BULL']}, OSC={all_old['OSCILLATE']}, BEAR={all_old['BEAR']}")
