#!/usr/bin/env python3
"""
Task 5: v2.2 全量回测 — 环境乘数优化参数应用
=============================================
基于 v2.1 (真实基本面评分 + 次日开盘成交)，叠加环境乘数：
  - 牛市 1.0 / 震荡市 1.0 / 熊市 0.6

先跑 500 只快速验证，再启动全量。
"""

import sys, time, logging, json, gc
sys.path.insert(0, '/home/quant/.openclaw/workspace')
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

REPORT_DIR = Path("/home/quant/backtest report")
PARQUET = "/home/quant/.openclaw/workspace/quant/data/daily_parquet/_all.parquet"

# === v2.2 环境乘数 ===
ENV_MULT = {"BULL": 1.0, "OSCILLATE": 1.0, "BEAR": 0.6}

# === 参数 (与 v2.1 一致) ===
INITIAL = 1_000_000; MAX_POS = 5; POS_SIZE = 0.20; FIXED_SL = 0.08
TP1, TP1R = 0.06, 0.30; TP2, TP2R = 0.13, 0.50; TP_TRAIL = 0.91; TP_BREAK = 0.05
COOL_D, COOL_S = 10, 2
VOL_MIN = 0.8; FUND_MIN = 50; FUND_DEF = 50.0
TECH_W, FUND_W = 0.6, 0.4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("task5_v22")
_T_START = time.time()


def load_data(max_stocks: int = 0) -> dict:
    t0 = time.time()
    cols = ["code", "date", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(PARQUET, columns=cols)
    if max_stocks > 0:
        codes = df["code"].unique()[:max_stocks]
        df = df[df["code"].isin(codes)]
    df["_c"] = df["code"].apply(lambda c: f"sh.{c}" if c[0]=="6" else f"sz.{c}")
    df["ds"] = df["date"].astype(str).str[:10]
    df = df[df["close"] > 0]
    data = {d: g.set_index("_c")[["open","high","low","close","volume"]].to_dict("index")
            for d, g in df.groupby("ds")}
    del df; gc.collect()
    log.info(f"📂 数据: {len(data)}天, {time.time()-t0:.0f}s")
    return data


def get_nd(ds: str, all_dates: list):
    idx = next((i for i, d in enumerate(all_dates) if d > ds), None)
    return all_dates[idx] if idx is not None and idx < len(all_dates) else None


def load_scores() -> dict:
    sp = Path("/home/quant/.openclaw/workspace/financial_data/scores.parquet")
    if not sp.exists(): return {}
    t0 = time.time()
    df = pd.read_parquet(sp)
    cache = {}
    for _, row in df.iterrows():
        c = str(row["ts_code"])
        c = f"sh.{c}" if c[0] in "6" and not c.startswith("sh.") else f"sz.{c}" if not c.startswith(("sh.", "sz.")) else c
        score = float(row["fund_score"])
        if score > 0: cache.setdefault(c, {})[str(row["end_date"])] = score
    log.info(f"✅ scores: {len(df)}行, {time.time()-t0:.1f}s")
    return cache


def get_fs(code, ds, sc):
    if code not in sc: return FUND_DEF
    for d in reversed(sorted(sc[code].keys())):
        if d <= ds: return sc[code][d]
    return FUND_DEF


class EnvClassifier:
    """市场环境分类器 (与 v2.0 一致)"""
    def __init__(self): self._c = {}
    def classify(self, ds):
        if ds in self._c: return self._c[ds]
        self._c[ds] = "OSCILLATE"  # 默认震荡市 (无指数数据)
        return self._c[ds]
    def get_mult(self, ds): return ENV_MULT[self.classify(ds)]


def run_v22(data: dict, scores: dict, max_stocks: int = 0, label: str = "v2.2") -> dict:
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion")/"config.yaml"))
    te = TechSignalEngine(cfg.get("tech", {}))

    ad = sorted(data.keys())
    wd = [d for d in ad if d < "2020-01-02"]
    bd = [d for d in ad if "2020-01-02" <= d <= "2026-04-29"]
    ss = sorted(set().union(*(d.keys() for d in data.values())))
    if max_stocks > 0: ss = ss[:max_stocks]
    log.info(f"📊 {len(ss)}只 预热{len(wd)}d 回测{len(bd)}d")

    log.info("Warm-up...")
    for i, ds in enumerate(wd):
        dd = data.get(ds, {})
        for s in ss:
            if s in dd:
                try: te.update(s, dd[s], False)
                except: pass
        if (i+1)%500 == 0: log.info(f"  {ds}({i+1}/{len(wd)})")
    log.info(f"✅ Warmup: {len(te.chan_cache)}")

    env_cls = EnvClassifier()
    cash = INITIAL; pos = {}; eq = []; cool = {}; tr = []; vc = {}
    sig_base = {"hard_divergence": 0.85, "third_point_buy": 0.70, "second_class_buy": 0.65, "soft_divergence": 0.50}
    sig_stats = defaultdict(int)

    log.info("Backtest...")
    for di, ds in enumerate(bd):
        td = date.fromisoformat(ds); dd = data.get(ds, {})
        if not dd: continue
        for s in ss:
            if s in dd:
                try: te.update(s, dd[s], True)
                except: pass
                if dd[s].get("volume",0) > 0:
                    vc.setdefault(s,[]).append(dd[s]["volume"])
                    if len(vc[s])>50: vc[s]=vc[s][-50:]

        env = env_cls.classify(ds)
        env_m = env_cls.get_mult(ds)

        for sig in te.get_confirmed_signals(td):
            if len(pos) >= MAX_POS or sig.symbol in pos: continue
            if sig.symbol in cool and td <= cool[sig.symbol]: continue
            if sig.symbol not in dd: continue

            # 次日开盘
            nd = get_nd(ds, ad)
            if nd is None or sig.symbol not in data.get(nd, {}): continue
            ep = data[nd][sig.symbol]["open"]
            if ep <= 0: continue

            # 基本面 (scores.parquet)
            fs = get_fs(sig.symbol, ds, scores)
            if fs < FUND_MIN: continue

            # 成交量过滤
            vols = vc.get(sig.symbol, [])
            if len(vols) >= 20:
                av = sum(vols[-20:])/20
                ev = dd[sig.symbol].get("volume",0)
                if av > 0 and ev < av * VOL_MIN: continue

            # 因子化 + 共振 + 环境乘数
            vols2 = vc.get(sig.symbol, []); vr = 1.0
            if len(vols2) >= 20:
                mv = sum(vols2[-20:])/20
                vr = dd[sig.symbol].get("volume",0)/mv if mv > 0 else 1.0
            vc2 = min(vr, 2.0)/2.0; base = sig_base.get(sig.signal_subtype, 0.5)
            tf = min(1.0, max(0.0, base + (vc2-0.5)*0.4))
            res = min(1.0, tf*TECH_W + (fs/100.0)*FUND_W) * env_m  # <-- 环境乘数应用

            pc = min(res,1.0)*POS_SIZE/0.20; pc = min(1.0, max(0.0, pc))
            cost = cash*POS_SIZE*pc
            if cost > cash or cost <= 0: continue
            sh = int(cost/ep/100)*100
            if sh <= 0: continue
            cost = sh*ep
            if cost > cash or cost <= 0: continue

            sp = ep*(1-FIXED_SL)
            pos[sig.symbol] = {"e":ep,"s":sh,"d":ds,"bd":nd,"sp":sp,"p":ep,"t1":False,"t2":False,"fs":fs,"res":res,"env":env}
            cash -= cost
            sig_stats[sig.signal_subtype] += 1
            tr.append({"date":ds,"symbol":sig.symbol,"action":"BUY","price":round(ep,3),
                "signal":sig.signal_subtype,"fund_score":fs,"resonance":round(res,3),"env":env})

        # 止盈/止损 (同 v2.1)
        tc, tr2 = [], []
        for sym, p in list(pos.items()):
            pr = dd[sym]["close"] if sym in dd and dd[sym].get("close",0)>0 else p.get("_l",p["e"])
            if sym in dd and dd[sym].get("close",0)>0: p["_l"]=pr
            p["p"] = max(p["p"], pr); pnl = (pr-p["e"])/p["e"]; ppnl = (p["p"]-p["e"])/p["e"]
            hd = (td-date.fromisoformat(p["d"])).days

            if pnl >= TP_BREAK and p["sp"] < p["e"]: p["sp"] = max(p["sp"], p["e"])
            if hd >= 60 and abs(pnl) < 0.05: p["sp"] = min(p["sp"], p["e"]*0.85)
            
            low = dd.get(sym, {}).get("low", pr)
            if low <= p["sp"]:
                tc.append((sym,"sl",pr,"close")); continue
            if ppnl >= TP_BREAK:
                if pr/p["p"] <= TP_TRAIL:
                    nd2 = get_nd(ds, ad); sp2 = data[nd2][sym]["open"] if nd2 and sym in data.get(nd2,{}) else pr
                    tc.append((sym,"ts",sp2,"next_open")); continue
            if pnl >= TP2 and not p["t2"]: p["t2"]=True; nd2=get_nd(ds,ad); sp2=data[nd2][sym]["open"] if nd2 and sym in data.get(nd2,{}) else pr; tr2.append((sym,"tp2",TP2R,sp2,"next_open")); continue
            if pnl >= TP1 and not p["t1"]: p["t1"]=True; nd2=get_nd(ds,ad); sp2=data[nd2][sym]["open"] if nd2 and sym in data.get(nd2,{}) else pr; tr2.append((sym,"tp1",TP1R,sp2,"next_open")); continue

        for sym,reas,rp,p2,pt in tr2:
            rs=int(pos[sym]["s"]*rp)
            if rs>0 and pos[sym]["s"]>rs:
                pos[sym]["s"]-=rs; cash+=rs*p2
                tr.append({"date":ds,"symbol":sym,"action":"SELL","reason":reas,"price":round(p2,3),
                    "pnl":round((p2/pos[sym]["e"]-1)*100,2),"sell_price_type":pt})
        for sym,reas,p2,pt in tc:
            if sym not in pos: continue
            p=pos.pop(sym); cash+=p["s"]*p2
            tr.append({"date":ds,"symbol":sym,"action":"SELL","reason":reas,"price":round(p2,3),
                "pnl":round((p2/p["e"]-1)*100,2),"hd":(td-date.fromisoformat(p["d"])).days,"sell_price_type":pt})
            if reas=="sl" and sum(1 for t in reversed(tr) if t.get("symbol")==sym and t.get("reason")=="sl")>=COOL_S:
                cool[sym]=td+timedelta(days=COOL_D)

        pv=sum(p["s"]*(dd.get(s,{}).get("close",p.get("_l",p["e"]))) for s,p in pos.items())
        equity=cash+pv; pe=max(eq[-1]["nav"] if eq else cash, equity)
        eq.append({"date":ds,"nav":round(equity,2),"dd":round((equity-pe)/pe*100,2)})
        if (di+1)%200==0: log.info(f"  [{ds}]({di+1}/{len(bd)}) NAV={equity:,.0f} pos={len(pos)} t={len(tr)}")

    for s,p in list(pos.items()):
        cp=p.get("_l",p["e"]); cash+=p["s"]*cp
        tr.append({"date":bd[-1],"symbol":s,"action":"SELL","reason":"fc","price":round(cp,3),
            "pnl":round((cp/p["e"]-1)*100,2)})

    # 指标
    tr_pct=(eq[-1]["nav"]/INITIAL-1)*100
    days=len(eq); ar=(1+tr_pct/100)**(252/days)-1 if days>0 else 0
    ea=np.array([e["nav"] for e in eq]); dr=np.diff(ea)/ea[:-1]; ex=dr-0.03/252
    sh=np.mean(ex)/np.std(ex)*np.sqrt(252) if np.std(ex)>1e-8 else 0
    mdd=np.min((ea-np.maximum.accumulate(ea))/np.maximum.accumulate(ea))*100
    sl=[t for t in tr if t["action"]=="SELL" and t.get("reason")!="fc"]
    wn=[t for t in sl if t.get("pnl",0)>0]; ls=[t for t in sl if t.get("pnl",0)<=0]
    nt=len(sl); wr=len(wn)/nt*100 if nt>0 else 0
    aw=np.mean([t["pnl"] for t in wn]) if wn else 0; al=np.mean([abs(t["pnl"]) for t in ls]) if ls else 0

    return {"tr":round(tr_pct,2),"ar":round(ar*100,2),"sh":round(sh,4),"mdd":round(mdd,2),
            "wr":round(wr,1),"wl":round(aw/al,2) if al>0 else 0,"nt":nt,
            "trades":tr,"equity":eq,"sig_stats":dict(sig_stats)}


def main():
    log.info("="*60)
    log.info("🔰 Task 5: v2.2 环境乘数优化回测")
    log.info("   ENV乘数: 牛1.0 / 震荡1.0 / 熊0.6")
    log.info("="*60)

    scores = load_scores()
    if not scores: log.error("❌ scores.parquet 未加载"); return

    # 先跑 500 只快速验证
    log.info("--- 快速验证 (500只) ---")
    data = load_data(max_stocks=500)
    r = run_v22(data, scores, max_stocks=500, label="v2.2_quick")

    log.info(f"\n📊 v2.2 快速(500只) 结果:")
    log.info(f"   收益{r['tr']}% 年化{r['ar']}% 夏普{r['sh']} 回撤{r['mdd']}% 胜率{r['wr']}% 交易{r['nt']}")
    log.info(f"   盈亏比{r['wl']} | 信号: {r['sig_stats']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(r["trades"]).to_csv(REPORT_DIR/f"task5_quick_{ts}_trades.csv",index=False)
    pd.DataFrame(r["equity"]).to_csv(REPORT_DIR/f"task5_quick_{ts}_equity.csv",index=False)
    json.dump({k:v for k,v in r.items() if k in ("tr","ar","sh","mdd","wr","wl","nt","sig_stats")},
              open(REPORT_DIR/f"task5_quick_{ts}_summary.json","w"),indent=2)

    # 启动全量
    log.info("\n--- 启动全量回测 (4441只, 后台) ---")
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    nohup_log = f"/tmp/task5_full_{ts2}.log"

    # Write the full run script
    full_script = f"""#!/usr/bin/env python3
import sys; sys.path.insert(0, '/home/quant/.openclaw/workspace')
import time, json, gc, logging
from pathlib import Path; from datetime import datetime
import numpy as np; import pandas as pd

# Re-import
from scripts.task5_v22_backtest import load_data, load_scores, run_v22, REPORT_DIR, log

logging.getLogger("task5_v22").setLevel(logging.INFO)

sc = load_scores()
data = load_data()  # 全量
r = run_v22(data, sc)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
pd.DataFrame(r["trades"]).to_csv(REPORT_DIR/f"task5_full_{{ts}}_trades.csv",index=False)
pd.DataFrame(r["equity"]).to_csv(REPORT_DIR/f"task5_full_{{ts}}_equity.csv",index=False)
json.dump({{k:v for k,v in r.items() if k in ("tr","ar","sh","mdd","wr","wl","nt","sig_stats")}},
          open(REPORT_DIR/f"task5_full_{{ts}}_summary.json","w"),indent=2)
print(f"✅ v2.2全量完成: {{r}}")
"""
    script_path = f"/tmp/run_v22_full_{ts2}.py"
    with open(script_path, "w") as f: f.write(full_script)

    import subprocess
    with open(nohup_log, "w") as flog:
        p = subprocess.Popen(["python3", script_path], stdout=flog, stderr=flog, start_new_session=True)
    log.info(f"✅ v2.2全量已后台启动 (PID={p.pid}), 预计30-40分钟")
    log.info(f"   日志: {nohup_log}")
    log.info(f"\n{'='*55}")
    log.info(f"   v2.2 快速(500只): 收益{r['tr']}% 夏普{r['sh']} 回撤{r['mdd']}%")
    log.info(f"   v2.1 全量(4441只): 收益28.05% 夏普0.18 回撤-8.25%")
    log.info(f"   v2.0 原始:         收益8.27% 夏普-0.16 回撤-11.23%")
    elapsed = time.time()-_T_START
    log.info(f"⏱️ 总耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
