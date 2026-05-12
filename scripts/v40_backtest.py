#!/usr/bin/env python3
"""
v4.0 — v3.1 + 模拟器精度提升 (Round 1)
============================================
改动：
  1. 滑点：买入×1.001，卖出×0.999
  2. 涨跌停限制：涨停不可买，跌停不可卖
  3. 成交量限制：单笔≤日成交量5%
  4. 其余完全同 v3.1
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

sys.path.insert(0, '/home/quant/.openclaw/workspace')

OUTPUT_DIR = Path("/home/quant/.openclaw/workspace/output/v40")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[
        logging.StreamHandler(), logging.FileHandler(str(OUTPUT_DIR / "v40_run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("v40")

from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, calc_metrics,
    get_fund_score_v21, resonance_score_v21, get_next_trade_date,
    INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE,
    COOLING_PERIOD_DAYS, COOLING_STOP_COUNT, VOL_MIN_RATIO,
    TP_BREAKEVEN_LOCK,
)

# ===== v4.0 新增参数：模拟器精度 =====
SLIPPAGE_BUY = 1.001      # 买入滑点 0.1%
SLIPPAGE_SELL = 0.999     # 卖出滑点 0.1%
VOL_CAP_RATIO = 0.05      # 单笔不超过日成交量的 5%

FIXED_STOP_LOSS = 0.08
FUND_SCORE_MIN = 30.0

def load_csi300():
    import pandas as pd
    df = pd.read_csv("/home/quant/.openclaw/workspace/data/index_sh000300.csv")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return {r['date']: (r['open'], r['high'], r['low'], r['close']) for _, r in df.iterrows()}

# 环境分类器（同v3.1）
class CSI300EnvClassifier:
    def __init__(self):
        self.dates, self.closes, self.highs, self.lows = [], [], [], []
        self.current_env = "OSCILLATE"; self.daily_log = []
    def update(self, ds, ohlc):
        o,h,l,c = ohlc
        if c<=0: return
        self.dates.append(ds); self.closes.append(c); self.highs.append(h); self.lows.append(l)
        if len(self.closes)<120: return
        ma20=np.mean(self.closes[-20:]); ma60=np.mean(self.closes[-60:]); ma120=np.mean(self.closes[-120:])
        bull=ma20>ma60>ma120; bear=ma20<ma60<ma120; adx=self._adx(14); st=adx>22
        atr_ratio=self._atr(20)/c
        if len(self.closes)>=200:
            hh=[np.mean([max(self.highs[j]-self.lows[j],abs(self.highs[j]-self.closes[j-1]),abs(self.lows[j]-self.closes[j-1])) for j in range(i-20,i)])/self.closes[i-1] for i in range(20,min(120,len(self.closes)-1)+20)]
            p85,p30=np.percentile(hh,85),np.percentile(hh,30); hv, lv = atr_ratio>p85, atr_ratio<p30
        else: hv, lv = False, False
        if bull and st and not hv: self.current_env="BULL"
        elif bear and st and not hv: self.current_env="BEAR"
        elif hv or (not st and not lv): self.current_env="OSCILLATE"
        else: self.current_env="BULL" if bull else "BEAR"
    def _adx(self,p=14):
        n=len(self.closes); lb=min(p*3,n-1)
        if n<p*2+2: return 0.0
        tr,pd,nd=[],[],[]
        for i in range(n-lb,n):
            hl=self.highs[i]-self.lows[i]; hc=abs(self.highs[i]-self.closes[i-1]); lc=abs(self.lows[i]-self.closes[i-1])
            tr.append(max(hl,hc,lc))
            up=self.highs[i]-self.highs[i-1]; down=self.lows[i-1]-self.lows[i]
            pd.append(up if up>down and up>0 else 0.0); nd.append(down if down>up and down>0 else 0.0)
        tp=np.mean(tr[-p:]) if len(tr)>=p else np.mean(tr); dp=100*np.mean(pd[-p:])/tp if tp>0 else 0; dn=100*np.mean(nd[-p:])/tp if tp>0 else 0
        return 100*abs(dp-dn)/(dp+dn) if dp+dn>0 else 0.0
    def _atr(self,p=20):
        return np.mean([max(self.highs[i]-self.lows[i],abs(self.highs[i]-self.closes[i-1]),abs(self.lows[i]-self.closes[i-1])) for i in range(-p,0)])

class MAEnvClassifier:
    def __init__(self):
        self.prices=[]; self.current_env="OSCILLATE"
    def update(self,price):
        if price<=0: return
        self.prices.append(price)
        if len(self.prices)<60: return
        if len(self.prices)>=21:
            rets=[(self.prices[i]-self.prices[i-1])/self.prices[i-1] for i in range(-20,0)]
            if np.std(rets)*np.sqrt(252)>0.35: return
        ma20=sum(self.prices[-20:])/20; ma60=sum(self.prices[-60:])/60
        self.current_env="BULL" if ma20>ma60 else "BEAR"

def get_fund_score(qlib_code,ds,cache):
    sc=cache.get(qlib_code)
    if not sc: return None
    for d in reversed(sorted(sc.keys())):
        if d<=ds: return sc[d]
    return None

def calc_stock_atr(sd):
    if len(sd)<21: return 0.02
    r=sd[-(21):]; tr=[max(r[i][1]-r[i][2],abs(r[i][1]-r[i-1][3]),abs(r[i][2]-r[i-1][3])) for i in range(1,len(r))]
    return np.mean(tr)/r[-1][3] if r[-1][3]>0 else 0.02

def get_env_params(env):
    if env=="BULL": return {"tp1":0.12,"tp2":0.18,"trail":0.88,"env_stop":None}
    if env=="OSCILLATE": return {"tp1":0.06,"tp2":0.10,"trail":0.94,"env_stop":None}
    return {"tp1":0.03,"tp2":0.05,"trail":None,"env_stop":0.03}

def run_v40():
    t0=time.time()
    logger.info(f"{'='*55}")
    logger.info(f"🔰 v4.0 — 滑点+涨跌停+成交量限制")
    logger.info(f"  滑点: 买×1.001 卖×0.999 | 涨跌停检查 | 单笔≤日成交量5%")
    logger.info(f"{'='*55}")

    data=load_data(max_stocks=0); all_dates=sorted(data.keys()); scores_cache=load_scores_parquet()

    cutoff="2020-01-01"; vol_ranking=defaultdict(list)
    for ds in all_dates:
        if ds>=cutoff: break
        for sym,bar in data.get(ds,{}).items():
            if bar.get("volume",0)>0: vol_ranking[sym].append(bar["volume"])
    avg_vols={sym:np.mean(vs[-250:]) for sym,vs in vol_ranking.items() if len(vs)>=20}
    top_stocks=set(sorted(avg_vols,key=avg_vols.get,reverse=True)[:800])
    logger.info(f"  筛选: {len(top_stocks)}只 | {all_dates[0]} ~ {all_dates[-1]}")

    csi300=load_csi300()
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg=load_config(str(Path("/home/quant/.openclaw/workspace/chanfund_fusion/config.yaml")))
    tech_engine=TechSignalEngine(cfg.get("tech",{}))
    warmup_dates=[d for d in all_dates if d<cutoff]; bt_dates=[d for d in all_dates if cutoff<=d<="2026-12-31"]
    csi_env=CSI300EnvClassifier(); old_env=MAEnvClassifier()

    logger.info("Phase 1: Warm-up")
    for i,ds in enumerate(warmup_dates):
        if ds in csi300: csi_env.update(ds,csi300[ds]); old_env.update(csi300[ds][3])
        for stock in top_stocks:
            if stock in data.get(ds,{}):
                try: tech_engine.update(stock,data[ds][stock],False)
                except: pass
        if (i+1)%500==0: logger.info(f"  Warmup {ds} ({i+1}/{len(warmup_dates)})")
    logger.info(f"✅ 预热完成, {len(tech_engine.chan_cache)}只")

    cash,peak_equity=INITIAL_CAPITAL,INITIAL_CAPITAL
    positions,equity_curve,cooling,trade_log={},[],{},[]
    vol_cache,sig_stats,exit_stats,tp_stats={},{},{},{}
    env_counts,old_env_counts={},{}
    score_stats={"used":0,"default":0,"rejected":0,"passed":0}
    stock_history=defaultdict(list)

    logger.info("Phase 2: Backtest v4.0...")
    for di,ds in enumerate(bt_dates):
        tdt=date.fromisoformat(ds)
        if ds in csi300: csi_env.update(ds,csi300[ds]); old_env.update(csi300[ds][3]); csi_env.daily_log.append(csi_env.current_env)
        env=csi_env.current_env
        old_env_counts[old_env.current_env]=old_env_counts.get(old_env.current_env,0)+1
        env_counts[env]=env_counts.get(env,0)+1
        ep=get_env_params(env)
        day_data=data.get(ds,{})
        if not day_data: continue

        for stock in top_stocks:
            if stock in day_data:
                try: tech_engine.update(stock,day_data[stock],True)
                except: pass
                b=day_data[stock]
                stock_history[stock].append((ds,b.get("high",0),b.get("low",0),b.get("close",0)))
                if len(stock_history[stock])>250: stock_history[stock]=stock_history[stock][-250:]
                if b.get("volume",0)>0:
                    vol_cache.setdefault(stock,[]).append(b["volume"])
                    if len(vol_cache[stock])>50: vol_cache[stock]=vol_cache[stock][-50:]

        signals=tech_engine.get_confirmed_signals(tdt)
        for sig in signals:
            if len(positions)>=MAX_POSITIONS: break
            if sig.symbol in positions or sig.symbol not in top_stocks: continue
            if sig.symbol in cooling and tdt<=cooling[sig.symbol]: continue
            if sig.symbol not in day_data: continue

            nd=get_next_trade_date(ds,all_dates)
            if nd is None or sig.symbol not in data.get(nd,{}): continue
            entry_price=data[nd][sig.symbol]["open"]
            if entry_price<=0: continue

            # ======== v4.0: 涨跌停限制 ========
            prev_close=data.get(ds,{}).get(sig.symbol,{}).get("close",0)
            if prev_close>0:
                limit_up=round(prev_close*1.10,2); limit_down=round(prev_close*0.90,2)
                if entry_price>=limit_up: continue  # 涨停买不到
                if entry_price<=limit_down: continue  # 跌停买不了

            # ======== v4.0: 滑点 ========
            exec_price=entry_price*SLIPPAGE_BUY

            fund_score=get_fund_score(sig.symbol,ds,scores_cache)
            if fund_score is None: fund_score=50.0; score_stats["default"]+=1
            else: score_stats["used"]+=1
            if fund_score<FUND_SCORE_MIN: score_stats["rejected"]+=1; continue
            score_stats["passed"]+=1

            vols=vol_cache.get(sig.symbol,[])
            if len(vols)>=20:
                av=sum(vols[-20:])/20
                if av>0 and day_data[sig.symbol].get("volume",0)<av*VOL_MIN_RATIO: continue

            base_signal={"third_point_buy":0.6,"hard_divergence":1.0}.get(sig.signal_subtype,0.5)
            stock_atr=calc_stock_atr(stock_history[sig.symbol]); stock_atr=max(stock_atr,0.005)
            vol_fc=base_signal/max(stock_atr,0.01); fund_m=fund_score/100.0
            raw_amt=(cash*0.04*vol_fc*fund_m)/max(stock_atr,0.01)
            raw_amt=min(raw_amt,cash*0.40)

            # ======== v4.0: 成交量限制 ========
            daily_vol=data.get(nd,{}).get(sig.symbol,{}).get("volume",0)
            if daily_vol>0:
                max_shares_by_vol=int(daily_vol*VOL_CAP_RATIO/100)*100
            else: max_shares_by_vol=9999999

            shares=int(raw_amt/exec_price/100)*100
            shares=min(shares,max_shares_by_vol)
            if shares<=0: continue
            cost=shares*exec_price
            if cost>cash or cost<=0: continue

            positions[sig.symbol]={
                "entry_price":entry_price,"buy_price":round(exec_price,3),"shares":shares,
                "entry_date":ds,"buy_exec_date":nd,
                "stop_price":entry_price*(1-FIXED_STOP_LOSS),
                "peak_price":entry_price,"signal_type":sig.signal_subtype,
                "tp1_done":False,"tp2_done":False,"fund_score":fund_score,"env":env,
            }
            cash-=cost
            sig_stats[sig.signal_subtype]=sig_stats.get(sig.signal_subtype,0)+1
            trade_log.append({"date":ds,"symbol":sig.symbol,"action":"BUY",
                "price":round(exec_price,3),"entry_price":round(entry_price,3),
                "signal":sig.signal_subtype,"fund_score":fund_score,"env":env,
                "slippage":round(exec_price-entry_price,3)})

        # C. 止盈止损（确认价用滑点）
        to_close,to_reduce=[],[]
        for stock,pos in list(positions.items()):
            raw_price=day_data[stock]["close"] if stock in day_data and day_data[stock].get("close",0)>0 else pos.get("_last_price",pos["entry_price"])
            pos["_last_price"]=raw_price
            price=raw_price*SLIPPAGE_SELL  # 卖出滑点
            pos["peak_price"]=max(pos["peak_price"],raw_price)
            pnl=(price-pos["entry_price"])/pos["entry_price"]
            peak_pnl=(pos["peak_price"]-pos["entry_price"])/pos["entry_price"]
            hold=(tdt-date.fromisoformat(pos["entry_date"])).days

            if pnl>=TP_BREAKEVEN_LOCK and pos["stop_price"]<pos["entry_price"]: pos["stop_price"]=max(pos["stop_price"],pos["entry_price"])
            if hold>=60 and abs(pnl)<0.05: pos["stop_price"]=min(pos["stop_price"],pos["entry_price"]*0.85)
            if ep["env_stop"]:
                es=pos["entry_price"]*(1-ep["env_stop"])
                if day_data.get(stock,{}).get("low",raw_price)<=es: to_close.append((stock,"env_stop_loss",price,"close")); continue
            if day_data.get(stock,{}).get("low",raw_price)<=pos["stop_price"]: to_close.append((stock,"stop_loss",price,"close")); continue
            if ep["trail"] and peak_pnl>=TP_BREAKEVEN_LOCK and raw_price/pos["peak_price"]<=ep["trail"]:
                nd2=get_next_trade_date(ds,all_dates); sp=data[nd2][stock]["open"]*SLIPPAGE_SELL if nd2 and stock in data.get(nd2,{}) else price
                to_close.append((stock,"trailing_stop",sp,"next_open")); tp_stats["trailing_stop"]=tp_stats.get("trailing_stop",0)+1; continue
            if pnl>=ep["tp2"]:
                pos["tp2_done"]=True; nd2=get_next_trade_date(ds,all_dates); sp=data[nd2][stock]["open"]*SLIPPAGE_SELL if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock,"tp2",0.50,sp,"next_open")); tp_stats["tp2"]=tp_stats.get("tp2",0)+1; continue
            if pnl>=ep["tp1"]:
                pos["tp1_done"]=True; nd2=get_next_trade_date(ds,all_dates); sp=data[nd2][stock]["open"]*SLIPPAGE_SELL if nd2 and stock in data.get(nd2,{}) else price
                to_reduce.append((stock,"tp1",0.30,sp,"next_open")); tp_stats["tp1"]=tp_stats.get("tp1",0)+1; continue

        for sym,reason,rpct,sp,_ in to_reduce:
            p=positions[sym]; rs=int(p["shares"]*rpct)
            if rs>0 and p["shares"]>rs: p["shares"]-=rs; cash+=rs*sp; exit_stats[f"reduce_{reason}"]=exit_stats.get(f"reduce_{reason}",0)+1
        for sym,reason,price,_ in to_close:
            p=positions.pop(sym,None)
            if not p: continue
            pnl=(price-p["entry_price"])/p["entry_price"]*100; cash+=p["shares"]*price
            exit_stats[reason]=exit_stats.get(reason,0)+1
            trade_log.append({"date":ds,"symbol":sym,"action":"SELL","price":round(price,3),
                "pnl_pct":round(pnl,2),"reason":reason,"hold_days":hold,
                "fund_score":p["fund_score"],"env":p.get("env",env),
                "slippage":round(price-p["entry_price"],3)})
            if reason=="stop_loss": cooling[sym]=tdt+timedelta(days=COOLING_PERIOD_DAYS)

        equity=cash+sum(p["shares"]*day_data.get(sym,{}).get("close",p["entry_price"]) for sym,p in positions.items())
        peak_equity=max(peak_equity,equity)
        equity_curve.append({"date":ds,"nav":round(equity,2),"dd":round((equity/peak_equity-1)*100,2)})
        if (di+1)%200==0 or di==len(bt_dates)-1:
            logger.info(f"  [{ds}]({di+1}/{len(bt_dates)}) NAV={equity:,.0f} pos={len(positions)} t={len(trade_log)} env={env}")

    pd=__import__('pandas')
    metrics=calc_metrics(trade_log,equity_curve,INITIAL_CAPITAL)
    buys=[t for t in trade_log if t["action"]=="BUY"]; sells=[t for t in trade_log if t["action"]=="SELL"]
    wins=[t for t in sells if t.get("pnl_pct",0)>0]; wr=len(wins)/max(len(sells),1)*100
    pf=sum(t["pnl_pct"] for t in wins)/max(sum(abs(t["pnl_pct"]) for t in sells if t.get("pnl_pct",0)<=0),1)

    logger.info(f"\n{'='*55}")
    logger.info(f"📊 v4.0 全量结果")
    logger.info(f"{'='*55}")
    logger.info(f"  总收益率: {metrics['total_return']:.2f}%")
    logger.info(f"  年化收益率: {metrics['ann_return']:.2f}%")
    logger.info(f"  夏普比率: {metrics['sharpe']:.4f}")
    logger.info(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    logger.info(f"  胜率: {wr:.1f}%  盈亏比: {metrics.get('win_loss_ratio',pf):.2f}")
    logger.info(f"  交易: {len(trade_log)}(买入{len(buys)}/卖出{len(sells)})")
    logger.info(f"  信号: {sig_stats}  平仓: {exit_stats}")
    logger.info(f"  环境: {dict(sorted(env_counts.items()))}")
    logger.info(f"  scores: {score_stats}")
    logger.info(f"  v4.0参数: 滑点买×{SLIPPAGE_BUY}卖×{SLIPPAGE_SELL} 涨跌停限制 单笔≤{VOL_CAP_RATIO*100}%日量")

    pd.DataFrame(trade_log).to_csv(OUTPUT_DIR/"v40_trades.csv",index=False)
    pd.DataFrame(equity_curve).to_csv(OUTPUT_DIR/"v40_equity.csv",index=False)
    ms={k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}
    ms.update({"win_rate":round(wr,1),"profit_factor":round(pf,2)})
    with open(OUTPUT_DIR/"v40_summary.json","w") as f: json.dump(ms,f,indent=2,ensure_ascii=False)
    logger.info(f"\n⏱️ {time.time()-t0:.0f}s 💾 {OUTPUT_DIR}")

if __name__=="__main__":
    run_v40()
