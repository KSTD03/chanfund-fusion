#!/usr/bin/env python3
"""
T2 + 动量过滤 — 融合版
========================
T2 同花顺行业环境分类器 (无 env_stop)
+ v6ab 动量排名过滤 (拒绝底部15%)
+ 完整逐笔记录和分析

回测: 2010-07-01 → 2026-04-30
"""

import sys, time, json, os, numpy as np
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

WORKSPACE = "/home/quant/.openclaw/workspace"
sys.path.insert(0, WORKSPACE)
OUT_DIR = Path(WORKSPACE) / "output" / "t2_momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(),
        logging.FileHandler(str(OUT_DIR / "run.log"), mode="w", encoding="utf-8")])
log = logging.getLogger("t2m")

import pickle, pandas as pd
from scripts.task3_v21_backtest import (
    load_data, load_scores_parquet, get_next_trade_date,
    get_fund_score_v21 as get_fund_score, calc_metrics,
    INITIAL_CAPITAL, MAX_POSITIONS, COOLING_PERIOD_DAYS, TP_BREAKEVEN_LOCK, VOL_MIN_RATIO,
)

# ═══════════════ 参数
SLIPPAGE_BUY, SLIPPAGE_SELL = 1.001, 0.999
VOL_CAP_RATIO, FIXED_STOP_LOSS = 0.05, 0.08
FUND_SCORE_MIN = 30.0

MOMENTUM_WINDOW = 20
MOMENTUM_PERCENTILE = 15
MOMENTUM_UPDATE_FREQ = 20

def buy_cost(sh, p): return sh * p * SLIPPAGE_BUY * 1.0003
def sell_net(sh, p): g = sh * p * SLIPPAGE_SELL; return g - g * 0.0003 - g * 0.001

# 同花顺环境参数 (T2继承: 无env_stop for BULL/OSC)
def env_p(env):
    if env == "BULL": return {"tp1":0.12,"tp2":0.18,"trail":0.88,"env_stop":None}
    if env == "OSCILLATE": return {"tp1":0.06,"tp2":0.10,"trail":0.94,"env_stop":None}
    return {"tp1":0.03,"tp2":0.05,"trail":None,"env_stop":0.03}

def calc_atr(sd):
    if len(sd)<21: return 0.02
    r=sd[-21:];tr=[max(r[i][1]-r[i][2],abs(r[i][1]-r[i-1][3]),abs(r[i][2]-r[i-1][3])) for i in range(1,len(r))]
    return np.mean(tr)/r[-1][3] if r[-1][3]>0 else 0.02

def to_qlib(c):
    if c.endswith('.SH'): return 'sh.'+c[:-3]
    if c.endswith('.SZ'): return 'sz.'+c[:-3]
    if c.endswith('.BJ'): return 'bj.'+c[:-3]
    return c

# ═══════════════ 动量排名
def compute_momentum_ranking(daily_close, top_stocks):
    momentum = {}
    for stock in top_stocks:
        prices = daily_close.get(stock, [])
        if len(prices) >= MOMENTUM_WINDOW + 1:
            ret = (prices[-1] - prices[-MOMENTUM_WINDOW-1])/max(prices[-MOMENTUM_WINDOW-1], 0.01)
            momentum[stock] = ret
        else: momentum[stock] = 0.0
    if not momentum: return {}, {}
    values = sorted(momentum.values())
    n = len(values)
    return {s: (sum(1 for v in values if v<v0)/max(n,1))*100 for s,v0 in momentum.items()}, momentum

# ═══════════════ THS 行业环境
def load_ths_env():
    log.info("加载同花顺行业数据...")
    ths_mem = pd.read_parquet(str(Path(WORKSPACE)/"output/t2_ths_env/ths_members.parquet"))
    ths_daily = pd.read_parquet(str(Path(WORKSPACE)/"data/tushare_sector/ths_daily.parquet"))
    idx_codes = set(ths_mem['ths_code'].unique())
    log.info(f"  THS行业: {len(idx_codes)} 个")
    
    stock_to_ths = {}
    for _, r in ths_mem.iterrows():
        qc = to_qlib(r['con_code'])
        if qc not in stock_to_ths: stock_to_ths[qc] = r['ths_code']
    log.info(f"  映射: {len(stock_to_ths)} 只")
    
    env_map = {}
    valid = 0
    for code in idx_codes:
        sub = ths_daily[ths_daily['ts_code']==code].sort_values('trade_date')
        if len(sub)<200: continue
        closes = []; envs = {}
        for _, r in sub.iterrows():
            dt = r['trade_date']; ds = f"{dt[:4]}-{dt[4:6]}-{dt[6:]}"
            closes.append(r['close'])
            if len(closes)<120: envs[ds]="OSCILLATE"; continue
            m20=np.mean(closes[-20:]); m60=np.mean(closes[-60:]); m120=np.mean(closes[-120:])
            envs[ds]="BULL" if m20>m60>m120 else ("BEAR" if m20<m60<m120 else "OSCILLATE")
        env_map[code]=envs; valid+=1
    log.info(f"  ✓ {valid} 个行业指数完成")
    return env_map, stock_to_ths

def get_ths_env(stock, ds, env_map, stock_to_ths):
    code = stock_to_ths.get(stock)
    if code is None or code not in env_map: return "OSCILLATE"
    return env_map[code].get(ds, "OSCILLATE")

def check_alignment(stock_history, stock, ds, env_map, stock_to_ths):
    scp = stock_history.get(stock, [])
    if len(scp)<60: return True, "i"
    closes = [c[3] for c in scp]
    m20=np.mean(closes[-20:]); m60=np.mean(closes[-60:])
    mb = m20 > m60
    code = stock_to_ths.get(stock)
    if code is None or code not in env_map: return True, "no_ind"
    ie = env_map[code].get(ds, "OSCILLATE")
    if ie=="OSCILLATE": return True, "oa"
    if ie=="BULL": return (True, "bb") if mb else (False, "bd")
    if ie=="BEAR": return (True, "rb") if mb else (False, "rd")
    return True, "u"

# ═══════════════ 主回测
def run():
    t0 = time.time()
    log.info("=" * 55)
    log.info("T2 + 动量过滤 — 融合版")
    log.info(f"  动量拒绝底部{MOMENTUM_PERCENTILE}%")
    log.info("=" * 55)
    
    # THS环境
    ths_env_map, stock_to_ths = load_ths_env()
    
    # 数据
    DATA_CACHE = "/tmp/daily_data_cache.pkl"
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, 'rb') as f: data = pickle.load(f)
    else:
        data = load_data(max_stocks=0)
        with open(DATA_CACHE, 'wb') as f: pickle.dump(data, f)
    all_dates = sorted(data.keys())
    scores_cache = load_scores_parquet()
    
    # 股票池
    vol_dates = all_dates[:min(500, len(all_dates))]
    vr = defaultdict(list)
    for ds in vol_dates:
        for sym, bar in data.get(ds,{}).items():
            if bar.get("volume",0)>0: vr[sym].append(bar["volume"])
    avg_vols = {s:np.mean(v[-250:]) for s,v in vr.items() if len(v)>=20}
    top_stocks = set(sorted(avg_vols, key=avg_vols.get, reverse=True)[:800])
    log.info(f"  股票池: {len(top_stocks)} 只")
    
    bt_start = "2010-07-01"; end_d = "2026-04-30"
    wd = [d for d in all_dates if d < bt_start]
    bd = [d for d in all_dates if bt_start <= d <= end_d]
    log.info(f"  预热: {len(wd)}天 | 回测: {len(bd)}天")
    
    from chanfund_fusion.tech.signal_engine import TechSignalEngine
    from chanfund_fusion.config_schema import load_config
    cfg = load_config(str(Path(WORKSPACE)/"chanfund_fusion/config.yaml"))
    eng = TechSignalEngine(cfg.get("tech",{}))
    
    # 预热
    log.info("预热中...")
    for i, ds in enumerate(wd):
        for stock in top_stocks:
            if stock in data.get(ds,{}):
                try: eng.update(stock, data[ds][stock], False)
                except: pass
    log.info(f"✅ 预热完成 ({time.time()-t0:.0f}s)")
    
    # 回测
    cash = INITIAL_CAPITAL; peak_eq = INITIAL_CAPITAL
    positions = {}; equity = []; cooling = {}
    trades = []; sig_stats = defaultdict(int); exit_stats = defaultdict(int)
    vc = defaultdict(list); sh = defaultdict(list); sc2 = defaultdict(list)
    daily_close = defaultdict(list)
    momentum_pct = {}; mom_last_update = -1
    mom_passed = 0; mom_rejected = 0; align_passed = 0; align_rejected = 0
    env_counts = defaultdict(int); vol_rejected = 0
    
    for di, ds in enumerate(bd):
        tdt = date.fromisoformat(ds)
        dd = data.get(ds, {})
        if not dd: continue
        
        for stock in top_stocks:
            if stock in dd:
                b = dd[stock]
                daily_close[stock].append(b.get("close",0))
                if len(daily_close[stock])>250: daily_close[stock]=daily_close[stock][-250:]
        
        if di - mom_last_update >= MOMENTUM_UPDATE_FREQ:
            momentum_pct, _ = compute_momentum_ranking(daily_close, top_stocks)
            mom_last_update = di
        
        for stock in top_stocks:
            if stock in dd:
                b = dd[stock]
                try: eng.update(stock, b, True)
                except: pass
                sh[stock].append((ds,b.get("high",0),b.get("low",0),b.get("close",0)))
                if len(sh[stock])>250: sh[stock]=sh[stock][-250:]
                if b.get("volume",0)>0:
                    vc[stock].append(b["volume"])
                    if len(vc[stock])>50: vc[stock]=vc[stock][-50:]
                sc2[stock].append(b.get("close",0))
                if len(sc2[stock])>250: sc2[stock]=sc2[stock][-250:]
        
        # 买入
        signals = eng.get_confirmed_signals(tdt)
        for sig in signals:
            if len(positions)>=MAX_POSITIONS: break
            if sig.symbol in positions or sig.symbol not in top_stocks: continue
            if sig.symbol in cooling and tdt<=cooling[sig.symbol]: continue
            if sig.symbol not in dd: continue
            
            cur_env = get_ths_env(sig.symbol, ds, ths_env_map, stock_to_ths)
            a,_ = check_alignment(sh, sig.symbol, ds, ths_env_map, stock_to_ths)
            if not a: continue
            
            nd = get_next_trade_date(ds, all_dates)
            if nd is None or sig.symbol not in data.get(nd,{}): continue
            ep_d = data[nd][sig.symbol]["open"]
            if ep_d<=0: continue
            pc = dd.get(sig.symbol,{}).get("close",0)
            if pc>0 and (ep_d>=round(pc*1.10,2) or ep_d<=round(pc*0.90,2)): continue
            
            fund_score = get_fund_score(sig.symbol, ds, scores_cache)
            if fund_score is None: fund_score = 50.0
            if fund_score < FUND_SCORE_MIN: continue
            
            # 动量过滤
            pct = momentum_pct.get(sig.symbol, 50.0)
            if pct < MOMENTUM_PERCENTILE: mom_rejected+=1; continue
            mom_passed+=1
            
            # 量过滤
            vols = vc.get(sig.symbol, [])
            if len(vols)>=20:
                av=sum(vols[-20:])/20
                if av>0 and dd[sig.symbol].get("volume",0)<av*VOL_MIN_RATIO: continue
            
            bs = {"third_point_buy":0.6,"hard_divergence":1.0}.get(sig.signal_subtype,0.5)
            sa = max(calc_atr(sh[sig.symbol]), 0.005)
            vf = bs/max(sa, 0.01); fm = fund_score/100.0
            ra = (cash*0.04*vf*fm)/max(sa, 0.01); ra = min(ra, cash*0.40)
            dv = data.get(nd,{}).get(sig.symbol,{}).get("volume",0)
            ms = int(dv*VOL_CAP_RATIO/100)*100 if dv>0 else 9999999
            sh2 = int(ra/(ep_d*SLIPPAGE_BUY)/100)*100; sh2 = min(sh2, ms)
            if sh2<=0: continue
            ct = buy_cost(sh2, ep_d)
            if ct>cash or ct<=0: continue
            
            positions[sig.symbol] = {"entry_px":ep_d,"exec_px":ep_d*SLIPPAGE_BUY,"shares":sh2,
                "entry_date":ds,"stop_px":ep_d*(1-FIXED_STOP_LOSS),"peak_px":ep_d,
                "tp1":False,"tp2":False,"fs":fund_score,"env":cur_env,"signal":sig.signal_subtype,
                "cost_basis":ct,"pv_at_entry":cash+sum(p3["shares"]*dd.get(s3,{}).get("close",p3["entry_px"]) for s3,p3 in positions.items())}
            cash -= ct; sig_stats[sig.signal_subtype]+=1
            trades.append({"date":ds,"symbol":sig.symbol,"action":"BUY","price":round(ep_d*SLIPPAGE_BUY,3),
                "shares":sh2,"signal":sig.signal_subtype,"fund_score":round(fund_score,1),"ths_env":cur_env,
                "momentum_pct":round(pct,1)})
        
        # 卖出
        tc,tr = [],[]
        for stk,p2 in list(positions.items()):
            if stk not in dd: continue
            rp = dd[stk]["close"] if dd[stk].get("close",0)>0 else p2.get("_lp",p2["entry_px"])
            p2["_lp"]=rp; sp=rp*SLIPPAGE_SELL
            p2["peak_px"]=max(p2["peak_px"],rp)
            pnl_ = (sp-p2["entry_px"])/p2["entry_px"]
            ppnl_ = (p2["peak_px"]-p2["entry_px"])/p2["entry_px"]
            hold = (tdt-date.fromisoformat(p2["entry_date"])).days
            ep = env_p(p2["env"])
            if pnl_>=TP_BREAKEVEN_LOCK and p2["stop_px"]<p2["entry_px"]: p2["stop_px"]=max(p2["stop_px"],p2["entry_px"])
            if hold>=60 and abs(pnl_)<0.05: p2["stop_px"]=min(p2["stop_px"],p2["entry_px"]*0.85)
            
            hit=False
            if ep["env_stop"] and dd.get(stk,{}).get("low",rp)<=p2["entry_px"]*(1-ep["env_stop"]):
                tc.append((stk,"env_stop_loss",sp,rp)); hit=True
            if not hit and dd.get(stk,{}).get("low",rp)<=p2["stop_px"]:
                tc.append((stk,"stop_loss",sp,rp)); hit=True
            if not hit and ep["trail"] and ppnl_>=TP_BREAKEVEN_LOCK and rp/p2["peak_px"]<=ep["trail"]:
                nd2=get_next_trade_date(ds,all_dates); tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
                tc.append((stk,"trailing_stop",tsp,rp)); hit=True
            if not hit and pnl_>=ep["tp2"]:
                p2["tp2"]=True; nd2=get_next_trade_date(ds,all_dates); tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
                tr.append((stk,"tp2",0.50,tsp)); hit=True
            if not hit and pnl_>=ep["tp1"]:
                p2["tp1"]=True; nd2=get_next_trade_date(ds,all_dates); tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
                tr.append((stk,"tp1",0.30,tsp))
        
        for sym,reason,rpct,sp in tr:
            p2=positions[sym]; rs=int(p2["shares"]*rpct)
            if rs<=0 or p2["shares"]<=rs: continue
            net=sell_net(rs,sp/SLIPPAGE_SELL); p2["shares"]-=rs; cash+=net
            exit_stats[f"reduce_{reason}"]+=1
        
        for sym,reason,price,_ in tc:
            p2=positions.pop(sym,None)
            if not p2: continue
            net=sell_net(p2["shares"],price/SLIPPAGE_SELL)
            pnl_pct_ = (price-p2["entry_px"])/p2["entry_px"]*100
            cash+=net; exit_stats[reason]+=1
            trades.append({"date":ds,"symbol":sym,"action":"SELL","price":round(price,3),
                "shares":p2["shares"],"pnl_pct":round(pnl_pct_,2),"reason":reason,"hold_days":hold,
                "fund_score":p2["fs"],"ths_env":p2.get("env","?")})
            if reason=="stop_loss": cooling[sym]=tdt+timedelta(days=COOLING_PERIOD_DAYS)
        
        eq_p = cash+sum(p3["shares"]*dd.get(s3,{}).get("close",p3["entry_px"]) for s3,p3 in positions.items())
        peak_eq = max(peak_eq, eq_p)
        equity.append({"date":ds,"nav":round(eq_p,2),"dd":round((eq_p/peak_eq-1)*100,2)})
        if (di+1)%500==0 or di==len(bd)-1:
            log.info(f"  [{ds}]({di+1}/{len(bd)}) NAV={eq_p:,.0f} pos={len(positions)} trades={len(trades)}")
    
    # 统计分析
    log.info("计算指标...")
    eq_df = pd.DataFrame(equity); eq_df.to_csv(str(OUT_DIR/"equity.csv"),index=False)
    trades_df = pd.DataFrame(trades); trades_df.to_csv(str(OUT_DIR/"trades.csv"),index=False)
    
    rets = eq_df['nav'].pct_change().dropna()
    sharpe = np.mean(rets)/max(np.std(rets),1e-8)*np.sqrt(252)
    md = eq_df['dd'].min()
    tot = (eq_df.iloc[-1]['nav']/INITIAL_CAPITAL-1)*100
    n_years = len(bd)/252
    ann_ret = ((1+tot/100)**(1/max(n_years,1))-1)*100
    
    sells = trades_df[trades_df["action"]=="SELL"]
    wins = sells[sells["pnl_pct"]>0]
    wr = len(wins)/max(len(sells),1)*100
    aw = wins["pnl_pct"].mean() if len(wins)>0 else 0
    al = sells[sells["pnl_pct"]<=0]["pnl_pct"].mean() if len(sells[sells["pnl_pct"]<=0])>0 else 0
    wlr = abs(aw/max(al,0.01)) if al<0 else 0
    
    yearly = {}
    for y in range(2010,2027):
        ys = str(y)
        yeq = eq_df[eq_df["date"].str.startswith(ys)]
        if len(yeq)<10: continue
        yr = (yeq.iloc[-1]["nav"]/yeq.iloc[0]["nav"]-1)*100
        ysel = sells[sells["date"].str.startswith(ys)]
        ywin = ysel[ysel["pnl_pct"]>0]
        yearly[ys] = {"return_pct":round(yr,2),"n_trades":len(ysel),"win_rate":round(len(ywin)/max(len(ysel),1)*100,1),"max_dd":round(yeq["dd"].min(),2)}
    
    exit_analysis = {}
    for reason in sells["reason"].unique():
        sub = sells[sells["reason"]==reason]; pnls = sub["pnl_pct"]
        exit_analysis[reason] = {"count":len(sub),"avg_pnl":round(pnls.mean(),2),
            "avg_hold_days":round(sub["hold_days"].mean(),1),"win_rate":round((pnls>0).sum()/max(len(sub),1)*100,1)}
    
    analysis = {
        "total_return_pct":round(tot,2),"ann_return_pct":round(ann_ret,2),
        "sharpe":round(sharpe,4),"max_drawdown_pct":round(md,2),"win_rate_pct":round(wr,1),
        "n_buys":len(trades_df[trades_df["action"]=="BUY"]),"n_sells":len(sells),
        "profit_loss_ratio":round(wlr,2),"avg_win_pct":round(aw,2),"avg_loss_pct":round(al,2),
        "momentum_passed":mom_passed,"momentum_rejected":mom_rejected,
        "yearly_returns":yearly,"exit_reasons":exit_analysis,"signal_types":dict(sig_stats),
        "backtest_start":bt_start,"backtest_end":end_d,"final_nav":round(eq_df.iloc[-1]["nav"],2),
        "elapsed_seconds":round(time.time()-t0,1),
    }
    with open(str(OUT_DIR/"analysis.json"),"w") as f: json.dump(analysis,f,indent=2,ensure_ascii=False)
    summary = {k:v for k,v in analysis.items() if isinstance(v,(int,float,str))}
    with open(str(OUT_DIR/"summary.json"),"w") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    
    # 打印
    print(f"\n{'='*65}")
    print("📊 T2 + 动量过滤 — 结果")
    print(f"{'='*65}")
    print(f"  夏普: {sharpe:.4f} | 总收益: {tot:+.2f}% | 年化: {ann_ret:+.2f}%")
    print(f"  回撤: {md:.2f}% | 胜率: {wr:.1f}% | 交易: {len(sells)} | 盈亏比: {wlr:.2f}")
    print(f"  动量通过: {mom_passed} 拒绝: {mom_rejected} ({mom_rejected/max(mom_passed+mom_rejected,1)*100:.0f}%)")
    print(f"  终值: {eq_df.iloc[-1]['nav']:,.0f} | 耗时: {time.time()-t0:.0f}s")
    
    print(f"\n{'─'*55}\n逐年收益\n{'─'*55}")
    print(f"{'年份':>6s}  {'收益':>8s}  {'交易':>6s}  {'胜率':>7s}  {'回撤':>8s}")
    for y,yd in sorted(yearly.items()):
        print(f"{y:>6s}  {yd['return_pct']:>+7.2f}%  {yd['n_trades']:>6d}  {yd['win_rate']:>6.1f}%  {yd['max_dd']:>7.2f}%")
    
    print(f"\n{'─'*55}\n退出原因\n{'─'*55}")
    for r,d in sorted(exit_analysis.items(),key=lambda x:-x[1]["count"]):
        p=d["count"]/max(len(sells),1)*100
        print(f"  {r:<20s}  {d['count']:>5d}({p:4.0f}%)  avg={d['avg_pnl']:>+.2f}%  hold={d['avg_hold_days']:.0f}d  wr={d['win_rate']:.0f}%")
    
    log.info(f"✅ T2+动量 完成 ({time.time()-t0:.0f}s) 💾 {OUT_DIR}")

if __name__ == "__main__":
    run()
