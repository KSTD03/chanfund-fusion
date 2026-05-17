#!/usr/bin/env python3
"""
v6.5a-S_czsc-F_thsMmt-C_real-R_r1-minhold3
控制实验: minhold3 + hdOnly + BULL
"""
import sys, time, json, os, numpy as np
from pathlib import Path; from datetime import date, timedelta; from collections import defaultdict
WORKSPACE="/home/quant/.openclaw/workspace"; sys.path.insert(0,WORKSPACE)
OUT_DIR=Path(WORKSPACE)/"output/v6.5a-S_czsc-F_thsMmt-C_real-R_r1-minhold3"
OUT_DIR.mkdir(parents=True,exist_ok=True)
import logging; logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",datefmt="%H:%M:%S",handlers=[logging.StreamHandler(),logging.FileHandler(str(OUT_DIR/"run.log"),mode="w")])
log=logging.getLogger("v65a_r1")
import pickle,pandas as pd; from scripts.task3_v21_backtest import*
DATA_CACHE="/tmp/daily_data_cache.pkl"
if os.path.exists(DATA_CACHE):
    with open(DATA_CACHE,'rb') as f: data=pickle.load(f)
else:
    data=load_data(max_stocks=0)
    with open(DATA_CACHE,'wb') as f: pickle.dump(data,f)
all_dates=sorted(data.keys()); scores_cache=load_scores_parquet()

SLIPPAGE_BUY,SLIPPAGE_SELL=1.001,0.999; VOL_CAP_RATIO,FIXED_STOP_LOSS=0.05,0.08; FUND_SCORE_MIN=30.0
MOMENTUM_WINDOW,MOMENTUM_PERCENTILE,MOMENTUM_UPDATE_FREQ=20,15,20

def sell_net(sh,p): g=sh*p*0.999; return g-g*0.0003-g*0.001

def env_p(e):
    if e=="BULL": return {"tp1":0.12,"tp2":0.18,"trail":0.88,"env_stop":None}
    if e=="OSCILLATE": return {"tp1":0.06,"tp2":0.10,"trail":0.94,"env_stop":None}
    return {"tp1":0.03,"tp2":0.05,"trail":None,"env_stop":0.03}

def calc_atr(sd):
    if len(sd)<21: return 0.02
    r=sd[-21:];tr=[max(r[i][1]-r[i][2],abs(r[i][1]-r[i-1][3]),abs(r[i][2]-r[i-1][3])) for i in range(1,len(r))]
    return np.mean(tr)/r[-1][3] if r[-1][3]>0 else 0.02
def to_qlib(c):
    if c.endswith('.SH'):return 'sh.'+c[:-3]
    if c.endswith('.SZ'):return 'sz.'+c[:-3]
    if c.endswith('.BJ'):return 'bj.'+c[:-3]
    return c

def compute_momentum_ranking(dc,ts):
    m={}; [m.update({s:(p[-1]-p[-MOMENTUM_WINDOW-1])/max(p[-MOMENTUM_WINDOW-1],0.01)}) for s,p in dc.items() if len(p)>=MOMENTUM_WINDOW+1]
    if not m: return {},{}
    v=sorted(m.values()); n=len(v)
    return {s:sum(1 for x in v if x<vo)/max(n,1)*100 for s,vo in m.items()},m

def load_ths():
    ths_mem=pd.read_parquet(str(Path(WORKSPACE)/"output/t2_ths_env/ths_members.parquet"))
    ths_daily=pd.read_parquet(str(Path(WORKSPACE)/"data/tushare_sector/ths_daily.parquet"))
    ic=set(ths_mem['ths_code'].unique())
    s2t={}; [s2t.update({to_qlib(r['con_code']):r['ths_code']}) for _,r in ths_mem.iterrows() if r['ths_code']]
    
    # 用 groupby + agg 代替逐行业循环
    def daily_env(grp):
        grp=grp.sort_values('trade_date')
        if len(grp)<200: return {}
        cl=grp['close'].values
        dates=grp['trade_date'].values
        envs={}
        for i in range(len(cl)):
            ds=f"{dates[i][:4]}-{dates[i][4:6]}-{dates[i][6:]}"
            if i<120: envs[ds]="OSCILLATE"; continue
            m20=np.mean(cl[:i][-20:] if i>=20 else cl[:i]);m60=np.mean(cl[:i][-60:] if i>=60 else cl[:i]);m120=np.mean(cl[:i][-120:] if i>=120 else cl[:i])
            envs[ds]="BULL" if m20>m60>m120 else ("BEAR" if m20<m60<m120 else "OSCILLATE")
        return envs
    
    em={}
    for code, grp in ths_daily[ths_daily['ts_code'].isin(ic)].groupby('ts_code'):
        env=daily_env(grp)
        if env: em[code]=env
    log.info(f"  THS: {len(em)}个行业有效"); return em,s2t

ths_env_map,stock_to_ths=load_ths()
def get_env(s,d): c=stock_to_ths.get(s); return "OSCILLATE" if c is None or c not in ths_env_map else ths_env_map[c].get(d,"OSCILLATE")

vol_dates=all_dates[:min(500,len(all_dates))]
vr=defaultdict(list)
for ds in vol_dates:
    for s,b in data.get(ds,{}).items():
        if b.get("volume",0)>0: vr[s].append(b["volume"])
av={s:np.mean(v[-250:]) for s,v in vr.items() if len(v)>=20}
ts=set(sorted(av,key=av.get,reverse=True)[:800])
log.info(f"  池: {len(ts)}只")

bt_start="2010-07-01"; end_d="2026-04-30"
wd=[d for d in all_dates if d<bt_start]; bd=[d for d in all_dates if bt_start<=d<=end_d]
t0=time.time()
log.info(f"  预热:{len(wd)}天 回测:{len(bd)}天")

from chanfund_fusion.tech.signal_engine import TechSignalEngine; from chanfund_fusion.config_schema import load_config
cfg=load_config(str(Path(WORKSPACE)/"chanfund_fusion/config.yaml"))
eng=TechSignalEngine(cfg.get("tech",{}))
for i,ds in enumerate(wd):
    for s in ts:
        if s in data.get(ds,{}):
            try: eng.update(s,data[ds][s],False)
            except: pass

cash=INITIAL_CAPITAL;peak_eq=INITIAL_CAPITAL;pos={};eq=[];cool={};trades=[];sig_stats=defaultdict(int);exit_stats=defaultdict(int)
vc=defaultdict(list);sh=defaultdict(list);sc2=defaultdict(list);dc=defaultdict(list)
mp={};mlu=-1;mpa=0;mre=0;bep=0;ber=0;ec=defaultdict(int)

for di,ds in enumerate(bd):
    tdt=date.fromisoformat(ds); dd=data.get(ds,{});
    if not dd: continue
    for s in ts:
        if s in dd:
            b=dd[s];dc[s].append(b.get("close",0))
            if len(dc[s])>250: dc[s]=dc[s][-250:]
    if di-mlu>=MOMENTUM_UPDATE_FREQ: mp,_=compute_momentum_ranking(dc,ts);mlu=di
    for s in ts:
        if s in dd:
            b=dd[s]
            try: eng.update(s,b,True)
            except: pass
            sh[s].append((ds,b.get("high",0),b.get("low",0),b.get("close",0)))
            if len(sh[s])>250: sh[s]=sh[s][-250:]
            if b.get("volume",0)>0:
                vc[s].append(b["volume"])
                if len(vc[s])>50: vc[s]=vc[s][-50:]
            sc2[s].append(b.get("close",0))
            if len(sc2[s])>250: sc2[s]=sc2[s][-250:]
    
    signals=eng.get_confirmed_signals(tdt)
    for sig in signals:
        if len(pos)>=MAX_POSITIONS: break
        if sig.symbol in pos or sig.symbol not in ts: continue
        if sig.symbol in cool and tdt<=cool[sig.symbol]: continue
        if sig.symbol not in dd: continue
        cur_env=get_env(sig.symbol,ds)
        
        # ═══ 核心修改: 仅 BULL 开仓 ═══
        if cur_env!="BULL": bep+=1; continue
        if sig.signal_subtype!="hard_divergence": continue
        nd=get_next_trade_date(ds,all_dates)
        if nd is None or sig.symbol not in data.get(nd,{}): continue
        ep_d=data[nd][sig.symbol]["open"]
        if ep_d<=0: continue
        pc=dd.get(sig.symbol,{}).get("close",0)
        if pc>0 and (ep_d>=round(pc*1.10,2) or ep_d<=round(pc*0.90,2)): continue
        fs=get_fund_score_v21(sig.symbol,ds,scores_cache)
        if fs is None: fs=50.0
        if fs<FUND_SCORE_MIN: continue
        pct=mp.get(sig.symbol,50.0)
        if pct<MOMENTUM_PERCENTILE: mre+=1; continue
        mpa+=1
        vols=vc.get(sig.symbol,[])
        if len(vols)>=20:
            av2=sum(vols[-20:])/20
            if av2>0 and dd[sig.symbol].get("volume",0)<av2*VOL_MIN_RATIO: continue
        bs={"third_point_buy":0.6,"hard_divergence":1.0}.get(sig.signal_subtype,0.5)
        sa=max(calc_atr(sh[sig.symbol]),0.005)
        vf=bs/max(sa,0.01);fm=fs/100.0
        ra=(cash*0.04*vf*fm)/max(sa,0.01);ra=min(ra,cash*0.40)
        dv=data.get(nd,{}).get(sig.symbol,{}).get("volume",0)
        ms_=int(dv*VOL_CAP_RATIO/100)*100 if dv>0 else 9999999
        sh2=int(ra/(ep_d*SLIPPAGE_BUY)/100)*100;sh2=min(sh2,ms_)
        if sh2<=0: continue
        ct=sh2*ep_d*SLIPPAGE_BUY*1.0003
        if ct>cash or ct<=0: continue
        pos[sig.symbol]={"entry_px":ep_d,"exec_px":ep_d*SLIPPAGE_BUY,"shares":sh2,
            "entry_date":ds,"stop_px":ep_d*(1-FIXED_STOP_LOSS),"peak_px":ep_d,
            "tp1":False,"tp2":False,"fs":fs,"env":cur_env,"signal":sig.signal_subtype,
            "cost_basis":ct,"pv_at_entry":cash+sum(p3["shares"]*dd.get(s3,{}).get("close",p3["entry_px"]) for s3,p3 in pos.items())}
        cash-=ct;sig_stats[sig.signal_subtype]+=1
        trades.append({"date":ds,"symbol":sig.symbol,"action":"BUY","price":round(ep_d*SLIPPAGE_BUY,3),
            "shares":sh2,"signal":sig.signal_subtype,"fund_score":round(fs,1),"ths_env":cur_env,
            "momentum_pct":round(pct,1)})
    
    tc,tr=[],[]
    for stk,p2 in list(pos.items()):
        if stk not in dd: continue
        rp=dd[stk]["close"] if dd[stk].get("close",0)>0 else p2.get("_lp",p2["entry_px"])
        p2["_lp"]=rp;sp=rp*SLIPPAGE_SELL
        p2["peak_px"]=max(p2["peak_px"],rp)
        pnl_=(sp-p2["entry_px"])/p2["entry_px"]
        ppnl_=(p2["peak_px"]-p2["entry_px"])/p2["entry_px"]
        hold=(tdt-date.fromisoformat(p2["entry_date"])).days
        ep=env_p(p2["env"])
        if pnl_>=TP_BREAKEVEN_LOCK and p2["stop_px"]<p2["entry_px"]: p2["stop_px"]=max(p2["stop_px"],p2["entry_px"])
        if hold>=60 and abs(pnl_)<0.05: p2["stop_px"]=min(p2["stop_px"],p2["entry_px"]*0.85)
        hit=False
        if ep["env_stop"] and hold>=3 and dd.get(stk,{}).get("low",rp)<=p2["entry_px"]*(1-ep["env_stop"]):
            tc.append((stk,"env_stop_loss",sp,rp));hit=True
        if not hit and hold>=3 and dd.get(stk,{}).get("low",rp)<=p2["stop_px"]:
            tc.append((stk,"stop_loss",sp,rp));hit=True
        if not hit and ep["trail"] and ppnl_>=TP_BREAKEVEN_LOCK and rp/p2["peak_px"]<=ep["trail"]:
            nd2=get_next_trade_date(ds,all_dates);tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
            tc.append((stk,"trailing_stop",tsp,rp));hit=True
        if not hit and pnl_>=ep["tp2"]:
            p2["tp2"]=True;nd2=get_next_trade_date(ds,all_dates);tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
            tr.append((stk,"tp2",0.50,tsp));hit=True
        if not hit and pnl_>=ep["tp1"]:
            p2["tp1"]=True;nd2=get_next_trade_date(ds,all_dates);tsp=data[nd2][stk]["open"]*SLIPPAGE_SELL if nd2 and stk in data.get(nd2,{}) else sp
            tr.append((stk,"tp1",0.30,tsp))
    for sym,reason,rpct,sp in tr:
        p2=pos[sym];rs=int(p2["shares"]*rpct)
        if rs<=0 or p2["shares"]<=rs: continue
        csh=p2["shares"];net=sell_net(rs,sp/SLIPPAGE_SELL);p2["shares"]-=rs;cash+=net
        exit_stats[f"reduce_{reason}"]+=1
    for sym,reason,price,_ in tc:
        p2=pos.pop(sym,None)
        if not p2: continue
        net=sell_net(p2["shares"],price/SLIPPAGE_SELL)
        pnl_pct=(price-p2["entry_px"])/p2["entry_px"]*100;cash+=net;exit_stats[reason]+=1
        trades.append({"date":ds,"symbol":sym,"action":"SELL","price":round(price,3),
            "shares":p2["shares"],"pnl_pct":round(pnl_pct,2),"reason":reason,"hold_days":hold,
            "fund_score":p2["fs"],"ths_env":p2.get("env","?")})
        if reason=="stop_loss": cool[sym]=tdt+timedelta(days=COOLING_PERIOD_DAYS)
    eq_p=cash+sum(p3["shares"]*dd.get(s3,{}).get("close",p3["entry_px"]) for s3,p3 in pos.items())
    peak_eq=max(peak_eq,eq_p)
    eq.append({"date":ds,"nav":round(eq_p,2),"dd":round((eq_p/peak_eq-1)*100,2)})
    if (di+1)%500==0 or di==len(bd)-1:
        log.info(f"  [{ds}]({di+1}/{len(bd)}) NAV={eq_p:,.0f} pos={len(pos)} trades={len(trades)}")

eq_df=pd.DataFrame(eq);eq_df.to_csv(str(OUT_DIR/"equity.csv"),index=False)
trades_df=pd.DataFrame(trades);trades_df.to_csv(str(OUT_DIR/"trades.csv"),index=False)
rets=eq_df['nav'].pct_change().dropna()
sharpe=np.mean(rets)/max(np.std(rets),1e-8)*np.sqrt(252);md=eq_df['dd'].min();tot=(eq_df.iloc[-1]['nav']/INITIAL_CAPITAL-1)*100
n_years=len(bd)/252;ann_ret=((1+tot/100)**(1/max(n_years,1))-1)*100
sells=trades_df[trades_df["action"]=="SELL"];wins=sells[sells["pnl_pct"]>0]
wr=len(wins)/max(len(sells),1)*100
aw=wins["pnl_pct"].mean() if len(wins)>0 else 0
al=sells[sells["pnl_pct"]<=0]["pnl_pct"].mean() if len(sells[sells["pnl_pct"]<=0])>0 else 0
wlr=abs(aw/max(al,0.01)) if al<0 else 0
yearly={}
for y in range(2010,2027):
    ys=str(y); yeq=eq_df[eq_df["date"].str.startswith(ys)]
    if len(yeq)<10: continue
    yr=(yeq.iloc[-1]["nav"]/yeq.iloc[0]["nav"]-1)*100
    ysel=sells[sells["date"].str.startswith(ys)]; ywin=ysel[ysel["pnl_pct"]>0]
    yearly[ys]={"return_pct":round(yr,2),"n_trades":len(ysel),"win_rate":round(len(ywin)/max(len(ysel),1)*100,1),"max_dd":round(yeq["dd"].min(),2)}
exit_analysis={}
for r in sells["reason"].unique():
    sub=sells[sells["reason"]==r];pnls=sub["pnl_pct"]
    exit_analysis[r]={"count":len(sub),"avg_pnl":round(pnls.mean(),2),"avg_hold_days":round(sub["hold_days"].mean(),1),"win_rate":round((pnls>0).sum()/max(len(sub),1)*100,1)}
analysis={"total_return_pct":round(tot,2),"ann_return_pct":round(ann_ret,2),"sharpe":round(sharpe,4),"max_drawdown_pct":round(md,2),"win_rate_pct":round(wr,1),"n_buys":len(trades_df[trades_df["action"]=="BUY"]),"n_sells":len(sells),"profit_loss_ratio":round(wlr,2),"avg_win_pct":round(aw,2),"avg_loss_pct":round(al,2),"momentum_passed":mpa,"momentum_rejected":mre,"r1_filtered":bep,"yearly_returns":yearly,"exit_reasons":exit_analysis,"signal_types":dict(sig_stats),"backtest_start":bt_start,"backtest_end":end_d,"final_nav":round(eq_df.iloc[-1]["nav"],2),"elapsed_seconds":round(time.time()-t0,1),}
with open(str(OUT_DIR/"analysis.json"),"w") as f: json.dump(analysis,f,indent=2,ensure_ascii=False)
summary={k:v for k,v in analysis.items() if isinstance(v,(int,float,str))}
with open(str(OUT_DIR/"summary.json"),"w") as f: json.dump(summary,f,indent=2,ensure_ascii=False)

print(f"\n{'='*65}")
print("📊 v6.5a-R1 — min_hold=3")
print(f"{'='*65}")
print(f"  夏普: {sharpe:.4f} | 总收益: {tot:+.2f}% | 年化: {ann_ret:+.2f}%")
print(f"  回撤: {md:.2f}% | 胜率: {wr:.1f}% | 交易: {len(sells)} | 盈亏比: {wlr:.2f}")
print(f"  动量通过: {mpa} 拒绝: {mre} | r1过滤: {bep}")
print(f"  终值: {eq_df.iloc[-1]['nav']:,.0f} | 耗时: {time.time()-t0:.0f}s")
print(f"\n{'─'*55}\n逐年\n{'─'*55}")
print(f"{'年份':>6s}  {'收益':>8s}  {'交易':>6s}  {'胜率':>7s}  {'回撤':>8s}")
for y,yd in sorted(yearly.items()):
    print(f"{y:>6s}  {yd['return_pct']:>+7.2f}%  {yd['n_trades']:>6d}  {yd['win_rate']:>6.1f}%  {yd['max_dd']:>7.2f}%")
print(f"\n{'─'*55}\n退出原因\n{'─'*55}")
for r,d in sorted(exit_analysis.items(),key=lambda x:-x[1]["count"]):
    p=d["count"]/max(len(sells),1)*100
    print(f"  {r:<20s}  {d['count']:>4d}({p:3.0f}%)  avg={d['avg_pnl']:>+.2f}%  hold={d['avg_hold_days']:.0f}d  wr={d['win_rate']:.0f}%")
log.info(f"✅ v6.5 bullOnly 完成 ({time.time()-t0:.0f}s)")
