#!/usr/bin/env python3
"""
📊 宏观经济数据下载器

数据源优先级：本地 Tushare API → akshare
输出到：quant/data/macro/

数据项：
  1. GDP      - 本地API cn_gdp / akshare
  2. CPI      - 本地API cn_cpi / akshare
  3. PPI      - 本地API cn_ppi / akshare
  4. PMI      - 本地API cn_pmi / akshare
  5. M2/货币供应 - 本地API cn_m / akshare supply_of_money
  6. 社融     - akshare macro_china_shrzgm
  7. 工业增加值 - akshare macro_china_industrial_production_yoy
  8. 社零     - akshare macro_china_consumer_goods_retail
  9. 固投     - akshare macro_china_gdzctz
  10. 国债收益率 - akshare bond_china_yield
  11. Shibor   - 本地API shibor / akshare
  12. Libor    - 本地API libor
  13. LPR      - akshare macro_china_lpr
  14. 人民币汇率 - akshare macro_china_rmb
  15. 外汇日线 - 本地API fx_daily
"""

import os, sys, json, time, subprocess, warnings
from pathlib import Path
import pandas as pd
import akshare as ak
warnings.filterwarnings("ignore")

# ====== 配置 ======
TOKEN = "A9-3nOgSXEYgPCnZrPnA5tqXbCARyRyQSgVexU61wxY"
URL = "http://47.109.59.144:8989/dataapi"
DATA = Path(__file__).resolve().parent.parent / "quant" / "data"
MACRO = DATA / "macro"
MACRO.mkdir(parents=True, exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def save_parquet(df, subdir, name):
    """保存到 macro/{subdir}/data.parquet"""
    d = MACRO / subdir
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.parquet"
    df.to_parquet(out, compression="snappy", index=False)
    log(f"  ✅ {subdir}: {len(df)} 行 → {out}")

def tushare_api(api_name, params=None, fields=""):
    """调用本地 Tushare API"""
    p = json.dumps({"api_name": api_name, "token": TOKEN,
                     "params": params or {}, "fields": fields})
    try:
        r = subprocess.run(["curl", "-s", "--connect-timeout", "8", "--max-time", "20",
            "-X", "POST", URL, "-H", "Content-Type: application/json", "-d", p],
            capture_output=True, text=True, timeout=25)
        d = json.loads(r.stdout)
        if d.get("code") == 0 and d.get("data"):
            items = d["data"].get("items", [])
            cols = d["data"].get("fields", [])
            if items:
                return pd.DataFrame(items, columns=cols)
        return None
    except:
        return None

# ====== 1-5: 本地 API 数据 ======
def fetch_local_macro():
    log("\n" + "="*50)
    log("本地 Tushare API 宏观数据")

    apis = {
        "gdp": ("经济/GDP", {"fields": ""}),
        "cpi": ("价格/CPI", {"fields": ""}),
        "ppi": ("价格/PPI", {"fields": ""}),
        "pmi": ("PMI", {"fields": ""}),
        "cn_m": ("货币/M2", {"fields": ""}),
    }

    for api_name, (subdir, params) in apis.items():
        log(f"  📥 {api_name} ({subdir})...")
        df = tushare_api(api_name, params)
        if df is not None and len(df) > 0:
            log(f"    ✅ {len(df)} 行")
            save_parquet(df, f"cn_{api_name}", "data")
        else:
            log(f"    ⚠️ 无数据，尝试 akshare...")
            try:
                if api_name == "gdp":
                    df = ak.macro_china_gdp_yearly()
                elif api_name == "cpi":
                    df = ak.macro_china_cpi_yearly()
                elif api_name == "ppi":
                    df = ak.macro_china_ppi_yearly()
                elif api_name == "pmi":
                    df = ak.macro_china_pmi_yearly()
                elif api_name == "cn_m":
                    df = ak.macro_china_m2_yearly()
                if df is not None and len(df) > 0:
                    log(f"    ✅ akshare: {len(df)} 行")
                    save_parquet(df, f"cn_{api_name}", "data")
                else:
                    log(f"    ❌ 均无数据")
            except Exception as e:
                log(f"    ❌ akshare 失败: {str(e)[:60]}")

# ====== 6-14: akshare 数据 ======
def fetch_akshare_macro():
    log("\n" + "="*50)
    log("akshare 宏观数据")

    sources = [
        ("社融/社会融资规模", ak.macro_china_shrzgm, "cn_shrzgm"),
        ("工业增加值", ak.macro_china_industrial_production_yoy, "cn_ind_prod"),
        ("社零/消费品零售", ak.macro_china_consumer_goods_retail, "cn_retail"),
        ("固投/固定资产投资", ak.macro_china_gdzctz, "cn_fai"),
        ("M2/货币供应量", ak.macro_china_supply_of_money, "cn_money_supply"),
        ("LPR利率", ak.macro_china_lpr, "cn_lpr"),
        ("人民币汇率", ak.macro_china_rmb, "cn_rmb"),
        ("国债收益率", ak.bond_china_yield, "cn_bond_yield"),
    ]

    for desc, fn, subdir in sources:
        log(f"  📥 {desc}...")
        try:
            df = fn()
            if df is not None and len(df) > 0:
                log(f"✅ {len(df)} 行")
                save_parquet(df, subdir, "data")
            else:
                log("⚠️ 空数据")
        except Exception as e:
            log(f"❌ {str(e)[:60]}")

# ====== 15: 外汇 ======
def fetch_fx():
    log("\n" + "="*50)
    log("外汇数据")

    # fx_daily from local API
    df = tushare_api("fx_daily")
    if df is not None and len(df) > 0:
        save_parquet(df, "fx_daily", "data")
        log(f"  ✅ 外汇日线: {len(df)} 行")
    else:
        log("  ⚠️ 本地API无外汇数据")

    # 人民币汇率补充
    log("  📥 人民币汇率 (akshare)...")
    try:
        df = ak.macro_china_rmb()
        if df is not None and len(df) > 0:
            save_parquet(df, "cn_rmb", "data")
    except:
        pass

# ====== Shibor & Libor ======
def fetch_rates():
    log("\n" + "="*50)
    log("利率数据")

    for api_name, subdir in [("shibor", "shibor"), ("libor", "libor"),
                              ("shibor_lpr", "shibor_lpr"), ("shibor_quote", "shibor_quote")]:
        log(f"  📥 {api_name}...")
        df = tushare_api(api_name)
        if df is not None and len(df) > 0:
            log(f"✅ {len(df)} 行")
            save_parquet(df, subdir, "data")
        else:
            log("❌ 无数据")

    # akshare shibor 补充
    log("  📥 Shibor (akshare)...")
    try:
        df = ak.macro_china_shibor_all()
        if df is not None and len(df) > 0:
            save_parquet(df, "shibor_all", "data")
    except:
        pass

# ====== 主流程 ======
def main():
    t0 = time.time()
    log("📊 宏观经济数据下载")
    log(f"  输出目录: {MACRO}")

    fetch_local_macro()
    fetch_akshare_macro()
    fetch_fx()
    fetch_rates()

    # 写元数据
    meta = {
        "migrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": ["local_tushare_api", "akshare"],
        "total_time_s": int(time.time() - t0),
    }
    json.dump(meta, open(MACRO / "_meta.json", "w"), ensure_ascii=False, indent=2)

    log(f"\n{'='*50}")
    log(f"✅ 宏观数据下载完成，耗时 {time.time()-t0:.0f}s")
    log(f"  共 {len(list(MACRO.rglob('*.parquet')))} 个 parquet 文件")

    # 列出下载结果
    log(f"\n📂 结构:")
    for f in sorted(MACRO.rglob("*.parquet")):
        rel = f.relative_to(MACRO)
        df = pd.read_parquet(f)
        log(f"  {rel}: {len(df):>6} 行")

if __name__ == "__main__":
    main()
