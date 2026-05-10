"""
国家统计局 (stats.gov.cn) 宏观数据 API 调用工具
==============================================
参考: https://data.stats.gov.cn/easyquery.htm
接口: https://data.stats.gov.cn/easyquery.htm

核心参数:
  m=QueryData  — 查询数据
  dbcode=hgnd  — 数据库编码 (hgnd=年度宏观, hgyd=月度宏观, hgqd=季度宏观)
  rowcode=reg  — 行编码
  colcode=sj   — 列编码
  wds=[]       — 筛选条件 (json array)
  dfwds=[]     — 日期范围 (json array, [{"wdcode":"zb","valuecode":"A0101"}])

常用指标编码 (zb):
  A01 = 国内生产总值(GDP)
  A0101 = 国内生产总值_累计值
  A0102 = 第一产业增加值
  A0103 = 第二产业增加值
  A0104 = 第三产业增加值
  
  A02 = 农业
  A03 = 工业
  A0301 = 工业增加值_当月增速
  A0302 = 工业增加值_累计增速
  A0309 = 工业增加值_当月
  
  A04 = 能源
  A05 = 固定资产投资
  A0501 = 固定资产投资_累计值
  A0502 = 固定资产投资_累计增速
  
  A06 = 房地产
  A07 = 国内贸易
  A0701 = 社会消费品零售总额_当月
  A0702 = 社会消费品零售总额_累计
  A0703 = 社会消费品零售总额_当月增速
  
  A08 = 对外贸易
  A09 = 物价
  A0A = 人民生活
  A0B = 工资
  A0C = 财政
  A0D = 金融
  A0E = 证券
  A0F = 保险

用法:
  from stats_cn_api import NBSDataAPI
  api = NBSDataAPI()
  
  # 获取工业增加值月度数据
  df = api.query("A0301", dbcode="hgyd")
  
  # 获取社会消费品零售总额
  df = api.query("A0701", dbcode="hgyd")
  
  # 获取固定资产投资
  df = api.query("A0501", dbcode="hgyd")
  
  # 获取GDP
  df = api.query("A0101", dbcode="hgnd")
"""

import json
import time
import pandas as pd
import requests
from typing import Optional, List, Dict

class NBSDataAPI:
    """
    国家统计局数据查询API封装
    参考: https://data.stats.gov.cn/easyquery.htm
    """
    
    BASE_URL = "https://data.stats.gov.cn/easyquery.htm"
    
    # 常用数据库编码
    DB_CODES = {
        "hgnd": "年度数据",
        "hgyd": "月度数据", 
        "hgqd": "季度数据",
    }
    
    # 常用指标 (zb)
    INDICATORS = {
        # GDP
        "A0101": "国内生产总值(亿元)_累计值",
        "A010102": "国内生产总值(亿元)_当季值",
        "A010201": "第一产业增加值(亿元)",
        "A010301": "第二产业增加值(亿元)",
        "A010401": "第三产业增加值(亿元)",
        "A010101": "GDP同比增长(%)",
        
        # 工业
        "A030101": "工业增加值_当月同比(%)",
        "A030201": "工业增加值_累计同比(%)",
        "A030901": "工业增加值(亿元)_当月",
        
        # 固定资产投资
        "A050101": "固定资产投资_累计值(亿元)",
        "A050201": "固定资产投资_累计同比(%)",
        
        # 社会消费品零售
        "A070101": "社会消费品零售总额(亿元)_当月",
        "A070102": "社会消费品零售总额(亿元)_累计",
        "A070103": "社会消费品零售总额_当月同比(%)",
        
        # CPI/PPI
        "A090101": "居民消费价格指数(CPI)_当月同比(%)",
        "A090201": "商品零售价格指数_当月同比(%)",
        "A090301": "工业生产者出厂价格指数(PPI)_当月同比(%)",
        
        # 进出口
        "A080101": "进出口总额(千美元)_当月",
        "A080102": "出口总额(千美元)_当月",
        "A080103": "进口总额(千美元)_当月",
    }
    
    def __init__(self, timeout: int = 30):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://data.stats.gov.cn/easyquery.htm",
        })
        self.timeout = timeout
    
    def query(self, zb_code: str, dbcode: str = "hgyd", 
              sj_start: str = "2000", sj_end: str = "2026",
              max_retries: int = 3) -> pd.DataFrame:
        """
        查询统计数据
        
        Args:
            zb_code: 指标编码 (如 "A030101" = 工业增加值当月同比)
            dbcode: 数据库编码 (hgnd/hgyd/hgqd)
            sj_start: 起始年份 (如 "2000")
            sj_end: 结束年份/月 (如 "2025" 或 "202512")
            max_retries: 最大重试次数
            
        Returns:
            DataFrame with columns: [date, value, ...]
        """
        params = {
            "m": "QueryData",
            "dbcode": dbcode,
            "rowcode": "reg",
            "colcode": "sj",
            "wds": json.dumps([]),
            "dfwds": json.dumps([
                {"wdcode": "zb", "valuecode": zb_code},
                {"wdcode": "sj", "valuecode": f">{sj_start}"}
            ]),
        }
        
        for attempt in range(max_retries):
            try:
                resp = self.session.get(
                    self.BASE_URL, 
                    params=params,
                    timeout=self.timeout
                )
                resp.encoding = "utf-8"
                data = resp.json()
                
                if data.get("retcode") != 200:
                    raise ValueError(f"API返回错误: {data.get('retmsg', 'unknown')}")
                    
                return self._parse_response(data, zb_code)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
    
    def _parse_response(self, data: dict, zb_code: str) -> pd.DataFrame:
        """解析API返回的JSON数据"""
        result = []
        try:
            nodes = data.get("retdata", {}).get("datanodes", [])
            wdnodes = data.get("retdata", {}).get("wdnodes", [])
            
            # 提取时间标签
            sj_map = {}
            for wdnode in wdnodes:
                if wdnode.get("wdcode") == "sj":
                    for node in wdnode.get("nodes", []):
                        sj_map[node["code"]] = node["name"]
            
            # 提取数值
            for node in nodes:
                wds = {w["wdcode"]: w["valuecode"] for w in node.get("wds", [])}
                if wds.get("zb") == zb_code:
                    sj_code = wds.get("sj", "")
                    sj_name = sj_map.get(sj_code, sj_code)
                    value = node.get("data", {}).get("data")
                    if value is not None:
                        result.append({"date": sj_name, "value": value})
            
        except Exception as e:
            print(f"解析错误: {e}, 原始数据: {str(data)[:200]}")
        
        df = pd.DataFrame(result)
        if not df.empty:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        return df
    
    def batch_query(self, indicators: List[str], dbcode: str = "hgyd",
                    sj_start: str = "2000", sj_end: str = "2026") -> Dict[str, pd.DataFrame]:
        """批量查询多个指标"""
        results = {}
        for zb in indicators:
            try:
                name = self.INDICATORS.get(zb, zb)
                print(f"📥 [{name}] {zb}...", end=" ")
                df = self.query(zb, dbcode, sj_start, sj_end)
                print(f"{len(df)} rows")
                results[zb] = df
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ {e}")
                results[zb] = None
        return results
    
    def save_to_csv(self, results: Dict[str, pd.DataFrame], output_dir: str = "./"):
        """保存查询结果到CSV"""
        from pathlib import Path
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for zb, df in results.items():
            if df is not None and not df.empty:
                name = self.INDICATORS.get(zb, zb)
                safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
                path = out_dir / f"nbs_{safe_name}.csv"
                df.to_csv(path, index=False, encoding="utf-8-sig")
                print(f"  ✅ 保存: {path}")


def demo():
    """演示用法"""
    api = NBSDataAPI()
    
    # ===== 月度数据 =====
    print("=" * 50)
    print("月度宏观数据")
    print("=" * 50)
    
    monthly = [
        "A030101",  # 工业增加值_当月同比
        "A050101",  # 固定资产投资_累计值
        "A070101",  # 社会消费品零售_当月
    ]
    
    for zb in monthly:
        name = api.INDICATORS.get(zb, zb)
        print(f"\n{name} ({zb}):")
        df = api.query(zb, "hgyd")
        if not df.empty:
            print(df.tail(10).to_string(index=False))
        print()
    
    # ===== 年度数据 =====
    print("=" * 50)
    print("年度宏观数据 (GDP)")
    print("=" * 50)
    
    df = api.query("A0101", "hgnd")  # GDP年度
    if not df.empty:
        print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    # 测试连接
    api = NBSDataAPI()
    try:
        df = api.query("A030101", "hgyd", "2020", "202512")
        print(f"✅ 连接成功! 工业增加值数据: {len(df)} 行")
        if not df.empty:
            print(df.tail(5).to_string(index=False))
            
        # 保存调用方法说明
        print("\n✅ 调用方法已保存, 后续可直接:")
        print("  from stats_cn_api import NBSDataAPI")
        print("  api = NBSDataAPI()")
        print("  df = api.query('A030101', 'hgyd')  # 工业增加值")
        print("  df = api.query('A070101', 'hgyd')  # 社消")
        print("  df = api.query('A050101', 'hgyd')  # 固投")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n⚠️ 注意: stats.gov.cn API 有IP访问限制, 可能需要:")
        print("  1. 在浏览器中访问 https://data.stats.gov.cn 确认IP可访问")
        print("  2. 如果被墙, 尝试使用代理或国内服务器")
        print("  3. 或者使用本地网络环境运行")
