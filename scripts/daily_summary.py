#!/usr/bin/env python3
"""
每日工作总结脚本 — 每天凌晨 2:00 运行
读取 memory/YYYY-MM-DD.md，生成结构化总结
存入 memory/daily-summaries/YYYY-MM-DD.md
"""

import os, re, json
from datetime import date, timedelta, datetime
from pathlib import Path

WORKSPACE = Path(os.environ.get("_WORKSPACE", "/home/quant/.openclaw/workspace"))
MEMORY_DIR = WORKSPACE / "memory"
SUMMARY_DIR = MEMORY_DIR / "daily-summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

yesterday = date.today() - timedelta(days=1)
yesterday_str = yesterday.strftime("%Y-%m-%d")
today_str = date.today().strftime("%Y-%m-%d")

# 读取前一天的工作日志
log_path = MEMORY_DIR / f"{yesterday_str}.md"
if not log_path.exists():
    # 也检查今天（首次运行时）或最近的工作日志
    log_path = MEMORY_DIR / f"{date.today().strftime('%Y-%m-%d')}.md"
    
if not log_path.exists():
    print(f"[{today_str}] ⚠️ 未找到工作日志: {log_path}")
    # 尝试找最近3天
    for i in range(1, 4):
        test_path = MEMORY_DIR / f"{(date.today() - timedelta(days=i)).strftime('%Y-%m-%d')}.md"
        if test_path.exists():
            log_path = test_path
            yesterday_str = (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            print(f"  使用最近日志: {log_path}")
            break
    else:
        exit(0)

log_text = log_path.read_text(encoding="utf-8")

# 解析结构化内容
summary = {
    "date": yesterday_str,
    "generated_at": today_str,
    "title": "",
    "completed_tasks": [],
    "pending_tasks": [],
    "key_findings": [],
    "knowledge_items": [],
    "metrics": {},
}

# 提取标题
title_match = re.search(r'^# (.+)$', log_text, re.MULTILINE)
if title_match:
    summary["title"] = title_match.group(1).strip()

# 提取完成事项
task_section = re.split(r'## 一、|## 完成事项|## 今日总览', log_text)
completed = []
for line in log_text.split('\n'):
    if re.match(r'^\d+\.\s+', line) or re.match(r'^- \[x\]', line, re.IGNORECASE) or re.match(r'^- \*\*', line):
        cleaned = re.sub(r'^[\d\.\-\*\[\]\s]+', '', line).strip()
        if cleaned and not cleaned.startswith('|') and not cleaned.startswith('---'):
            completed.append(cleaned)

# 提取关键发现
findings = []
for line in log_text.split('\n'):
    if re.match(r'^####?\s+(关键发现|结论|核心发现|根因|建议)', line):
        findings.append(line)
    elif '✅' in line and len(line) < 80:
        completed.append(line.replace('✅', '').strip())

# 生成总结文件
summary_text = f"""# 每日工作摘要 — {yesterday_str}

> 生成时间: {today_str}

## 完成事项

"""

for i, task in enumerate(completed[:20], 1):
    summary_text += f"{i}. {task}\n"

summary_text += f"""

## 原始日志
> 详见: {log_path}
"""

# 写入
output_path = SUMMARY_DIR / f"{yesterday_str}.md"
output_path.write_text(summary_text, encoding="utf-8")
print(f"[{today_str}] ✅ 每日总结已生成: {output_path}")
print(f"  提取完成事项: {len(completed)} 条")
