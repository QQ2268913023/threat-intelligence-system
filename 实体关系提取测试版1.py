import csv
import json
import os
import time
import sys
import re
from tqdm import tqdm
from neo4j_loader import push_to_neo4j
import dashscope

# ====== 配置 ======
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    dashscope.api_key = config.get("api_key", "")

INPUT_FILE = "F:/威胁情报系统/CVE_2025_combined.csv"
OUTPUT_DIR = 'output'
ERROR_LOG = 'error_log.txt'
BATCH_SIZE = 100
MAX_SAMPLES = 5  # 小样本测试

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ====== 通用函数 ======

def log_error(message):
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n\n')

def check_response_status(response, context="未知调用"):
    if response is None:
        log_error(f"[{context}] API返回None")
        sys.exit(1)
    if hasattr(response, 'status_code') and response.status_code != 200:
        log_error(f"[{context}] 错误状态码：{response.status_code}")
        sys.exit(1)
    if hasattr(response, 'message') and response.message:
        if any(k in response.message for k in ["InsufficientQuota", "InvalidApiKey", "AccessDenied"]):
            log_error(f"[{context}] API权限或额度错误：{response.message}")
            sys.exit(1)

def extract_json_block(text):
    """从返回内容中提取 JSON对象或数组"""
    try:
        # 优先提取```json```代码块
        code_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            for block in code_blocks:
                block = block.strip()
                if block.startswith('{') or block.startswith('['):
                    return json.loads(block)

        # 再尝试裸的 { }
        match_obj = re.search(r'\{.*?\}', text, re.DOTALL)
        if match_obj:
            json_str = match_obj.group(0)
            return json.loads(json_str)

        # 或裸的 [ {...} ]
        match_list = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match_list:
            json_str = match_list.group(0)
            return json.loads(json_str)

    except Exception as e:
        log_error(f"[提取JSON异常] {str(e)}\n原始返回内容：{text}")
    return None

# ====== 核心处理函数 ======

def extract_entities(text):
    prompt = f"""
你是网络安全专家，请严格只返回如下JSON格式（不要任何注释或解释）：
{{
  "vulnerability": [],
  "vendor": [],
  "product": [],
  "malware": [],
  "technique": [],
  "exploit": [],
  "mitigation": []
}}

漏洞描述如下：
\"\"\"{text}\"\"\"
"""
    for _ in range(3):
        try:
            response = dashscope.Generation.call(
                model='qwen-plus',
                prompt=prompt,
                temperature=0.2,
            )
            check_response_status(response, context="实体提取")
            result_text = response.output.get('text', '').strip()
            result = extract_json_block(result_text)
            if result:
                return result
        except Exception as e:
            log_error(f"[实体提取重试] {str(e)}\n原文：{text}")
            time.sleep(5)
    return {}

def extract_relations(text):
    prompt = f"""
根据漏洞描述，只输出漏洞实体间的因果/利用关系。严格以JSON数组返回：
[
  {{"subject": "", "relation": "", "object": ""}},
  ...
]

漏洞描述如下：
\"\"\"{text}\"\"\"
"""
    for _ in range(3):
        try:
            response = dashscope.Generation.call(
                model='qwen-plus',
                prompt=prompt,
                temperature=0.2,
            )
            check_response_status(response, context="关系提取")
            result_text = response.output.get('text', '').strip()
            result = extract_json_block(result_text)
            if result:
                return result
        except Exception as e:
            log_error(f"[关系提取重试] {str(e)}\n原文：{text}")
            time.sleep(5)
    return []

# ====== 主处理逻辑 ======


def process_batch(start_line=0):
    with open(INPUT_FILE, 'r', encoding='gb18030') as csvfile:
        reader = list(csv.DictReader(csvfile))
        total = len(reader)

        if MAX_SAMPLES is not None:
            reader = reader[start_line:start_line + MAX_SAMPLES]
            total = len(reader)

        batch_num = start_line // BATCH_SIZE

        for i in tqdm(range(0, total, BATCH_SIZE), desc="批次处理"):
            batch = reader[i:i + BATCH_SIZE]
            results = []

            for row in tqdm(batch, desc="处理单条数据", leave=False):
                try:
                    text = row.get('description', '')
                    if not text:
                        print("警告：当前行缺少 description 字段")
                        continue

                    print(f"正在处理描述：{text[:60]}...")

                    entities = extract_entities(text)
                    relations = extract_relations(text)

                    row['extracted_entities'] = entities
                    row['extracted_relations'] = relations
                    results.append(row)

                except Exception as e:
                    log_error(f"[主处理异常] {str(e)}\n行内容：{row}")

            # 保存 JSON 文件
            output_file = os.path.join(OUTPUT_DIR, f'cve_batch_{batch_num:03d}.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # ✅ 推送到 Neo4j 知识图谱
            try:
                push_to_neo4j(results)
            except Exception as e:
                log_error(f"[Neo4j 推送异常] {str(e)}")

            batch_num += 1


# ====== 程序入口 ======

if __name__ == '__main__':
    start = 0
    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    process_batch(start_line=start)
