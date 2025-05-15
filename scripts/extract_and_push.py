import csv
import json
import os
import time
import sys
import re
from tqdm import tqdm
import dashscope
from neo4j import GraphDatabase

# ==== 配置读取 ====
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    dashscope.api_key = config.get("api_key", "")

INPUT_FILE = "data/CVE_2025_combined.csv"
OUTPUT_DIR = 'output'
ERROR_LOG = os.path.join(OUTPUT_DIR, 'error_log.txt')

# ==== Neo4j 设置 ====
NEO4J_URI = "neo4j+s://7ffe6d75.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "7l2PonEcEPWgdWYsD_Q8i4bfBTwAKTFjhGe2oD0f_Dw"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==== 工具函数 ====

def log_error(message):
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        f.write(message + '\n\n')

def check_response_status(response, context="未知调用"):
    if response is None or hasattr(response, 'status_code') and response.status_code != 200:
        log_error(f"[{context}] 错误：{response}")
        sys.exit(1)

def extract_json_block(text):
    try:
        code_blocks = re.findall(r'```json(.*?)```', text, re.DOTALL | re.IGNORECASE)
        for block in code_blocks:
            block = block.strip()
            if block.startswith('{') or block.startswith('['):
                return json.loads(block)
        match_obj = re.search(r'\{.*?\}', text, re.DOTALL)
        if match_obj:
            return json.loads(match_obj.group(0))
        match_list = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match_list:
            return json.loads(match_list.group(0))
    except Exception as e:
        log_error(f"[提取JSON异常] {str(e)}\n原始返回内容：{text}")
    return None

# ==== 核心提取函数 ====

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
漏洞描述如下：\"\"\"{text}\"\"\"
"""
    for _ in range(3):
        try:
            response = dashscope.Generation.call(model='qwen-plus', prompt=prompt, temperature=0.2)
            check_response_status(response, context="实体提取")
            result = extract_json_block(response.output.get('text', ''))
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
漏洞描述如下：\"\"\"{text}\"\"\"
"""
    for _ in range(3):
        try:
            response = dashscope.Generation.call(model='qwen-plus', prompt=prompt, temperature=0.2)
            check_response_status(response, context="关系提取")
            result = extract_json_block(response.output.get('text', ''))
            if result:
                return result
        except Exception as e:
            log_error(f"[关系提取重试] {str(e)}\n原文：{text}")
            time.sleep(5)
    return []

# ==== 推送图谱 ====

def push_to_neo4j(results):
    with driver.session() as session:
        for item in results:
            entities = item.get('extracted_entities', {})
            relations = item.get('extracted_relations', [])

            for etype, items in entities.items():
                for name in items:
                    if name.strip():
                        session.run(
                            f"MERGE (n:`{etype.capitalize()}` {{name: $name}}) SET n.type = $type",
                            {"name": name.strip(), "type": etype}
                        )

            for rel in relations:
                sub = rel.get("subject", "").strip()
                obj = rel.get("object", "").strip()
                rel_type = rel.get("relation", "").strip()
                if sub and obj and rel_type:
                    session.run(
                        f"""
                        MATCH (a {{name: $sub}}), (b {{name: $obj}})
                        MERGE (a)-[r:`{rel_type.upper()}`]->(b)
                        """,
                        {"sub": sub, "obj": obj}
                    )
    print("✅ Neo4j 图谱构建完成")

# ==== 主流程 ====

def process_batch(start_line=0, max_samples=5, batch_size=100):
    with open(INPUT_FILE, 'r', encoding='gb18030') as f:
        reader = list(csv.DictReader(f))
        total = len(reader)
        reader = reader[start_line:start_line + max_samples]
        for i in tqdm(range(0, len(reader), batch_size), desc="处理批次"):
            batch = reader[i:i+batch_size]
            results = []
            for row in tqdm(batch, desc="解析数据", leave=False):
                text = row.get('description', '')
                if not text:
                    continue
                entities = extract_entities(text)
                relations = extract_relations(text)
                row['extracted_entities'] = entities
                row['extracted_relations'] = relations
                results.append(row)
            out_file = os.path.join(OUTPUT_DIR, f'cve_batch_{i//batch_size:03d}.json')
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            push_to_neo4j(results)
