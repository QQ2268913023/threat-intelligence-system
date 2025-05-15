
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import networkx as nx
import os
from openai import OpenAI
from neo4j import GraphDatabase  # 导入 Neo4j 驱动
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 设置 Matplotlib 支持汉字
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# API 密钥和客户端配置
api_key = os.environ.get("DASHSCOPE_API_KEY", "sk-c40c9b6a73294d519a7a13cb00ffb983")
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Neo4j 数据库连接配置（请根据你的实际设置修改）
NEO4J_URI = "neo4j+s://7ffe6d75.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "7l2PonEcEPWgdWYsD_Q8i4bfBTwAKTFjhGe2oD0f_Dw"


# 连接到 Neo4j 数据库
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 从 Neo4j 查询知识图谱数据
def query_attack_graph():
    """从 Neo4j 查询攻击边（任意关系类型），提取 source.name、target.name 和关系属性"""
    query = """
    MATCH (source)-[r]->(target)
    RETURN source.name AS source_id, target.name AS target_id, 
           type(r) AS relation_type,
           r.risk_score AS risk_score
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            data = []
            for record in result:
                data.append({
                    "source": record["source_id"],
                    "target": record["target_id"],
                    "relation": record["relation_type"],
                    "risk_score": record.get("risk_score", 1.0) or 1.0  # 默认值防止为 None
                })
            return data
    except Exception as e:
        print(f"Neo4j 查询失败: {e}")
        return []


# 从 Neo4j 构建 NetworkX 图
def build_attack_graph_from_neo4j():
    """构建 NetworkX 图，边权重取自 risk_score 或默认值 1.0"""
    data = query_attack_graph()
    if not data:
        raise ValueError("无法从 Neo4j 获取知识图谱数据")

    G = nx.DiGraph()
    for entry in data:
        source = entry["source"]
        target = entry["target"]
        weight = entry["risk_score"] or 1.0
        relation = entry["relation"]

        G.add_edge(source, target, weight=weight, label=relation)

    return G


# 数据收集和预处理
def load_threat_data(file_path='threat_data.csv'):
    """加载和预处理历史威胁情报数据"""
    df = pd.read_csv(file_path, parse_dates=['日期'])
    df.set_index('日期', inplace=True)
    return df

# 特征工程
def extract_features(df):
    """从数据中提取特征"""
    df['攻击频率'] = df['攻击次数'].rolling(window=7).mean()  # 7天移动平均
    df['攻击类型变化'] = df['攻击类型'].ne(df['攻击类型'].shift()).cumsum()
    return df.dropna()

# 时间序列预测模型
def train_lstm_model(data, timesteps=30):
    """训练 LSTM 模型预测攻击次数"""
    if len(data) <= timesteps:
        raise ValueError(f"数据不足：需要大于 {timesteps} 条记录")
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(Input(shape=(timesteps, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    return model

# 预测未来攻击次数
def predict_future(model, last_sequence, days=7):
    """预测未来 N 天的攻击次数"""
    predictions = []
    current_sequence = last_sequence.reshape((1, -1, 1))
    for _ in range(days):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = pred[0, 0]
    return predictions

# 趋势分析与异常检测
def detect_trends_and_anomalies(data):
    """检测趋势和异常"""
    clf = IsolationForest(contamination=0.01)
    anomalies = clf.fit_predict(data.reshape(-1, 1))
    return anomalies

# 预测攻击路径
def predict_attack_paths(G):
    """从知识图谱预测高风险攻击路径"""
    paths = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target and nx.has_path(G, source, target):
                path = nx.shortest_path(G, source, target, weight='weight')
                risk_score = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)) / len(path)
                paths.append({"path": {"nodes": [{"id": node} for node in path]}, "risk_score": risk_score})
    return sorted(paths, key=lambda x: x['risk_score'], reverse=True)[:3]

# 可视化
def visualize_graph(G, paths):
    """可视化攻击路径"""
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    for path in paths:
        path_edges = [(path['path']['nodes'][i]['id'], path['path']['nodes'][i + 1]['id']) for i in range(len(path['path']['nodes']) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.savefig('threat_paths.png')
    plt.close()

def visualize_time_series(data, predictions, anomalies):
    """可视化时间序列和预测"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, label='历史攻击次数')
    plt.plot(range(len(data), len(data) + len(predictions)), predictions, label='预测攻击次数', linestyle='--')
    anomaly_indices = [i for i, a in enumerate(anomalies) if a == -1]
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label='异常点')
    plt.legend()
    plt.xlabel('天数')
    plt.ylabel('攻击次数')
    plt.title('攻击趋势预测')
    plt.savefig('threat_trend.png')
    plt.close()

# 生成警告报告
def generate_warning_report(paths, predictions, anomalies, df):
    """生成安全威胁警告报告"""
    path_summary = "\n".join([f"路径: {' -> '.join([node['id'] for node in p['path']['nodes']])} (风险评分: {p['risk_score']:.2f})" for p in paths])
    anomaly_summary = "\n".join([f"日期: {df.index[-30 + i].date()}, 攻击次数: {df['攻击次数'].iloc[-30 + i]}" for i, a in enumerate(anomalies) if a == -1])
    prompt = f"""
    根据历史威胁情报和当前趋势，预测未来7天的攻击次数为{predictions}。
    过去30天内检测到的异常点：
    {anomaly_summary}
    高风险攻击路径：
    {path_summary}
    请生成一份中文安全威胁警告报告，并建议应对措施。
    """
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 调用错误：{e}")
        return "无法生成报告，请检查 API 配置。"

# 主函数
def main():
    # 数据加载和预处理
    df = load_threat_data()
    df = extract_features(df)

    # 从 Neo4j 构建知识图谱并预测路径
    G = build_attack_graph_from_neo4j()
    attack_paths = predict_attack_paths(G)

    # 时间序列预测
    attack_data = df['攻击次数'].values
    timesteps = 30
    train_size = int(len(attack_data) * 0.8)
    if len(attack_data) < train_size + 10:
        raise ValueError(f"数据不足：需要至少 {train_size + 10} 条记录")
    train_data = attack_data[:train_size]
    model = train_lstm_model(train_data, timesteps)
    last_sequence = attack_data[-timesteps:]
    future_predictions = predict_future(model, last_sequence)

    # 趋势和异常检测
    anomalies = detect_trends_and_anomalies(attack_data[-30:])

    # 可视化
    visualize_graph(G, attack_paths)
    visualize_time_series(attack_data[-30:], future_predictions, anomalies)

    # 生成警告报告
    report = generate_warning_report(attack_paths, future_predictions, anomalies, df)
    print("\n安全威胁警告报告：")
    print(report)
    print("\n已生成可视化文件：threat_paths.png 和 threat_trend.png")

if __name__ == "__main__":
    main()

