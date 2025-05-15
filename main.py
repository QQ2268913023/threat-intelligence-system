# main.py
from scripts.extract_and_push import process_batch
from scripts.graph_loader import build_attack_graph_from_neo4j
from scripts.path_predictor import predict_attack_paths
from scripts.path_scorer import train_path_classifier, predict_path_risks
from scripts.trend_predictor import load_threat_data, extract_features, train_lstm_model, predict_future
from scripts.anomaly_detector import detect_trends_and_anomalies
from scripts.visualizer import visualize_graph, visualize_time_series
from scripts.report_generator import generate_warning_report


def main():
    print("✅ [1] 实体关系提取并导入图谱...")
    process_batch(start_line=0)  # 提取实体关系并推送至 Neo4j

    print("✅ [2] 构建攻击图...")
    G = build_attack_graph_from_neo4j()

    print("✅ [3] 枚举候选路径...")
    raw_paths = predict_attack_paths(G)

    print("✅ [4] 准备训练数据（路径 + 标签）...")
    # 构造训练样本（用已有路径打标签）
    paths_with_labels = [
        ({"path": [node["id"] for node in p["path"]["nodes"]]}, 1)
        for p in raw_paths[:3]  # 假设前3条是攻击路径
    ]

    print("✅ [5] 训练路径评分模型...")
    clf = train_path_classifier(G, paths_with_labels)

    print("✅ [6] 预测路径攻击概率...")
    scored_paths = predict_path_risks(G, raw_paths, clf)[:3]  # 取前3条

    print("✅ [7] 加载趋势数据并预测...")
    df = load_threat_data()
    df = extract_features(df)
    timesteps = 30
    train_data = df['攻击次数'].values[:-7]
    last_seq = df['攻击次数'].values[-timesteps:]
    lstm_model = train_lstm_model(train_data, timesteps=timesteps)
    future_pred = predict_future(lstm_model, last_seq, days=7)

    print("✅ [8] 异常点检测...")
    anomalies = detect_trends_and_anomalies(df['攻击次数'].values[-30:])

    print("✅ [9] 可视化输出...")
    visualize_graph(G, scored_paths)
    visualize_time_series(df['攻击次数'].values[-30:], future_pred, anomalies)

    print("✅ [10] 生成中文报告...")
    report = generate_warning_report(scored_paths, future_pred, anomalies, df)

    print("\n📄 安全威胁报告如下：\n")
    print(report)


if __name__ == '__main__':
    main()
