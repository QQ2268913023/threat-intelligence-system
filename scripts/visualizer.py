import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, paths):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    for p in paths:
        edges = [(p['path'][i], p['path'][i+1]) for i in range(len(p['path'])-1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
    plt.savefig('output/threat_paths.png')
    plt.close()

def visualize_time_series(data, predictions, anomalies):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), data, label='历史攻击次数')
    plt.plot(range(len(data), len(data)+len(predictions)), predictions, linestyle='--', label='预测攻击次数')
    outliers = [i for i, a in enumerate(anomalies) if a == -1]
    plt.scatter(outliers, [data[i] for i in outliers], color='red', label='异常点')
    plt.legend()
    plt.savefig('output/threat_trend.png')
    plt.close()