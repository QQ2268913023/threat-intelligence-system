import networkx as nx

def predict_attack_paths(G):
    paths = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                try:
                    if nx.has_path(G, source, target):
                        path = nx.shortest_path(G, source, target, weight='weight')
                        score = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)) / len(path)
                        paths.append({
                            "path": {"nodes": [{"id": n} for n in path]},
                            "risk_score": score
                        })
                except nx.NetworkXNoPath:
                    continue
                except Exception as e:
                    print(f"⚠️ 路径处理异常：{source} → {target}，原因：{e}")
                    continue
    return sorted(paths, key=lambda x: x['risk_score'], reverse=True)
