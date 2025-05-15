import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def extract_path_features(path, G):
    length = len(path) - 1
    weights = [G[path[i]][path[i+1]].get('weight', 1.0) for i in range(length)]
    labels = [G[path[i]][path[i+1]].get('label', '') for i in range(length)]
    node_types = [G.nodes[n].get('type', '') for n in path]
    return {
        "path_len": length,
        "avg_weight": sum(weights) / length if length else 0,
        "max_weight": max(weights) if weights else 0,
        "distinct_node_types": len(set(node_types)),
        "distinct_rel_types": len(set(labels))
    }

def train_path_classifier(G, paths_with_labels):
    features, labels = [], []
    for entry, label in paths_with_labels:
        feat = extract_path_features(entry['path'], G)
        features.append(feat)
        labels.append(label)
    df = pd.DataFrame(features)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(df, labels)
    return clf

def predict_path_risks(G, paths, clf):
    results = []
    for entry in paths:
        path = [node['id'] for node in entry['path']['nodes']]
        feat = extract_path_features(path, G)
        prob = clf.predict_proba(pd.DataFrame([feat]))[0][1]
        results.append({"path": path, "risk_prob": prob})
    return sorted(results, key=lambda x: x['risk_prob'], reverse=True)