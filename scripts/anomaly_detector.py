from sklearn.ensemble import IsolationForest
import numpy as np

def detect_trends_and_anomalies(data):
    clf = IsolationForest(contamination=0.01)
    anomalies = clf.fit_predict(np.array(data).reshape(-1, 1))
    return anomalies