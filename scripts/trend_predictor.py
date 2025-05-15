import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

def load_threat_data(file='data/threat_data.csv'):
    df = pd.read_csv(file, parse_dates=['日期'])
    df.set_index('日期', inplace=True)
    return df

def extract_features(df):
    df['攻击频率'] = df['攻击次数'].rolling(window=7).mean()
    df['攻击类型变化'] = df['攻击类型'].ne(df['攻击类型'].shift()).cumsum()
    return df.dropna()

def train_lstm_model(data, timesteps=30):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    X = np.array(X).reshape((-1, timesteps, 1))
    y = np.array(y)
    model = Sequential([Input(shape=(timesteps, 1)), LSTM(50), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, verbose=0)
    return model

def predict_future(model, last_seq, days=7):
    preds = []
    current = last_seq.reshape((1, -1, 1))
    for _ in range(days):
        p = model.predict(current, verbose=0)
        preds.append(p[0, 0])
        current = np.roll(current, -1, axis=1)
        current[0, -1, 0] = p[0, 0]
    return preds