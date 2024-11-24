import time
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 설정값
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절
COOLDOWN_TIME = timedelta(minutes=5)
MODEL_PREDICTION_THRESHOLD = 0.02  # 트랜스포머 모델의 매수 신호 임계값

recent_trades = {}
entry_prices = {}

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_dim,
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (batch, seq, feature) -> (seq, batch, feature)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (seq, batch, feature) -> (batch, seq, feature)
        return self.fc(x[:, -1, :])  # 마지막 시점의 출력값 반환


# 데이터셋 정의
class CryptoDataset(Dataset):
    def __init__(self, data, seq_len=20):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx + self.seq_len].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# 데이터 준비 함수
def prepare_data(ticker, seq_len=20):
    """거래량 기반 지표와 함께 데이터 준비"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    df['return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1

    # NaN 제거
    df.dropna(inplace=True)
    dataset = CryptoDataset(df[['return', 'volume_change', 'vwap', 'obv']], seq_len=seq_len)
    return dataset


# Transformer 모델 학습
def train_transformer_model(ticker, epochs=10, batch_size=32):
    """Transformer 모델 학습"""
    dataset = prepare_data(ticker)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 4  # return, volume_change, vwap, obv
    embed_dim = 16
    num_heads = 4
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    model = TransformerModel(input_dim, embed_dim, num_heads, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{ticker}] Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    return model


# 매매 신호 생성 함수
def get_transformer_signal(model, ticker):
    """Transformer 모델로 매수 신호 예측"""
    try:
        dataset = prepare_data(ticker)
        x_latest, _ = dataset[-1]
        x_latest = x_latest.unsqueeze(0)  # (1, seq_len, feature)
        prediction = model(x_latest).item()
        return prediction
    except Exception as e:
        print(f"[{ticker}] Transformer 신호 계산 중 에러 발생: {e}")
        return 0


# 메인 로직
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    # 모든 코인에 대해 모델 초기화
    tickers = pyupbit.get_tickers(fiat="KRW")
    models = {}
    for ticker in tickers:
        print(f"[{ticker}] Transformer 모델 학습 중...")
        models[ticker] = train_transformer_model(ticker)

    try:
        while True:
            krw_balance = get_balance("KRW")

            for ticker in tickers:
                try:
                    now = datetime.now()

                    # 쿨다운 타임 체크
                    if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                        continue

                    # Transformer 신호 계산
                    model = models[ticker]
                    signal = get_transformer_signal(model, ticker)

                    # 매수 조건
                    if signal > MODEL_PREDICTION_THRESHOLD and krw_balance > 5000:
                        buy_amount = krw_balance * 0.1
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = pyupbit.get_current_price(ticker)
                            recent_trades[ticker] = now
                            print(f"[{ticker}] 매수 완료: {buy_amount:.2f} KRW")

                    # 매도 조건
                    elif ticker in entry_prices:
                        current_price = pyupbit.get_current_price(ticker)
                        entry_price = entry_prices[ticker]
                        change_ratio = (current_price - entry_price) / entry_price

                        if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
                            coin_balance = get_balance(ticker.split('-')[1])
                            sell_result = sell_crypto_currency(ticker, coin_balance)
                            if sell_result:
                                recent_trades[ticker] = now
                                print(f"[{ticker}] 매도 완료. 잔고: {coin_balance:.4f}")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
