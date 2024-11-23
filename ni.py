import time
import pyupbit
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments, pipeline
from stable_baselines3 import SAC
import gym
from gym import spaces
from sklearn.preprocessing import MinMaxScaler
import requests
from datasets import load_dataset

# Upbit API Key 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# News API Key 설정
NEWS_API_KEY = "9288c1beaa4740f28223d9cca0e2af5a"

# 모델 및 거래 설정
STOP_LOSS_THRESHOLD = -0.03
TAKE_PROFIT_THRESHOLD = 0.05
TRADE_COOLDOWN = timedelta(minutes=5)
recent_trades = {}

# Pyupbit 객체
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 뉴스 분석용 Hugging Face 감정 분석 파이프라인 설정
news_sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased", device=0 if torch.cuda.is_available() else -1)

# 거래량 상위 및 급상승 예상 코인 선택
def get_top_and_trending_tickers(top_n=10):
    tickers = pyupbit.get_tickers(fiat="KRW")
    volumes = []

    for ticker in tickers:
        df = pyupbit.get_ohlcv(ticker, interval="minute5", count=30)
        if df is not None:
            volume = df['volume'].sum()
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            volumes.append((ticker, volume, price_change))

    top_tickers = sorted(volumes, key=lambda x: x[1], reverse=True)[:top_n]
    trending_tickers = sorted(volumes, key=lambda x: x[2], reverse=True)[:top_n]
    return list(set([t[0] for t in top_tickers + trending_tickers]))

# 뉴스 데이터 가져오기 및 분석
def fetch_and_analyze_news(ticker, max_articles=5):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])[:max_articles]
        sentiment_scores = []
        for article in articles:
            text = article.get("title", "") + " " + article.get("description", "")
            sentiment = news_sentiment_analyzer(text)[0]
            sentiment_scores.append(1 if sentiment['label'] == "POSITIVE" else -1)

        # 뉴스 감정 평균 점수 반환
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return 0

# 강화학습 환경 정의
class CryptoTradingEnv(gym.Env):
    def __init__(self, ticker, initial_balance=100000, max_steps=100):
        super(CryptoTradingEnv, self).__init__()
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.stock_owned = 0
        self.entry_price = 0
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.entry_price = 0
        self.current_step = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        df = pyupbit.get_ohlcv(self.ticker, interval="minute5", count=50)
        macd, signal = self._get_macd(df)
        rsi = self._get_rsi(df)
        upper_band, lower_band, sma = self._get_bollinger_bands(df)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
        news_score = fetch_and_analyze_news(self.ticker)  # 뉴스 감정 점수 추가

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform([[macd, signal, rsi, upper_band, lower_band, price_change, news_score]])
        return scaled_features.flatten()

    def _get_macd(self, df, short_window=12, long_window=26, signal_window=9):
        short_ema = df['close'].ewm(span=short_window).mean()
        long_ema = df['close'].ewm(span=long_window).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def _get_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _get_bollinger_bands(self, df, window=20, num_std=2):
        sma = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]

    def step(self, action):
        current_price = pyupbit.get_current_price(self.ticker)
        reward = 0

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.balance -= current_price
                self.stock_owned += 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                self.balance += current_price * self.stock_owned
                reward = (current_price - self.entry_price) * self.stock_owned
                self.stock_owned = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        return self._get_observation(), reward, self.done, {}

# 모델 로드 및 Fine-tuning
def fine_tune_distilbert():
    dataset = load_dataset("imdb")

    # DistilBERT 모델 로드
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 훈련 파라미터 설정
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    trainer.train()
    return model, tokenizer

# 강화학습 모델 학습
def train_sac_model(ticker, model, tokenizer):
    env = CryptoTradingEnv(ticker)
    sac_model = SAC("MlpPolicy", env, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")
    sac_model.learn(total_timesteps=100000)
    return sac_model

# 메인 실행
if __name__ == "__main__":
    print("AI 기반 자동 매매 시작!")
    
    # Fine-tuning 모델 준비
    model, tokenizer = fine_tune_distilbert()
    tickers = get_top_and_trending_tickers()
    models = {}

    for ticker in tickers:
        print(f"학습 중: {ticker}")
        models[ticker] = train_sac_model(ticker, model, tokenizer)

    while True:
        for ticker in tickers:
            if ticker in recent_trades and datetime.now() - recent_trades[ticker] < TRADE_COOLDOWN:
                continue

            obs = models[ticker].env.reset()
            action, _ = models[ticker].predict(obs)
            if action == 1:  # Buy
                buy_crypto_currency(ticker, 10000)
                recent_trades[ticker] = datetime.now()
            elif action == 2:  # Sell
                sell_crypto_currency(ticker, 10000)
                recent_trades[ticker] = datetime.now()

        time.sleep(60)
