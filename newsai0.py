import time
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import DQN
from textblob import TextBlob
import requests
import gym
from gym import spaces
import random

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"
NEWS_API_KEY = "9288c1beaa4740f28223d9cca0e2af5a"

# 손절 및 익절 비율 설정
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절

# 쿨다운 타임 설정
COOLDOWN_TIME = timedelta(minutes=5)

# 최근 매매 기록 저장 (쿨다운 타임 관리)
recent_trades = {}

# 진입가 저장
entry_prices = {}

# Upbit API 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 매수 함수
def buy_crypto_currency(ticker, amount):
    try:
        result = upbit.buy_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매수 에러: {e}")
        return None

# 매도 함수
def sell_crypto_currency(ticker, amount):
    try:
        result = upbit.sell_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매도 에러: {e}")
        return None

# 잔고 확인 함수
def get_balance(ticker):
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            return float(b.get('balance', 0))
    return 0

# MACD 계산 함수
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

# RSI 계산 함수
def get_rsi(ticker, period=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# Bollinger Bands 계산 함수
def get_bollinger_bands(ticker, window=20, num_std=2):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    sma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band.iloc[-1], lower_band.iloc[-1], sma.iloc[-1]

# Moving Average 계산 함수
def get_moving_average(ticker, window=50):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    sma = df['close'].rolling(window=window).mean()
    return sma.iloc[-1]

# 뉴스 감성 분석
def analyze_news_sentiment():
    news_url = f'https://newsapi.org/v2/everything?q=crypto&apiKey={NEWS_API_KEY}'
    response = requests.get(news_url)
    news_data = response.json()
    sentiment = 0
    for article in news_data['articles']:
        sentiment += TextBlob(article['title']).sentiment.polarity
    return sentiment

# AI 모델을 위한 환경 (OpenAI Gym)
class CryptoTradingEnv(gym.Env):
    def __init__(self, ticker, initial_balance=100000):
        super(CryptoTradingEnv, self).__init__()
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.stock_owned = 0
        self.entry_price = 0
        self.current_step = 0
        self.done = False

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.entry_price = 0
        self.current_step = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        macd, signal = get_macd(self.ticker)
        rsi = get_rsi(self.ticker)
        upper_band, lower_band, sma = get_bollinger_bands(self.ticker)
        moving_avg = get_moving_average(self.ticker)
        sentiment = analyze_news_sentiment()

        return np.array([macd, signal, rsi, upper_band, lower_band, sentiment], dtype=np.float32)

    def step(self, action):
        current_price = pyupbit.get_current_price(self.ticker)
        reward = 0

        if action == 1:  # Buy
            if self.balance > current_price:
                self.balance -= current_price
                self.stock_owned += 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.stock_owned > 0:
                self.balance += current_price
                self.stock_owned -= 1
                reward = current_price - self.entry_price  # 이익 또는 손실

        # 종료 조건 (매매가 종료될 때까지 반복)
        self.current_step += 1
        if self.current_step >= 100:  # 예시로 100번의 시간 단위로 끝냄
            self.done = True

        return self._next_observation(), reward, self.done, {}

# DQN 모델 훈련 함수
def train_ai_model(ticker):
    env = CryptoTradingEnv(ticker)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # 학습할 시간 단위 설정
    return model

# 거래량이 높은 상위 10개 코인 추출 함수
def get_top_10_tickers():
    tickers = pyupbit.get_tickers(fiat="KRW")
    volumes = []
    
    for ticker in tickers:
        df = pyupbit.get_ohlcv(ticker, interval="minute5", count=30)
        volume = df['volume'].sum()
        volumes.append((ticker, volume))
    
    volumes.sort(key=lambda x: x[1], reverse=True)
    return [ticker for ticker, _ in volumes[:10]]

# 매수 조건 추가: 저평가된 코인 (RSI 30 이하) 또는 급상승 (MACD, Bollinger Bands 등)
def should_buy(ticker):
    macd, signal = get_macd(ticker)
    rsi = get_rsi(ticker)
    upper_band, lower_band, sma = get_bollinger_bands(ticker)
    
    # 저평가된 코인: RSI < 30 (과매도 상태)
    if rsi < 30:
        return True
    
    # 급상승할 가능성이 있는 코인: MACD가 신호선 위로 교차하고 Bollinger Bands 상한선 돌파
    if macd > signal and upper_band > sma:
        return True
    
    return False

# 매도 조건 추가: 익절 (5%) 또는 손절 (-3%)
def should_sell(ticker, entry_price):
    current_price = pyupbit.get_current_price(ticker)
    change_ratio = (current_price - entry_price) / entry_price

    # 손절 및 익절
    if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
        return True
    
    return False

# 메인 로직
if __name__ == "__main__":
    print("AI 기반 자동매매 시작!")
    model = None
    tickers = get_top_10_tickers()  # 거래량 상위 10개 코인 조회
    
    for ticker in tickers:
        model = train_ai_model(ticker)  # 각 코인에 대해 AI 모델 훈련

    while True:
        for ticker in tickers:
            env = CryptoTradingEnv(ticker)
            observation = env.reset()

            # 매수 조건을 AI 모델과 기술적 지표 기반으로 결정
            if should_buy(ticker):
                buy_crypto_currency(ticker, env.balance * 0.1)  # 잔고의 10%만 매수
            
            # 매도 조건을 AI 모델과 기술적 지표 기반으로 결정
            if should_sell(ticker, env.entry_price):
                sell_crypto_currency(ticker, env.stock_owned)

        time.sleep(60)  # 1분마다 주기적으로 매매 진행

  

