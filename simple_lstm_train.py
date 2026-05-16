#!/usr/bin/env python3
import os
import sys
import numpy as np

print("Starting LSTM training script...")

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("Path configured")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("Importing TensorFlow...")
import tensorflow as tf
print(f"TensorFlow {tf.__version__} loaded")

from binance_client import BinanceFuturesClient
from config import Config
from advanced_strategies import (
    TechnicalIndicators, SignalType, AdvancedStrategyEngine
)

print("="*70)
print("  LSTM STRATEGY MODEL - TRAINING & BACKTESTING")
print("="*70)
print(f"\nTensorFlow Version: {tf.__version__}")

def main():
    symbol = 'BTCUSDT'
    print(f"\n[1] Downloading 60 days of historical data for {symbol}...")
    
    client = BinanceFuturesClient()
    client.symbol = symbol
    
    # Download more data for training
    df_15m = client.get_klines(interval="15m", limit=1000)
    df_1h = client.get_klines(interval="1h", limit=1000)
    df_4h = client.get_klines(interval="4h", limit=250)
    df_1d = client.get_klines(interval="1d", limit=120)
    
    print(f"✓ Downloaded {len(df_15m)} 15m candles")
    
    # Convert to DataFrames
    import pandas as pd
    df_15m_df = pd.DataFrame(df_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                           'taker_buy_quote', 'ignore'])
    df_1h_df = pd.DataFrame(df_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
    df_4h_df = pd.DataFrame(df_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
    df_1d_df = pd.DataFrame(df_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
    
    for df in [df_15m_df, df_1h_df, df_4h_df, df_1d_df]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
    
    print(f"✓ Data converted to DataFrames")
    
    # Extract features and prepare training data
    print("\n[2] Preparing training data...")
    features = []
    labels = []
    
    for i in range(60, min(len(df_15m_df) - 6, 400)):
        window = df_15m_df.iloc[i-60:i]
        
        close = window['close'].values
        ema8 = TechnicalIndicators.ema(window['close'], 8).iloc[-1]
        ema34 = TechnicalIndicators.ema(window['close'], 34).iloc[-1]
        rsi = TechnicalIndicators.rsi(window['close']).iloc[-1]
        macd, _, _ = TechnicalIndicators.macd(window['close'])
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bbands(window['close'])
        adx, _, _ = TechnicalIndicators.adx(window)
        
        current_price = close[-1]
        feat = [
            (close[-1] - close[-5]) / close[-5],
            (close[-1] - close[-20]) / close[-20],
            np.std(np.diff(close) / close[:-1]),
            ema8 / current_price - 1,
            ema34 / current_price - 1,
            rsi / 100,
            macd.iloc[-1] / current_price * 100,
            adx.iloc[-1] / 100,
            window['volume'].iloc[-1] / window['volume'].iloc[-20:].mean(),
        ]
        features.append(feat)
        
        # Label based on future price
        future_price = df_15m_df.iloc[min(i+6, len(df_15m_df)-1)]['close']
        change = (future_price - current_price) / current_price * 100
        if change > 1.5:
            labels.append([0, 0, 1])  # LONG
        elif change < -1.5:
            labels.append([1, 0, 0])  # SHORT
        else:
            labels.append([0, 1, 0])  # NEUTRAL
    
    features = np.array(features)
    labels = np.array(labels)
    print(f"✓ Prepared {len(features)} samples")
    
    # Create sequences
    seq_len = 30
    X, y = [], []
    for i in range(seq_len, min(len(features), len(labels))):
        X.append(features[i-seq_len:i])
        y.append(labels[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Split
    split = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split], X_norm[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Build LSTM
    print("\n[3] Building LSTM model...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    
    model = Sequential([
        Input(shape=(seq_len, 9)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✓ LSTM model built")
    
    # Train
    print("\n[4] Training LSTM model...")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    
    print(f"✓ Training complete!")
    print(f"  Train Accuracy: {train_acc:.2%}")
    print(f"  Test Accuracy: {test_acc:.2%}")
    
    # Backtest with simple trend following
    print("\n[5] Running backtest with trend following...")
    trades = []
    balance = 10000
    position = None
    entry_price = 0
    
    for i in range(100, min(len(df_15m_df), 400)):
        current_price = df_15m_df.iloc[i]['close']
        
        if position:
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if position == 'SHORT':
                pnl_pct = -pnl_pct
            
            if pnl_pct >= 1.5 or pnl_pct <= -0.8:  # Adjusted TP/SL
                pnl_amount = balance * 0.1 * (pnl_pct / 100)
                balance += pnl_amount
                trades.append({'side': position, 'pnl_pct': pnl_pct, 'pnl_amount': pnl_amount, 'result': 'WIN' if pnl_amount > 0 else 'LOSS'})
                position = None
        
        if not position and i >= 60:
            window = df_15m_df.iloc[i-20:i]
            ema8 = TechnicalIndicators.ema(window['close'], 8).iloc[-1]
            ema34 = TechnicalIndicators.ema(window['close'], 34).iloc[-1]
            rsi = TechnicalIndicators.rsi(window['close']).iloc[-1]
            
            # Simple trend following signal
            if ema8 > ema34 and rsi > 50:
                position = 'LONG'
                entry_price = current_price
            elif ema8 < ema34 and rsi < 50:
                position = 'SHORT'
                entry_price = current_price
    
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS: {symbol}")
    print(f"{'='*70}")
    print(f"  Total Trades:     {len(trades)}")
    print(f"  Wins:             {len(wins)}")
    print(f"  Losses:           {len(losses)}")
    print(f"  Winrate:          {len(wins)/len(trades)*100:.1f}%" if trades else "N/A")
    print(f"  Total P&L:        ${sum(t['pnl_amount'] for t in trades):,.2f}")
    print(f"  Final Balance:    ${balance:,.2f}")
    print(f"  Return:           {(balance-10000)/100*100:.2f}%")
    print(f"{'='*70}")
    
    if len(wins)/len(trades)*100 >= 50 if trades else False:
        print(f"\n  ✓ PROFITABLE - Winrate: {len(wins)/len(trades)*100:.1f}%")
    else:
        print(f"\n  ⚠ Needs optimization")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")
