"""
LSTM Model for Advanced Trading Strategies
Deep learning model with backtesting and winrate calculation
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

from advanced_strategies import (
    TechnicalIndicators, SignalType, AdvancedStrategyEngine
)
from binance_client import BinanceFuturesClient
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting"""
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    winrate: float
    total_profit: float
    total_loss: float
    net_pnl: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict]


class LSTMStrategyModel:
    """LSTM model for predicting strategy signals"""
    
    def __init__(self, symbol: str, sequence_length: int = 60):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.model_dir = "backend/ml_models/lstm"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.strategy_engine = AdvancedStrategyEngine()
        
        # Model parameters
        self.lstm_units = 128
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 50
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        model = Sequential([
            # First LSTM layer
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, return_sequences=True),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Third LSTM layer
            LSTM(self.lstm_units // 4, return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 classes: SHORT, NEUTRAL, LONG
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_data(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        logger.info(f"[{self.symbol}] Preparing LSTM data...")
        
        # Calculate features (same as advanced strategies)
        features = []
        
        for i in range(len(df_15m)):
            if i < 60:
                continue
            
            window_5m = df_5m.iloc[max(0, i*3-180):i*3].copy() if len(df_5m) > 0 else df_5m.copy()
            window_15m = df_15m.iloc[i-60:i].copy()
            idx_1h = min(i // 4, len(df_1h) - 1)
            idx_4h = min(i // 16, len(df_4h) - 1)
            
            window_1h = df_1h.iloc[max(0, idx_1h-15):idx_1h+1].copy() if len(df_1h) > 0 else df_1h.copy()
            window_4h = df_4h.iloc[max(0, idx_4h-5):idx_4h+1].copy() if len(df_4h) > 0 else df_4h.copy()
            
            # Technical indicators
            close = window_15m['close'].values
            high = window_15m['high'].values
            low = window_15m['low'].values
            volume = window_15m['volume'].values
            
            # Price-based features
            returns = np.diff(close) / close[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0
            
            # EMAs
            ema8 = TechnicalIndicators.ema(window_15m['close'], 8).iloc[-1]
            ema34 = TechnicalIndicators.ema(window_15m['close'], 34).iloc[-1]
            ema50 = TechnicalIndicators.ema(window_15m['close'], 50).iloc[-1]
            
            # RSI
            rsi = TechnicalIndicators.rsi(window_15m['close']).iloc[-1]
            
            # MACD
            macd, macd_signal, macd_hist = TechnicalIndicators.macd(window_15m['close'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bbands(window_15m['close'])
            
            # ADX
            adx, plus_di, minus_di = TechnicalIndicators.adx(window_15m)
            
            # Stochastic
            stoch_k, stoch_d = TechnicalIndicators.stochastic(window_15m)
            
            # Multi-timeframe features
            close_price = close[-1]
            if len(window_5m) > 0:
                close_5m = window_5m['close'].values
                high_5m = window_5m['high'].values
                low_5m = window_5m['low'].values
                volume_5m = window_5m['volume'].values
                five_min_return = (close_5m[-1] - close_5m[-2]) / close_5m[-2] if len(close_5m) >= 2 else 0
                five_min_volatility = np.std(np.diff(close_5m) / close_5m[:-1]) if len(close_5m) > 1 else 0
                five_min_range = (high_5m[-1] - low_5m[-1]) / close_5m[-1] if len(close_5m) > 0 else 0
                five_min_volume_ratio = volume_5m[-1] / np.mean(volume_5m[-20:]) if len(volume_5m) >= 20 else 1
            else:
                five_min_return = 0
                five_min_volatility = 0
                five_min_range = 0
                five_min_volume_ratio = 1
            
            feature_vector = [
                # Price action (4)
                (close[-1] - close[-2]) / close[-2],  # Last return
                (close[-1] - close[-5]) / close[-5],  # 5-period return
                volatility,
                (high[-1] - low[-1]) / close[-1],  # Range
                
                # 5m candle window features (4)
                five_min_return,
                five_min_volatility,
                five_min_range,
                five_min_volume_ratio,
                
                # EMAs (4)
                ema8 / close_price - 1,
                ema34 / close_price - 1,
                ema50 / close_price - 1,
                1 if ema8 > ema34 else 0,  # EMA cross
                
                # RSI (3)
                rsi / 100,
                1 if 40 < rsi < 60 else 0,  # Neutral zone
                (rsi - 50) / 50,  # RSI deviation
                
                # MACD (3)
                macd.iloc[-1] / close_price * 100,
                macd_hist.iloc[-1] / close_price * 100,
                1 if macd.iloc[-1] > macd_signal.iloc[-1] else 0,
                
                # Bollinger Bands (3)
                (close_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]),
                (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1],
                1 if close_price > bb_upper.iloc[-1] else (1 if close_price < bb_lower.iloc[-1] else 0),
                
                # ADX (4)
                adx.iloc[-1] / 100,
                1 if adx.iloc[-1] > 25 else 0,
                plus_di.iloc[-1] / 100,
                minus_di.iloc[-1] / 100,
                
                # Stochastic (3)
                stoch_k.iloc[-1] / 100,
                stoch_d.iloc[-1] / 100,
                1 if stoch_k.iloc[-1] < 20 else (1 if stoch_k.iloc[-1] > 80 else 0),
                
                # Volume (3)
                volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1,
                np.log(volume[-1] + 1),
                volume[-1] * close_price,
            ]
            
            features.append(feature_vector)
        
        # Convert to numpy array
        features = np.array(features)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])
            
            # Label based on future price movement
            current_price = df_15m.iloc[i + 60 if i < len(df_15m) - 60 else len(df_15m) - 1]['close']
            future_price = df_15m.iloc[i + 60 if i < len(df_15m) - 60 else len(df_15m) - 1]['close']
            
            price_change = (future_price - current_price) / current_price * 100
            
            # One-hot encoding: [SHORT, NEUTRAL, LONG]
            if price_change > 2.0:
                y.append([0, 0, 1])  # LONG
            elif price_change < -2.0:
                y.append([1, 0, 0])  # SHORT
            else:
                y.append([0, 1, 0])  # NEUTRAL
        
        return np.array(X), np.array(y)
    
    def train(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Dict:
        """Train LSTM model"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training LSTM Model for {self.symbol}")
        logger.info(f"{'='*60}")
        
        # Prepare data
        X, y = self.prepare_data(df_5m, df_15m, df_1h, df_4h)
        
        if len(X) < 100:
            logger.warning(f"[{self.symbol}] Insufficient data for LSTM training")
            return {'success': False, 'error': 'Insufficient data'}
        
        logger.info(f"[{self.symbol}] Training data shape: {X.shape}")
        logger.info(f"[{self.symbol}] Labels distribution: SHORT={sum(y[:,0])}, NEUTRAL={sum(y[:,1])}, LONG={sum(y[:,2])}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        logger.info(f"[{self.symbol}] Model architecture built")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        logger.info(f"[{self.symbol}] Training LSTM...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"[{self.symbol}] Training accuracy: {train_acc:.2%}")
        logger.info(f"[{self.symbol}] Test accuracy: {test_acc:.2%}")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"lstm_{self.symbol}.h5")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.symbol}.pkl")
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"[{self.symbol}] ✓ Model saved to {model_path}")
        
        return {
            'success': True,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'epochs_trained': len(history.history['loss']),
            'samples': len(X)
        }
    
    def load(self) -> bool:
        """Load trained model"""
        model_path = os.path.join(self.model_dir, f"lstm_{self.symbol}.h5")
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.symbol}.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
        
        try:
            self.model = load_model(model_path)
            
            import pickle
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"[{self.symbol}] ✓ LSTM model loaded")
            return True
        except Exception as e:
            logger.error(f"[{self.symbol}] Failed to load model: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """Predict signal and confidence"""
        if self.model is None:
            return 1, 0.0  # NEUTRAL, 0% confidence
        
        prediction = self.model.predict(X, verbose=0)
        pred_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return pred_class, confidence


class LSTMBacktester:
    """Backtest LSTM model with strategy signals"""
    
    def __init__(self, symbol: str, initial_balance: float = 10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.model = LSTMStrategyModel(symbol)
    
    def download_historical_data(self, days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Download extensive historical data for backtesting"""
        logger.info(f"[{self.symbol}] Downloading {days} days of historical data...")
        
        client = BinanceFuturesClient()
        client.symbol = self.symbol
        
        # Calculate required candles
        candles_5m = days * 24 * 12  # 5m candles
        candles_15m = days * 24 * 4  # 15m candles
        candles_1h = days * 24       # 1h candles
        candles_4h = days * 6          # 4h candles
        
        klines_5m = client.get_klines(interval="5m", limit=min(candles_5m, 1000))
        klines_15m = client.get_klines(interval="15m", limit=min(candles_15m, 1000))
        klines_1h = client.get_klines(interval="1h", limit=min(candles_1h, 1000))
        klines_4h = client.get_klines(interval="4h", limit=min(candles_4h, 1000))
        
        def klines_to_df(klines):
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        
        return klines_to_df(klines_5m), klines_to_df(klines_15m), klines_to_df(klines_1h), klines_to_df(klines_4h)
    
    def run_backtest(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                     min_confidence: float = 0.6, use_strategies: bool = True) -> BacktestResult:
        """Run backtest simulation"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Backtest for {self.symbol}")
        logger.info(f"{'='*60}")
        
        trades = []
        balance = self.initial_balance
        max_balance = balance
        max_drawdown = 0
        
        position = None
        entry_price = 0
        
        # Warmup period
        start_idx = 100
        
        for i in range(start_idx, len(df_15m) - 1):
            current_price = df_15m.iloc[i]['close']
            current_time = df_15m.iloc[i]['timestamp']
            
            # Check for position exit
            if position:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                if position == 'SHORT':
                    pnl_pct = -pnl_pct
                
                # Exit conditions
                take_profit = 2.0  # 2% profit
                stop_loss = 1.0    # 1% loss
                
                should_exit = False
                exit_reason = ""
                
                if pnl_pct >= take_profit:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif pnl_pct <= -stop_loss:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                if should_exit:
                    pnl_amount = balance * (pnl_pct / 100) * 0.1  # 10% position size
                    balance += pnl_amount
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'side': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason,
                        'balance': balance
                    })
                    
                    position = None
                    
                    # Update max drawdown
                    if balance > max_balance:
                        max_balance = balance
                    drawdown = (max_balance - balance) / max_balance * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            
            # Check for entry signal
            if not position and i < len(df_15m) - 10:
                signal = None
                confidence = 0
                
                if use_strategies:
                    # Use 5-strategy confluence
                    window_15m = df_15m.iloc[:i+1].copy()
                    window_1h = df_1h.iloc[:min(i//4+1, len(df_1h))].copy()
                    window_4h = df_4h.iloc[:min(i//16+1, len(df_4h))].copy()
                    
                    result = self.model.strategy_engine.analyze_all(window_15m, window_1h, window_4h)
                    
                    if result.signal == SignalType.LONG and result.confidence >= min_confidence * 100:
                        signal = 'LONG'
                        confidence = result.confidence / 100
                    elif result.signal == SignalType.SHORT and result.confidence >= min_confidence * 100:
                        signal = 'SHORT'
                        confidence = result.confidence / 100
                
                if signal:
                    position = signal
                    entry_price = current_price
                    entry_time = current_time
        
        # Close any open position at end
        if position:
            final_price = df_15m.iloc[-1]['close']
            pnl_pct = (final_price - entry_price) / entry_price * 100
            if position == 'SHORT':
                pnl_pct = -pnl_pct
            
            pnl_amount = balance * (pnl_pct / 100) * 0.1
            balance += pnl_amount
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df_15m.iloc[-1]['timestamp'],
                'side': position,
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount,
                'exit_reason': 'END_OF_DATA',
                'balance': balance
            })
        
        # Calculate statistics
        winning_trades = [t for t in trades if t['pnl_amount'] > 0]
        losing_trades = [t for t in trades if t['pnl_amount'] <= 0]
        
        total_profit = sum(t['pnl_amount'] for t in winning_trades)
        total_loss = abs(sum(t['pnl_amount'] for t in losing_trades))
        
        return BacktestResult(
            symbol=self.symbol,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            winrate=len(winning_trades) / len(trades) * 100 if trades else 0,
            total_profit=total_profit,
            total_loss=total_loss,
            net_pnl=balance - self.initial_balance,
            avg_profit=total_profit / len(winning_trades) if winning_trades else 0,
            avg_loss=total_loss / len(losing_trades) if losing_trades else 0,
            profit_factor=total_profit / total_loss if total_loss > 0 else float('inf'),
            max_drawdown=max_drawdown,
            sharpe_ratio=0,  # Simplified
            trades=trades
        )
    
    def print_results(self, result: BacktestResult):
        """Print backtest results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS: {result.symbol}")
        print(f"{'='*60}")
        print(f"Total Trades:     {result.total_trades}")
        print(f"Winning Trades:   {result.winning_trades}")
        print(f"Losing Trades:    {result.losing_trades}")
        print(f"Winrate:          {result.winrate:.1f}%")
        print(f"")
        print(f"Total Profit:     ${result.total_profit:,.2f}")
        print(f"Total Loss:       ${result.total_loss:,.2f}")
        print(f"Net P&L:          ${result.net_pnl:,.2f}")
        print(f"Profit Factor:    {result.profit_factor:.2f}")
        print(f"Max Drawdown:     {result.max_drawdown:.1f}%")
        print(f"{'='*60}\n")


class LSTMTrainer:
    """Main trainer class for all symbols"""
    
    def __init__(self):
        self.symbols = Config.TOP_20_SYMBOLS[:5]  # Top 5 for faster training
        self.backtest_days = 60
        
    def train_and_backtest_all(self):
        """Train LSTM and run backtest for all symbols"""
        results = {}
        
        for symbol in self.symbols:
            try:
                backtester = LSTMBacktester(symbol)
                
                # Download data
                df_5m, df_15m, df_1h, df_4h = backtester.download_historical_data(self.backtest_days)
                
                # Train model
                model = LSTMStrategyModel(symbol)
                train_result = model.train(df_5m, df_15m, df_1h, df_4h)
                
                if train_result['success']:
                    # Run backtest
                    backtest_result = backtester.run_backtest(df_15m, df_1h, df_4h)
                    backtester.print_results(backtest_result)
                    
                    results[symbol] = {
                        'train': train_result,
                        'backtest': backtest_result
                    }
                else:
                    logger.warning(f"[{symbol}] Training failed: {train_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"[{symbol}] Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print(f"{'='*60}")
        
        total_trades = sum(r['backtest'].total_trades for r in results.values() if 'backtest' in r)
        total_wins = sum(r['backtest'].winning_trades for r in results.values() if 'backtest' in r)
        total_pnl = sum(r['backtest'].net_pnl for r in results.values() if 'backtest' in r)
        
        print(f"Symbols Trained:  {len(results)}")
        print(f"Total Trades:     {total_trades}")
        print(f"Total Wins:       {total_wins}")
        print(f"Overall Winrate:  {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A")
        print(f"Total P&L:        ${total_pnl:,.2f}")
        print(f"{'='*60}\n")
        
        return results


if __name__ == "__main__":
    trainer = LSTMTrainer()
    trainer.train_and_backtest_all()
