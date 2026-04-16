"""
Enhanced Crypto-Specific ML Model
Optimised for BTC, ETH, and major altcoins with crypto-specific features

Changes vs original:
- train() now accepts optional market_hints dict (from AdaptiveMLTrainer)
  to bias lookahead and label thresholds toward current market structure.
- Label thresholds are tightened/relaxed based on ATR regime.
- Sample weights give 2× weight to candles from the most recent 20 % of the
  window so the model adapts faster to recent conditions.
"""

import json
import os
import logging
import warnings
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

logger = logging.getLogger(__name__)


class EnhancedCryptoModel:
    """
    Advanced model specifically designed for crypto markets.
    Uses ensemble of RandomForest + GradientBoosting.
    """

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.training_date = None
        self.symbol_performance = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ─────────────────────────────────────────────────────────────────────────

    def extract_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        # Price-action
        features["close"] = df["close"]
        features["returns_1h"] = df["close"].pct_change(1)
        features["returns_4h"] = df["close"].pct_change(4)
        features["returns_12h"] = df["close"].pct_change(12)
        features["returns_24h"] = df["close"].pct_change(24)

        # Volatility
        features["volatility_12h"] = features["returns_1h"].rolling(12).std()
        features["volatility_24h"] = features["returns_1h"].rolling(24).std()

        # EMA (Fibonacci windows)
        for window in [8, 13, 21, 34, 55, 89]:
            ema = EMAIndicator(close=df["close"], window=window).ema_indicator()
            features[f"ema_{window}_gap"] = (df["close"] - ema) / ema
            features[f"ema_{window}_slope"] = ema.diff(3) / ema

        # RSI multi-window
        for window in [6, 14, 21]:
            rsi = RSIIndicator(close=df["close"], window=window).rsi()
            features[f"rsi_{window}"] = rsi
            features[f"rsi_{window}_mom"] = rsi.diff(3)

        # MACD
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        features["macd"] = macd.macd()
        features["macd_signal"] = macd.macd_signal()
        features["macd_hist"] = macd.macd_diff()
        features["macd_cross"] = (features["macd"] > features["macd_signal"]).astype(int)

        # Bollinger Bands
        bb = BollingerBands(close=df["close"], window=20, window_dev=2)
        features["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
        features["bb_position"] = (df["close"] - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-10
        )

        # ATR
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        features["atr_14"] = atr.average_true_range() / df["close"]
        features["atr_ratio"] = features["atr_14"] / features["atr_14"].rolling(24).mean()

        # Volume
        features["volume"] = df["volume"]
        features["volume_sma_20"] = df["volume"].rolling(20).mean()
        features["volume_ratio"] = df["volume"] / features["volume_sma_20"]
        features["volume_trend"] = features["volume_ratio"].rolling(6).mean()

        # OBV
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        features["obv"] = obv.on_balance_volume()
        features["obv_slope"] = features["obv"].diff(6) / (features["obv"].abs().mean() + 1e-10)

        # VWAP deviation
        try:
            vwap = VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                window=24,
            )
            features["vwap_dev"] = (
                df["close"] - vwap.volume_weighted_average_price()
            ) / df["close"]
        except Exception:
            features["vwap_dev"] = 0.0

        # ADX
        adx = ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        features["adx"] = adx.adx()
        features["adx_pos"] = adx.adx_pos()
        features["adx_neg"] = adx.adx_neg()

        # Candle patterns
        features["body_pct"] = (df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-10
        )
        features["upper_wick"] = (
            df["high"] - df[["open", "close"]].max(axis=1)
        ) / (df["high"] - df["low"] + 1e-10)
        features["lower_wick"] = (
            df[["open", "close"]].min(axis=1) - df["low"]
        ) / (df["high"] - df["low"] + 1e-10)

        # Time (crypto is 24/7)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            features["hour"] = ts.dt.hour
            features["day_of_week"] = ts.dt.dayofweek
            features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

        # Lagged returns
        for lag in [1, 2, 3, 6]:
            features[f"return_lag_{lag}"] = features["returns_1h"].shift(lag)

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Label creation — now respects market_hints
    # ─────────────────────────────────────────────────────────────────────────

    def create_labels(
        self,
        df: pd.DataFrame,
        lookahead: int = 6,
        min_return: float = 0.005,
    ) -> pd.Series:
        """
        Create labels:
          1  = profitable LONG after fees
         -1  = profitable SHORT after fees
          0  = no trade / not profitable

        lookahead and min_return are tuned by market_hints at call-time.
        """
        fee = 0.0004  # 0.04 % taker fee (round trip)

        future_returns = df["close"].pct_change(lookahead).shift(-lookahead)
        future_high = df["high"].rolling(lookahead).max().shift(-lookahead)
        future_low = df["low"].rolling(lookahead).min().shift(-lookahead)

        entry_price = df["close"]
        long_potential = (future_high - entry_price) / entry_price - fee
        short_potential = (entry_price - future_low) / entry_price - fee

        labels = pd.Series(0, index=df.index)
        long_signals = (long_potential > min_return) & (long_potential > short_potential)
        labels[long_signals] = 1
        short_signals = (short_potential > min_return) & (short_potential > long_potential)
        labels[short_signals] = -1

        return labels

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        candles,
        symbol: str,
        market_hints: Optional[dict] = None,
    ) -> Dict:
        """
        Train model on historical data.

        market_hints (optional dict from AdaptiveMLTrainer) may contain:
          - bias       : "bullish" / "bearish" / "neutral"
          - features   : { vol_regime, trend_strength, atr, … }
        """
        logger.info(f"Training enhanced model for {symbol}...")

        hints = market_hints or {}
        mf = hints.get("features", {})
        bias = hints.get("bias", "neutral")

        # ── Tune lookahead based on vol regime ──────────────────────────────
        vol_regime = mf.get("vol_regime", "normal")
        if vol_regime == "high":
            lookahead = 4    # shorter horizon when volatile
            min_return = 0.008
        elif vol_regime == "low":
            lookahead = 8    # longer horizon in quiet markets
            min_return = 0.003
        else:
            lookahead = 6
            min_return = 0.005

        logger.info(
            f"  Hints: bias={bias} vol={vol_regime} "
            f"lookahead={lookahead} min_return={min_return:.3f}"
        )

        # ── Parse candles ────────────────────────────────────────────────────
        if isinstance(candles, pd.DataFrame):
            candles = candles[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ].values.tolist()

        if not candles or len(candles) == 0:
            raise ValueError("No candles provided")

        if isinstance(candles[0], dict):
            needed = ["timestamp", "open", "high", "low", "close", "volume"]
            rows = [{col: c.get(col, 0) for col in needed} for c in candles]
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(
                candles,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

        if len(df) < 200:
            raise ValueError(f"Insufficient data: {len(df)} rows (need 200+)")

        # ── Features & labels ────────────────────────────────────────────────
        features = self.extract_crypto_features(df)
        labels = self.create_labels(df, lookahead=lookahead, min_return=min_return)

        # Align
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx].values
        y = labels.loc[valid_idx].values

        if len(X) < 100:
            raise ValueError(f"Too few valid samples: {len(X)}")

        # ── Sample weights: 2× for most recent 20 % of data ─────────────────
        n = len(X)
        recent_cutoff = int(n * 0.80)
        sample_weights = np.ones(n)
        sample_weights[recent_cutoff:] = 2.0

        # ── Bias: if we know the market is trending, up-weight same-direction ─
        if bias == "bullish":
            sample_weights[y == 1] *= 1.3
        elif bias == "bearish":
            sample_weights[y == -1] *= 1.3

        self.feature_names = list(features.columns)

        # ── Scale ────────────────────────────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X)

        # ── Train models ─────────────────────────────────────────────────────
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.rf_model.fit(X_scaled, y, sample_weight=sample_weights)

        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.gb_model.fit(X_scaled, y, sample_weight=sample_weights)

        # ── Evaluate ─────────────────────────────────────────────────────────
        rf_score = self.rf_model.score(X_scaled, y)
        gb_score = self.gb_model.score(X_scaled, y)

        logger.info(f"  RandomForest  accuracy: {rf_score:.2%}")
        logger.info(f"  GradientBoost accuracy: {gb_score:.2%}")

        self.is_trained = True
        self.training_date = datetime.utcnow()

        return {
            "symbol": symbol,
            "samples": len(X),
            "rf_accuracy": rf_score,
            "gb_accuracy": gb_score,
            "features": len(self.feature_names),
            "training_date": self.training_date.isoformat(),
            "lookahead": lookahead,
            "min_return": min_return,
            "bias": bias,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, klines: List) -> Optional[Dict]:
        if not self.is_trained:
            return None

        if not klines:
            return None

        if isinstance(klines[0], dict):
            needed = ["timestamp", "open", "high", "low", "close", "volume"]
            rows = [{col: k.get(col, 0) for col in needed} for k in klines]
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ],
            )
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if len(df) < 100:
            return None

        features = self.extract_crypto_features(df)
        features = features.dropna()

        if len(features) == 0:
            return None

        latest_features = features.iloc[-1].values.reshape(1, -1)

        # Handle feature count mismatch gracefully
        if self.feature_names and len(latest_features[0]) != len(self.feature_names):
            logger.warning(
                f"Feature count mismatch: got {len(latest_features[0])}, "
                f"expected {len(self.feature_names)}"
            )
            return None

        latest_scaled = self.scaler.transform(latest_features)

        rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
        gb_proba = self.gb_model.predict_proba(latest_scaled)[0]

        classes = self.rf_model.classes_
        avg_proba = (rf_proba + gb_proba) / 2

        long_idx = list(classes).index(1) if 1 in classes else None
        short_idx = list(classes).index(-1) if -1 in classes else None
        neutral_idx = list(classes).index(0) if 0 in classes else None

        long_conf = avg_proba[long_idx] * 100 if long_idx is not None else 0
        short_conf = avg_proba[short_idx] * 100 if short_idx is not None else 0
        neutral_conf = avg_proba[neutral_idx] * 100 if neutral_idx is not None else 0

        if long_conf > short_conf and long_conf > neutral_conf:
            signal = "LONG"
            confidence = long_conf
        elif short_conf > long_conf and short_conf > neutral_conf:
            signal = "SHORT"
            confidence = short_conf
        else:
            return {
                "signal": None,
                "confidence": 0,
                "probas": {"long": long_conf, "short": short_conf, "neutral": neutral_conf},
            }

        probas = {"long": long_conf, "short": short_conf, "neutral": neutral_conf}

        if confidence < 75:
            return {
                "signal": None,
                "confidence": confidence,
                "probas": probas,
                "reason": "confidence_too_low",
            }

        latest = df.iloc[-1]
        atr = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range().iloc[-1]

        entry_price = float(latest["close"])

        if signal == "LONG":
            stop_loss = entry_price - atr * 1.0
            take_profit_1 = entry_price + atr * 1.5
            take_profit_2 = entry_price + atr * 2.5
        else:
            stop_loss = entry_price + atr * 1.0
            take_profit_1 = entry_price - atr * 1.5
            take_profit_2 = entry_price - atr * 2.5

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "probas": probas,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            "atr": round(float(atr), 2),
            "model_agreement": bool(abs(rf_proba.max() - gb_proba.max()) < 0.1),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "rf_model": self.rf_model,
                "gb_model": self.gb_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained,
                "training_date": self.training_date.isoformat()
                if self.training_date
                else None,
                "symbol_performance": self.symbol_performance,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EnhancedCryptoModel":
        import joblib

        data = joblib.load(path)
        model = cls()
        model.rf_model = data["rf_model"]
        model.gb_model = data["gb_model"]
        model.scaler = data["scaler"]
        model.feature_names = data["feature_names"]
        model.is_trained = data["is_trained"]
        model.training_date = (
            datetime.fromisoformat(data["training_date"])
            if data.get("training_date")
            else None
        )
        model.symbol_performance = data.get("symbol_performance", {})
        logger.info(f"Model loaded from {path}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Global cache helpers (backward-compatible)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache: Dict[str, EnhancedCryptoModel] = {}


def get_or_train_model(symbol: str, candles: List) -> EnhancedCryptoModel:
    if symbol in _model_cache:
        return _model_cache[symbol]
    model = EnhancedCryptoModel()
    model.train(candles, symbol)
    _model_cache[symbol] = model
    return model


def predict_with_enhanced_model(symbol: str, klines: List) -> Optional[Dict]:
    model = _model_cache.get(symbol)
    if not model:
        logger.warning(f"No model available for {symbol}")
        return None
    return model.predict(klines)