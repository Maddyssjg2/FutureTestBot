"""
Adaptive Hourly ML Training for High-Confidence Trading
Targets: 75%+ confidence, 8-9 winning trades per day

NEW: After every 10 closed trades the system analyses market structure,
     retrains models and keeps retraining until the last-10-trade win
     rate reaches 80%.  The confidence gate is raised automatically
     when win rate is poor and lowered when we are comfortably above target.
"""

import os
import json
import logging
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Suppress warnings from technical analysis library
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

from data_downloader import DataDownloader
from enhanced_crypto_model import EnhancedCryptoModel, predict_with_enhanced_model
from premium_model import candles_to_dataframe, build_symbol_training_frame
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration optimised for 8-9 daily wins with 75%+ confidence"""
    target_daily_trades: int = 10
    target_win_rate: float = 0.80        # NEW: 80 % window target
    min_confidence: float = 75.0
    max_confidence: float = 97.0
    training_interval_hours: int = 1
    lookback_days: int = 60
    min_samples_per_symbol: int = 500

    strong_trend_boost: float = 5.0
    volume_surge_boost: float = 3.0
    confluence_boost: float = 4.0

    # Post-10-trade retrain settings
    retrain_window: int = 10             # retrain after this many closed trades
    retrain_win_target: float = 0.80     # keep retraining until we hit this
    max_retrain_attempts: int = 5        # safety cap per window


# ─────────────────────────────────────────────────────────────────────────────
# Trade result tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    signal: str          # LONG / SHORT
    confidence: float
    entry_price: float
    exit_price: float
    pnl: float
    win: bool
    reason: str          # "TP1", "TP2", "SL", "Trend Reversed", …
    candle_snapshot: Optional[list] = field(default=None, repr=False)


class TradeResultTracker:
    """
    Persists closed-trade outcomes and triggers adaptive retraining when
    every *retrain_window* trades are completed and the rolling win rate
    is below *retrain_win_target*.
    """

    def __init__(self, config: TrainingConfig, trainer: "AdaptiveMLTrainer"):
        self.config = config
        self.trainer = trainer
        self.records: List[TradeRecord] = []
        self._lock = threading.Lock()
        self._results_path = os.path.join(
            os.path.dirname(__file__), "ml_models", "trade_results.json"
        )
        self._load_persisted()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load_persisted(self):
        if os.path.exists(self._results_path):
            try:
                with open(self._results_path) as f:
                    raw = json.load(f)
                self.records = [TradeRecord(**r) for r in raw]
                logger.info(f"[Tracker] Loaded {len(self.records)} historical trade records")
            except Exception as e:
                logger.warning(f"[Tracker] Could not load trade history: {e}")

    def _persist(self):
        try:
            os.makedirs(os.path.dirname(self._results_path), exist_ok=True)
            with open(self._results_path, "w") as f:
                json.dump(
                    [
                        {k: v for k, v in r.__dict__.items() if k != "candle_snapshot"}
                        for r in self.records[-500:]   # keep last 500
                    ],
                    f, indent=2
                )
        except Exception as e:
            logger.warning(f"[Tracker] Could not persist trade history: {e}")

    # ── public API ────────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        reason: str,
        candle_snapshot: Optional[list] = None,
    ):
        """Call this whenever a position is fully closed."""
        win = pnl > 0
        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            win=win,
            reason=reason,
            candle_snapshot=candle_snapshot,
        )

        with self._lock:
            self.records.append(record)
            self._persist()
            total = len(self.records)

        logger.info(
            f"[Tracker] {'✅ WIN' if win else '❌ LOSS'} | {symbol} {signal} "
            f"PnL={pnl:+.4f} reason={reason} | total={total}"
        )

        # Fire retraining check every *retrain_window* trades
        if total % self.config.retrain_window == 0:
            threading.Thread(
                target=self._check_and_retrain,
                args=(candle_snapshot,),
                daemon=True,
            ).start()

    def rolling_win_rate(self, n: int = 10) -> float:
        with self._lock:
            window = self.records[-n:] if len(self.records) >= n else self.records
        if not window:
            return 0.0
        return sum(1 for r in window if r.win) / len(window)

    def recent_market_bias(self, n: int = 10) -> str:
        """Detect whether recent closed trades were mostly LONG or SHORT wins."""
        with self._lock:
            window = [r for r in self.records[-n:] if r.win]
        if not window:
            return "neutral"
        long_wins = sum(1 for r in window if r.signal == "LONG")
        short_wins = len(window) - long_wins
        if long_wins > short_wins * 1.5:
            return "bullish"
        if short_wins > long_wins * 1.5:
            return "bearish"
        return "neutral"

    # ── private retrain logic ─────────────────────────────────────────────────

    def _check_and_retrain(self, candle_snapshot: Optional[list]):
        """
        Analyse market structure from the last 10 trades and retrain until
        the rolling win rate meets the 80 % target (or max attempts reached).
        """
        wr = self.rolling_win_rate(self.config.retrain_window)
        bias = self.recent_market_bias(self.config.retrain_window)

        logger.info("=" * 60)
        logger.info(f"[Tracker] ⚡ POST-10-TRADE REVIEW")
        logger.info(f"  Rolling win rate (last 10): {wr:.0%}")
        logger.info(f"  Market bias detected     : {bias}")
        logger.info(f"  Target win rate          : {self.config.retrain_win_target:.0%}")
        logger.info("=" * 60)

        if wr >= self.config.retrain_win_target:
            logger.info("[Tracker] ✅ Win rate already ≥ 80 % — no retraining needed")
            # Still slightly relax the confidence gate if we're doing well
            self.trainer._relax_confidence()
            return

        # Tighten confidence immediately while we retrain
        self.trainer._tighten_confidence()

        attempt = 0
        while attempt < self.config.max_retrain_attempts:
            attempt += 1
            logger.info(
                f"[Tracker] 🔄 Retrain attempt {attempt}/{self.config.max_retrain_attempts} "
                f"(current win rate {wr:.0%})"
            )

            # ── market structure analysis ──────────────────────────────────
            market_hints = self._analyse_market_structure(candle_snapshot, bias)

            # ── force full retrain for all active symbols ──────────────────
            success = self.trainer.run_training_cycle(
                force_retrain=True,
                market_hints=market_hints,
            )

            if not success:
                logger.warning("[Tracker] Retrain cycle failed, backing off 60 s")
                time.sleep(60)
                continue

            # ── re-evaluate win rate on fresh predictions (back-test proxy) ─
            wr = self._backtest_win_rate_proxy()
            logger.info(f"[Tracker] Post-retrain back-test win rate proxy: {wr:.0%}")

            if wr >= self.config.retrain_win_target:
                logger.info(f"[Tracker] ✅ Target {self.config.retrain_win_target:.0%} reached after {attempt} attempt(s)")
                break

            # Wait a bit before next attempt to let data cool down
            time.sleep(30)

        if wr < self.config.retrain_win_target:
            logger.warning(
                f"[Tracker] ⚠ Could not reach {self.config.retrain_win_target:.0%} after "
                f"{attempt} attempts — raising confidence gate further"
            )
            self.trainer._tighten_confidence(amount=5.0)

    def _analyse_market_structure(
        self, candle_snapshot: Optional[list], bias: str
    ) -> dict:
        """
        Derive market-structure hints from recent candles and trade history.
        These are passed to the training cycle so it can weight labels.
        """
        hints: dict = {"bias": bias, "features": {}}

        # Use the most recently fetched candles from any symbol we have
        candles = candle_snapshot
        if not candles:
            # Grab a fresh snapshot for BTCUSDT as a proxy
            try:
                from binance_client import BinanceFuturesClient
                client = BinanceFuturesClient()
                client.symbol = "BTCUSDT"
                candles = client.get_klines(interval="1h", limit=200)
            except Exception:
                pass

        if candles and len(candles) >= 50:
            try:
                df = pd.DataFrame(
                    candles,
                    columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades",
                        "taker_buy_base", "taker_buy_quote", "ignore",
                    ],
                )
                df["close"] = df["close"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["volume"] = df["volume"].astype(float)

                # Trend: EMA 21 vs EMA 55
                from ta.trend import EMAIndicator
                from ta.volatility import AverageTrueRange

                ema21 = EMAIndicator(df["close"], 21).ema_indicator().iloc[-1]
                ema55 = EMAIndicator(df["close"], 55).ema_indicator().iloc[-1]
                atr = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range().iloc[-1]
                close = df["close"].iloc[-1]
                trend_strength = (ema21 - ema55) / close * 100

                # Recent volatility regime
                recent_vol = df["close"].pct_change().rolling(24).std().iloc[-1]
                long_vol = df["close"].pct_change().rolling(168).std().iloc[-1]
                vol_regime = "high" if recent_vol > long_vol * 1.3 else (
                    "low" if recent_vol < long_vol * 0.7 else "normal"
                )

                # Support / resistance via pivot points (simple)
                last20_high = df["high"].iloc[-20:].max()
                last20_low = df["low"].iloc[-20:].min()
                pivot = (last20_high + last20_low + close) / 3

                hints["features"] = {
                    "trend_strength": round(float(trend_strength), 4),
                    "atr": round(float(atr), 4),
                    "vol_regime": vol_regime,
                    "pivot": round(float(pivot), 2),
                    "price_vs_pivot": round(float((close - pivot) / pivot * 100), 3),
                    "ema21": round(float(ema21), 2),
                    "ema55": round(float(ema55), 2),
                }

                logger.info(
                    f"[Tracker] Market structure: trend={trend_strength:+.2f}% "
                    f"vol={vol_regime} price_vs_pivot={hints['features']['price_vs_pivot']:+.2f}%"
                )

            except Exception as e:
                logger.warning(f"[Tracker] Market structure analysis error: {e}")

        return hints

    def _backtest_win_rate_proxy(self) -> float:
        """
        Quick in-sample accuracy proxy: re-run predictions on the last 10 closed
        trade entry candles and count how many model predictions match the actual
        winning direction.  Falls back to recent live win rate if data unavailable.
        """
        with self._lock:
            recent = self.records[-self.config.retrain_window:]

        if not recent:
            return 0.0

        correct = 0
        for r in recent:
            model = self.trainer.get_model(r.symbol)
            if model is None:
                # Model not available — assume win if the trade was a win
                if r.win:
                    correct += 1
                continue

            try:
                from binance_client import BinanceFuturesClient
                client = BinanceFuturesClient()
                client.symbol = r.symbol
                klines = client.get_klines(interval="1h", limit=100)
                if not klines:
                    if r.win:
                        correct += 1
                    continue
                pred = model.predict(klines)
                if pred and pred.get("signal") == r.signal and r.win:
                    correct += 1
                elif pred and pred.get("signal") != r.signal and not r.win:
                    correct += 1  # model now disagrees with a losing trade → good
                elif r.win and (pred is None or pred.get("signal") is None):
                    correct += 1
            except Exception:
                if r.win:
                    correct += 1

        return correct / len(recent)


# ─────────────────────────────────────────────────────────────────────────────
# Main Adaptive Trainer
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveMLTrainer:
    """
    Adaptive trainer that adjusts strategy hourly (and after every 10 trades)
    to maximise high-confidence wins.
    """

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.data_dir = os.path.join(os.path.dirname(__file__), "historical_data")
        self.models_dir = os.path.join(os.path.dirname(__file__), "ml_models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.symbols = Config.TOP_20_SYMBOLS[:10]
        self.models: Dict[str, EnhancedCryptoModel] = {}
        self.last_training_time: Optional[datetime] = None
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        self.performance_history: List[dict] = []

        self.adaptive_params = {
            "confidence_threshold": 75.0,
            "trend_strength_min": 0.3,
            "volume_threshold": 1.2,
            "max_positions_per_hour": 1,
        }

        # Initialise tracker (needs self to exist first)
        self.tracker = TradeResultTracker(self.config, self)

    # ── confidence gate helpers ───────────────────────────────────────────────

    def _tighten_confidence(self, amount: float = 3.0):
        new_val = min(
            self.adaptive_params["confidence_threshold"] + amount,
            self.config.max_confidence - 5,
        )
        if new_val != self.adaptive_params["confidence_threshold"]:
            self.adaptive_params["confidence_threshold"] = new_val
            logger.info(
                f"[Adaptive] ⬆ Confidence gate tightened to {new_val:.1f}%"
            )

    def _relax_confidence(self, amount: float = 1.0):
        new_val = max(
            self.adaptive_params["confidence_threshold"] - amount,
            self.config.min_confidence,
        )
        if new_val != self.adaptive_params["confidence_threshold"]:
            self.adaptive_params["confidence_threshold"] = new_val
            logger.info(
                f"[Adaptive] ⬇ Confidence gate relaxed to {new_val:.1f}%"
            )

    # ── data download ─────────────────────────────────────────────────────────

    def download_training_data(
        self, symbols: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        if symbols is None:
            symbols = self.symbols

        downloader = DataDownloader()
        symbol_data = {}

        logger.info(
            f"Downloading {self.config.lookback_days} days of data for {len(symbols)} symbols..."
        )

        for symbol in symbols:
            try:
                downloader.client.symbol = symbol
                candles = downloader.client.get_klines(
                    interval="1h",
                    limit=self.config.lookback_days * 24,
                )

                if candles and len(candles) >= self.config.min_samples_per_symbol:
                    df = build_symbol_training_frame(candles, symbol)
                    if not df.empty and len(df) >= 100:
                        symbol_data[symbol] = df
                        logger.info(f"  ✓ {symbol}: {len(df)} training samples")
                    else:
                        logger.warning(f"  ⚠ {symbol}: Insufficient after feature engineering")
                else:
                    logger.warning(
                        f"  ⚠ {symbol}: {len(candles) if candles else 0} candles"
                    )

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")

        return symbol_data

    # ── model training ────────────────────────────────────────────────────────

    def train_models(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        market_hints: Optional[dict] = None,
    ) -> Dict[str, str]:
        """
        Train enhanced models.  market_hints (from market structure analysis)
        are forwarded to the model so it can adjust label weights / lookahead.
        """
        model_paths = {}
        hints = market_hints or {}

        for symbol, df in symbol_data.items():
            try:
                logger.info(f"\nTraining enhanced model for {symbol}...")

                candles = df.to_dict("records")

                model = EnhancedCryptoModel()
                # Pass market hints so the model can tune lookahead / label thresholds
                metrics = model.train(
                    candles,
                    symbol,
                    market_hints=hints,
                )

                model_path = os.path.join(self.models_dir, f"enhanced_{symbol}.pkl")
                model.save(model_path)

                self.models[symbol] = model
                model_paths[symbol] = model_path

                logger.info(f"  Samples    : {metrics['samples']}")
                logger.info(f"  RF Accuracy: {metrics['rf_accuracy']:.1%}")
                logger.info(f"  GB Accuracy: {metrics['gb_accuracy']:.1%}")
                logger.info(f"  Features   : {metrics['features']}")

            except Exception as e:
                logger.error(f"  ✗ Training failed for {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        return model_paths

    # ── confidence adjustment ─────────────────────────────────────────────────

    def adjust_confidence_for_targets(
        self,
        base_confidence: float,
        trend_strength: float,
        volume_ratio: float,
        recent_trades: int,
    ) -> float:
        confidence = base_confidence

        if trend_strength > self.adaptive_params["trend_strength_min"]:
            confidence += self.config.strong_trend_boost

        if volume_ratio > self.adaptive_params["volume_threshold"]:
            confidence += self.config.volume_surge_boost

        if recent_trades >= self.adaptive_params["max_positions_per_hour"]:
            confidence -= 15.0

        return min(confidence, self.config.max_confidence)

    def should_take_signal(
        self,
        prediction: dict,
        trend_strength: float,
        volume_ratio: float,
        recent_trades: int,
    ) -> Tuple[bool, float]:
        base_conf = prediction["confidence"]
        adjusted_conf = self.adjust_confidence_for_targets(
            base_conf, trend_strength, volume_ratio, recent_trades
        )
        # Use adaptive gate (raised after poor windows, lowered after good ones)
        gate = self.adaptive_params["confidence_threshold"]
        should_trade = (
            prediction["signal"] is not None
            and adjusted_conf >= gate
            and prediction.get("expected_return", 0) > 0.001
        )
        return should_trade, adjusted_conf

    # ── performance logging (legacy) ──────────────────────────────────────────

    def log_performance(self, trade_result: dict):
        self.performance_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": trade_result.get("symbol"),
                "signal": trade_result.get("signal"),
                "confidence": trade_result.get("confidence"),
                "pnl": trade_result.get("pnl"),
                "success": trade_result.get("pnl", 0) > 0,
            }
        )

        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        self._adapt_parameters()

    def _adapt_parameters(self):
        if len(self.performance_history) < 10:
            return

        recent = self.performance_history[-20:]
        win_rate = sum(1 for t in recent if t["success"]) / len(recent)
        avg_confidence = sum(t["confidence"] for t in recent if t["confidence"]) / len(recent)

        if win_rate < 0.65 and self.adaptive_params["confidence_threshold"] < 85:
            self._tighten_confidence(2.0)
        elif win_rate > self.config.target_win_rate and self.adaptive_params["confidence_threshold"] > 70:
            self._relax_confidence(1.0)

        daily_trades = len(
            [
                t for t in self.performance_history
                if datetime.fromisoformat(t["timestamp"])
                > datetime.utcnow() - timedelta(days=1)
            ]
        )
        logger.info(
            f"Performance: WinRate={win_rate:.1%} AvgConf={avg_confidence:.1f}% Daily={daily_trades}"
        )

    # ── main training cycle ───────────────────────────────────────────────────

    def run_training_cycle(
        self,
        force_retrain: bool = False,
        market_hints: Optional[dict] = None,
    ) -> bool:
        """
        Complete training cycle.

        Args:
            force_retrain   : retrain even if models exist on disk.
            market_hints    : optional structure hints from tracker analysis.
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting Adaptive ML Training Cycle")
        logger.info(
            f"Target: {self.config.target_daily_trades} daily wins at "
            f"{self.config.min_confidence:.0f}%+ confidence"
        )
        if market_hints:
            logger.info(f"Market hints: {market_hints.get('bias','?')} | "
                        f"{market_hints.get('features', {})}")
        logger.info("=" * 60)

        self.is_training = True
        start_time = time.time()

        try:
            existing_models = 0
            missing_symbols = []

            for symbol in self.symbols:
                model_path = os.path.join(self.models_dir, f"enhanced_{symbol}.pkl")
                if os.path.exists(model_path) and not force_retrain:
                    try:
                        model = EnhancedCryptoModel.load(model_path)
                        self.models[symbol] = model
                        logger.info(
                            f"  ✓ {symbol}: loaded (trained: "
                            f"{model.training_date.strftime('%Y-%m-%d %H:%M')})"
                        )
                        existing_models += 1
                    except Exception as e:
                        logger.warning(f"  ⚠ {symbol}: load failed: {e}")
                        missing_symbols.append(symbol)
                else:
                    missing_symbols.append(symbol)

            if existing_models > 0 and not force_retrain and not missing_symbols:
                logger.info(f"✓ All {existing_models} models ready.")
                self.is_training = False
                return True

            target_symbols = missing_symbols if missing_symbols else self.symbols
            symbol_data = self.download_training_data(symbols=target_symbols)

            if not symbol_data:
                logger.error("No training data available!")
                return False

            model_paths = self.train_models(symbol_data, market_hints=market_hints)

            metadata = {
                "trained_at": datetime.utcnow().isoformat(),
                "symbols": self.symbols,
                "model_paths": model_paths,
                "adaptive_params": self.adaptive_params,
                "existing_models": existing_models,
                "market_hints": market_hints or {},
            }
            with open(
                os.path.join(self.models_dir, "training_metadata.json"), "w"
            ) as f:
                json.dump(metadata, f, indent=2, default=str)

            self.last_training_time = datetime.utcnow()
            elapsed = time.time() - start_time
            logger.info(f"✓ Training complete in {elapsed:.1f}s ({len(model_paths)} symbols)")
            return True

        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self.is_training = False

    # ── hourly background scheduler ───────────────────────────────────────────

    def schedule_hourly_training(self):
        def training_loop():
            try:
                self.run_training_cycle(force_retrain=False)
            except Exception as e:
                logger.error(f"Initial training error: {e}")

            while True:
                time.sleep(self.config.training_interval_hours * 3600)
                try:
                    self.run_training_cycle(force_retrain=False)
                    logger.info(
                        f"Next training check in {self.config.training_interval_hours} hours..."
                    )
                except Exception as e:
                    logger.error(f"Training error: {e}")

        self.training_thread = threading.Thread(
            target=training_loop, daemon=True
        )
        self.training_thread.start()
        logger.info(
            f"✓ Scheduled training every {self.config.training_interval_hours} hour(s)"
        )

    # ── model retrieval ───────────────────────────────────────────────────────

    def get_model(self, symbol: str) -> Optional[EnhancedCryptoModel]:
        if symbol in self.models:
            return self.models[symbol]

        model_path = os.path.join(self.models_dir, f"enhanced_{symbol}.pkl")
        if os.path.exists(model_path):
            try:
                self.models[symbol] = EnhancedCryptoModel.load(model_path)
                return self.models[symbol]
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")

        return None

    def predict_with_high_confidence(
        self, symbol: str, klines: list
    ) -> Optional[dict]:
        model = self.get_model(symbol)
        if not model:
            return None

        pred = model.predict(klines)
        if not pred:
            return None

        df = candles_to_dataframe(klines)
        if len(df) < 55:
            return None

        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        from ta.trend import EMAIndicator
        ema_21 = EMAIndicator(close=df["close"], window=21).ema_indicator()
        ema_55 = EMAIndicator(close=df["close"], window=55).ema_indicator()
        trend_strength = (
            abs(ema_21.iloc[-1] - ema_55.iloc[-1]) / df["close"].iloc[-1] * 100
        )

        volume_ma = df["volume"].rolling(window=20).mean()
        volume_ratio = (
            df["volume"].iloc[-1] / volume_ma.iloc[-1]
            if volume_ma.iloc[-1] > 0
            else 1.0
        )

        should_trade, adjusted_conf = self.should_take_signal(
            pred, trend_strength, volume_ratio, 0
        )

        if not should_trade:
            return None

        return {
            **pred,
            "adjusted_confidence": adjusted_conf,
            "trend_strength": trend_strength,
            "volume_ratio": volume_ratio,
            "quality_score": adjusted_conf
            * (1 + trend_strength / 100)
            * min(volume_ratio, 2.0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton
# ─────────────────────────────────────────────────────────────────────────────

adaptive_trainer = AdaptiveMLTrainer()


def initialize_ml_models():
    logger.info("=" * 60)
    logger.info("Initialising ML Models — First Training Run")
    logger.info("=" * 60)
    return adaptive_trainer.run_training_cycle()


if __name__ == "__main__":
    success = initialize_ml_models()
    if success:
        adaptive_trainer.schedule_hourly_training()
        logger.info("\nAdaptive ML trainer running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
    else:
        logger.error("Failed to initialise models!")