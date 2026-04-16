import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

BASE_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "ema_9_gap",
    "ema_21_gap",
    "ema_55_gap",
    "ema_9_21_spread",
    "ema_21_55_spread",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_position",
    "atr_pct",
    "volume_ratio",
    "volume_ratio_change",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "range_pct",
    "hour_sin",
    "hour_cos",
]


def candles_to_dataframe(candles: Iterable) -> pd.DataFrame:
    """Convert candles to DataFrame - handles both dict and list formats"""
    candles = list(candles)
    if not candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    first = candles[0]
    
    # Handle dict format (from JSON files)
    if isinstance(first, dict):
        rows = [
            {
                "timestamp": int(c["timestamp"]),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
            }
            for c in candles
        ]
        return pd.DataFrame(rows)

    # Handle Binance klines format (12-element list) - only take first 6 columns
    if isinstance(first, (list, tuple)) and len(first) >= 6:
        rows = [
            {
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
            for k in candles
        ]
        return pd.DataFrame(rows)
    
    # Unknown format
    raise ValueError(f"Unknown candle format: {type(first)} with {len(first) if hasattr(first, '__len__') else 'unknown'} elements")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    feature_df["return_1"] = feature_df["close"].pct_change(1)
    feature_df["return_3"] = feature_df["close"].pct_change(3)
    feature_df["return_6"] = feature_df["close"].pct_change(6)

    ema_9 = EMAIndicator(close=feature_df["close"], window=9).ema_indicator()
    ema_21 = EMAIndicator(close=feature_df["close"], window=21).ema_indicator()
    ema_55 = EMAIndicator(close=feature_df["close"], window=55).ema_indicator()
    feature_df["ema_9_gap"] = (feature_df["close"] - ema_9) / feature_df["close"]
    feature_df["ema_21_gap"] = (feature_df["close"] - ema_21) / feature_df["close"]
    feature_df["ema_55_gap"] = (feature_df["close"] - ema_55) / feature_df["close"]
    feature_df["ema_9_21_spread"] = (ema_9 - ema_21) / feature_df["close"]
    feature_df["ema_21_55_spread"] = (ema_21 - ema_55) / feature_df["close"]

    rsi = RSIIndicator(close=feature_df["close"], window=14)
    feature_df["rsi_14"] = rsi.rsi()

    macd = MACD(close=feature_df["close"], window_slow=26, window_fast=12, window_sign=9)
    feature_df["macd"] = macd.macd()
    feature_df["macd_signal"] = macd.macd_signal()
    feature_df["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(close=feature_df["close"], window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_span = bb_high - bb_low
    feature_df["bb_width"] = bb_span / feature_df["close"]
    feature_df["bb_position"] = (feature_df["close"] - bb_low) / (bb_span + 1e-12)

    atr = AverageTrueRange(
        high=feature_df["high"],
        low=feature_df["low"],
        close=feature_df["close"],
        window=14,
    )
    feature_df["atr_pct"] = atr.average_true_range() / feature_df["close"]

    volume_sma20 = feature_df["volume"].rolling(window=20).mean()
    feature_df["volume_ratio"] = feature_df["volume"] / (volume_sma20 + 1e-12)
    feature_df["volume_ratio_change"] = feature_df["volume_ratio"].pct_change()

    feature_df["body_pct"] = (feature_df["close"] - feature_df["open"]) / feature_df["open"]
    upper_wick = feature_df["high"] - np.maximum(feature_df["open"], feature_df["close"])
    lower_wick = np.minimum(feature_df["open"], feature_df["close"]) - feature_df["low"]
    feature_df["upper_wick_pct"] = upper_wick / feature_df["open"]
    feature_df["lower_wick_pct"] = lower_wick / feature_df["open"]
    feature_df["range_pct"] = (feature_df["high"] - feature_df["low"]) / feature_df["open"]

    ts = pd.to_datetime(feature_df["timestamp"], unit="ms", utc=True)
    hour = ts.dt.hour.astype(float)
    feature_df["hour_sin"] = np.sin((2 * np.pi * hour) / 24.0)
    feature_df["hour_cos"] = np.cos((2 * np.pi * hour) / 24.0)

    return feature_df


def _forward_close_max_min(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    future_close = np.full(n, np.nan, dtype=float)
    future_high_max = np.full(n, np.nan, dtype=float)
    future_low_min = np.full(n, np.nan, dtype=float)

    if n <= horizon:
        return future_close, future_high_max, future_low_min

    future_close[: n - horizon] = close[horizon:]

    high_windows = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)
    low_windows = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)
    valid_len = high_windows.shape[0]
    future_high_max[:valid_len] = high_windows.max(axis=1)
    future_low_min[:valid_len] = low_windows.min(axis=1)
    return future_close, future_high_max, future_low_min


def add_profit_targets(feature_df: pd.DataFrame, horizon: int = 6, fee_bps: float = 4.0) -> pd.DataFrame:
    close = feature_df["close"].to_numpy(dtype=float)
    high = feature_df["high"].to_numpy(dtype=float)
    low = feature_df["low"].to_numpy(dtype=float)
    future_close, future_high_max, future_low_min = _forward_close_max_min(close, high, low, horizon)

    fee = fee_bps / 10000.0
    close_ret = (future_close - close) / close
    max_up = (future_high_max - close) / close
    max_down = (close - future_low_min) / close

    target_long = (0.55 * close_ret) + (0.45 * max_up) - fee
    target_short = (0.55 * (-close_ret)) + (0.45 * max_down) - fee

    out = feature_df.copy()
    out["target_long"] = target_long
    out["target_short"] = target_short
    return out


def build_symbol_training_frame(
    candles: Iterable[dict], symbol: str, horizon: int = 6, fee_bps: float = 4.0
) -> pd.DataFrame:
    df = candles_to_dataframe(candles)
    if df.empty:
        return df

    feat = build_feature_frame(df)
    feat = add_profit_targets(feat, horizon=horizon, fee_bps=fee_bps)
    feat["symbol"] = symbol
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=BASE_FEATURE_COLUMNS + ["target_long", "target_short"])
    return feat


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([X, np.ones(len(X))])
    reg = np.eye(x_aug.shape[1], dtype=float) * alpha
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    return np.linalg.solve(lhs, rhs)


def _predict_ridge(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([X, np.ones(len(X))])
    return x_aug @ weights


def _evaluate_selection(
    pred_long: np.ndarray,
    pred_short: np.ndarray,
    y_long: np.ndarray,
    y_short: np.ndarray,
    edge_threshold: float,
    min_gap: float,
) -> Dict[str, float]:
    edge = np.maximum(pred_long, pred_short)
    gap = np.abs(pred_long - pred_short)
    side_long = pred_long >= pred_short
    realized = np.where(side_long, y_long, y_short)
    mask = (edge >= edge_threshold) & (gap >= min_gap)

    if mask.sum() == 0:
        return {
            "trade_count": 0,
            "trade_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "profit_factor": 0.0,
        }

    returns = realized[mask]
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    gross_profit = float(positive.sum()) if len(positive) else 0.0
    gross_loss = float(abs(negative.sum())) if len(negative) else 0.0

    cumulative = float(np.prod(1.0 + returns) - 1.0)
    return {
        "trade_count": int(mask.sum()),
        "trade_rate": float(mask.mean()),
        "avg_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float((returns > 0).mean()),
        "cumulative_return": cumulative,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class TrainingResult:
    artifact: Dict
    train_metrics: Dict
    validation_metrics: Dict
    samples_used: int


class PremiumSignalModelTrainer:
    def __init__(self, alpha: float = 8.0, horizon: int = 6, fee_bps: float = 4.0):
        self.alpha = float(alpha)
        self.horizon = int(horizon)
        self.fee_bps = float(fee_bps)

    def fit(self, symbol_frames: Dict[str, pd.DataFrame]) -> TrainingResult:
        non_empty = {k: v for k, v in symbol_frames.items() if v is not None and not v.empty}
        if not non_empty:
            raise ValueError("No training data available after feature engineering.")

        all_symbols = sorted(non_empty.keys())
        combined = pd.concat(non_empty.values(), axis=0, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna(subset=BASE_FEATURE_COLUMNS + ["target_long", "target_short"])

        symbol_cols = [f"sym_{s}" for s in all_symbols]
        symbol_dummies = pd.get_dummies(combined["symbol"], prefix="sym")
        for c in symbol_cols:
            if c not in symbol_dummies.columns:
                symbol_dummies[c] = 0
        symbol_dummies = symbol_dummies[symbol_cols]

        feature_columns = BASE_FEATURE_COLUMNS + symbol_cols
        x_df = pd.concat([combined[BASE_FEATURE_COLUMNS], symbol_dummies], axis=1)
        x_df = x_df[feature_columns].astype(float)
        y_long = combined["target_long"].to_numpy(dtype=float)
        y_short = combined["target_short"].to_numpy(dtype=float)

        x = x_df.to_numpy(dtype=float)
        split_idx = max(int(len(x) * 0.8), 1)
        if split_idx >= len(x):
            split_idx = len(x) - 1
        if split_idx <= 0:
            raise ValueError("Not enough rows to split train/validation.")

        x_train, x_val = x[:split_idx], x[split_idx:]
        yl_train, yl_val = y_long[:split_idx], y_long[split_idx:]
        ys_train, ys_val = y_short[:split_idx], y_short[split_idx:]

        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std < 1e-8] = 1.0

        x_train_n = (x_train - x_mean) / x_std
        x_val_n = (x_val - x_mean) / x_std

        long_w = _fit_ridge(x_train_n, yl_train, self.alpha)
        short_w = _fit_ridge(x_train_n, ys_train, self.alpha)

        pred_l_train = _predict_ridge(x_train_n, long_w)
        pred_s_train = _predict_ridge(x_train_n, short_w)
        pred_l_val = _predict_ridge(x_val_n, long_w)
        pred_s_val = _predict_ridge(x_val_n, short_w)

        edge_train = np.maximum(pred_l_train, pred_s_train)
        gap_train = np.abs(pred_l_train - pred_s_train)
        best_score = -1e18
        best_threshold = float(np.quantile(edge_train, 0.7))
        best_gap = float(np.quantile(gap_train, 0.55))

        for edge_q in np.linspace(0.55, 0.90, 8):
            edge_th = float(np.quantile(edge_train, edge_q))
            for gap_q in np.linspace(0.40, 0.85, 10):
                gap_th = float(np.quantile(gap_train, gap_q))
                eval_train = _evaluate_selection(
                    pred_long=pred_l_train,
                    pred_short=pred_s_train,
                    y_long=yl_train,
                    y_short=ys_train,
                    edge_threshold=edge_th,
                    min_gap=gap_th,
                )
                trades = eval_train["trade_count"]
                if trades < max(40, int(0.02 * len(yl_train))):
                    continue
                trade_rate = trades / len(yl_train)
                if trade_rate < 0.05 or trade_rate > 0.6:
                    continue
                score = eval_train["avg_return"] * np.sqrt(trades)
                if score > best_score:
                    best_score = score
                    best_threshold = edge_th
                    best_gap = gap_th

        train_selection = _evaluate_selection(
            pred_long=pred_l_train,
            pred_short=pred_s_train,
            y_long=yl_train,
            y_short=ys_train,
            edge_threshold=best_threshold,
            min_gap=best_gap,
        )
        val_selection = _evaluate_selection(
            pred_long=pred_l_val,
            pred_short=pred_s_val,
            y_long=yl_val,
            y_short=ys_val,
            edge_threshold=best_threshold,
            min_gap=best_gap,
        )

        # If validation has no trades, relax thresholds stepwise.
        if val_selection["trade_count"] == 0:
            fallback_quantiles = [
                (0.70, 0.55),
                (0.60, 0.50),
                (0.50, 0.40),
                (0.40, 0.30),
            ]
            for edge_q, gap_q in fallback_quantiles:
                edge_th = float(np.quantile(edge_train, edge_q))
                gap_th = float(np.quantile(gap_train, gap_q))
                candidate_val = _evaluate_selection(
                    pred_long=pred_l_val,
                    pred_short=pred_s_val,
                    y_long=yl_val,
                    y_short=ys_val,
                    edge_threshold=edge_th,
                    min_gap=gap_th,
                )
                if candidate_val["trade_count"] > 0:
                    best_threshold = edge_th
                    best_gap = gap_th
                    val_selection = candidate_val
                    train_selection = _evaluate_selection(
                        pred_long=pred_l_train,
                        pred_short=pred_s_train,
                        y_long=yl_train,
                        y_short=ys_train,
                        edge_threshold=best_threshold,
                        min_gap=best_gap,
                    )
                    break

        train_metrics = {
            "long_rmse": _rmse(yl_train, pred_l_train),
            "short_rmse": _rmse(ys_train, pred_s_train),
            "selection": train_selection,
        }
        val_metrics = {
            "long_rmse": _rmse(yl_val, pred_l_val),
            "short_rmse": _rmse(ys_val, pred_s_val),
            "selection": val_selection,
        }

        artifact = {
            "version": "premium-signal-v1",
            "trained_at_utc": datetime.utcnow().isoformat(),
            "alpha": self.alpha,
            "horizon": self.horizon,
            "fee_bps": self.fee_bps,
            "base_feature_columns": BASE_FEATURE_COLUMNS,
            "feature_columns": feature_columns,
            "symbols": all_symbols,
            "scaler": {
                "mean": x_mean.tolist(),
                "std": x_std.tolist(),
            },
            "weights": {
                "long": long_w.tolist(),
                "short": short_w.tolist(),
            },
            "selection": {
                "edge_threshold": best_threshold,
                "min_gap": best_gap,
                "edge_p90": float(np.quantile(edge_train, 0.90)),
            },
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "samples": int(len(combined)),
            },
        }

        return TrainingResult(
            artifact=artifact,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            samples_used=int(len(combined)),
        )

    @staticmethod
    def save_artifact(artifact: Dict, model_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)


class PremiumSignalModel:
    def __init__(self, artifact: Dict):
        self.artifact = artifact
        self.feature_columns = artifact["feature_columns"]
        self.base_feature_columns = artifact["base_feature_columns"]
        self.symbols = artifact["symbols"]
        self.mean = np.array(artifact["scaler"]["mean"], dtype=float)
        self.std = np.array(artifact["scaler"]["std"], dtype=float)
        self.long_weights = np.array(artifact["weights"]["long"], dtype=float)
        self.short_weights = np.array(artifact["weights"]["short"], dtype=float)
        self.edge_threshold = float(artifact["selection"]["edge_threshold"])
        self.min_gap = float(artifact["selection"]["min_gap"])
        self.edge_p90 = float(artifact["selection"]["edge_p90"])

    @classmethod
    def load(cls, model_path: str) -> "PremiumSignalModel":
        with open(model_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        return cls(artifact=artifact)

    def _vector_from_feature_map(self, feature_map: Dict[str, float], symbol: str) -> np.ndarray:
        values: List[float] = []
        for col in self.feature_columns:
            if col.startswith("sym_"):
                values.append(1.0 if col == f"sym_{symbol}" else 0.0)
            else:
                values.append(float(feature_map.get(col, 0.0)))
        return np.array(values, dtype=float)

    def predict(self, feature_map: Dict[str, float], symbol: str) -> Dict[str, float]:
        x = self._vector_from_feature_map(feature_map, symbol=symbol)
        x_n = (x - self.mean) / self.std
        x_aug = np.append(x_n, 1.0)

        pred_long = float(np.dot(x_aug, self.long_weights))
        pred_short = float(np.dot(x_aug, self.short_weights))
        edge = max(pred_long, pred_short)
        gap = abs(pred_long - pred_short)
        direction = "LONG" if pred_long >= pred_short else "SHORT"

        if edge < self.edge_threshold or gap < self.min_gap:
            return {
                "signal": None,
                "pred_long": pred_long,
                "pred_short": pred_short,
                "edge": edge,
                "gap": gap,
                "confidence": 0.0,
                "expected_return": 0.0,
            }

        edge_span = max(self.edge_p90 - self.edge_threshold, 1e-6)
        strength = (edge - self.edge_threshold) / edge_span
        confidence = float(np.clip(60.0 + 30.0 * np.tanh(1.2 * strength), 50.0, 97.0))
        expected_return = pred_long if direction == "LONG" else pred_short

        return {
            "signal": direction,
            "pred_long": pred_long,
            "pred_short": pred_short,
            "edge": edge,
            "gap": gap,
            "confidence": confidence,
            "expected_return": float(expected_return),
        }


def latest_feature_row_from_klines(klines: Iterable[list]) -> Optional[Dict[str, float]]:
    df = candles_to_dataframe(klines)
    if len(df) < 80:
        return None
    feat = build_feature_frame(df)
    feat = feat.dropna(subset=BASE_FEATURE_COLUMNS)
    if feat.empty:
        return None
    latest = feat.iloc[-1]
    return {c: float(latest[c]) for c in BASE_FEATURE_COLUMNS}
