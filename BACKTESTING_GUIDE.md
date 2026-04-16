# CORE STRATEGY - High Win-Rate Trading Engine
## Complete Documentation

**Version:** 2.0  
**Target:** 65-80% Win Rate  
**Pairs:** BTCUSDT, ETHUSDT  
**Timeframe:** 5m Entry, 15m/1h Confirmation

---

## Architecture Overview

```
+-----------------------------------------------------------------+
¦                    DIRECTIONAL BIAS ENGINE                       ¦
¦              (Rule-based: EMA 50 > EMA 200 = Bullish)           ¦
+-----------------------------------------------------------------+
                          ¦
                          ?
+-----------------------------------------------------------------+
¦                    6 CONFIRMATION BLOCKS                         ¦
¦  +----------+ +----------+ +----------+ +----------+          ¦
¦  ¦  TREND   ¦ ¦MOMENTUM  ¦ ¦  VOLUME  ¦ ¦ VOLATILE ¦          ¦
¦  ¦ (25 pts) ¦ ¦ (20 pts) ¦ ¦ (15 pts) ¦ ¦ (15 pts) ¦          ¦
¦  +----------+ +----------+ +----------+ +----------+          ¦
¦  +----------+ +----------+                                    ¦
¦  ¦ STRUCTURE¦ ¦DERIVATIV ¦                                    ¦
¦  ¦ (15 pts) ¦ ¦ (10 pts) ¦                                    ¦
¦  +----------+ +----------+                                    ¦
+-----------------------------------------------------------------+
                          ¦
                          ?
+-----------------------------------------------------------------+
¦                    SCORING SYSTEM                                ¦
¦         Total Score = 65/100 ? EXECUTE TRADE                   ¦
¦         Total Score < 65 ? NO TRADE                             ¦
+-----------------------------------------------------------------+
```

---

## The 6 Confirmation Blocks

### Block 1: TREND (25 points max)
**Purpose:** Ensure market is trending in our direction

**Settings:**
- EMA 50 (fast)
- EMA 200 (slow)

**Logic:**
```
LONG:  EMA 50 > EMA 200 ? +25 pts
SHORT: EMA 50 < EMA 200 ? +25 pts

Bonus: +5 pts if EMA separation > 1%
```

**Why this works:** EMA 50/200 crossover is a proven trend indicator. Trading with the trend dramatically improves win rate.

---

### Block 2: MOMENTUM (20 points max)
**Purpose:** Confirm entry at pullback with RSI + MACD

**Settings:**
- RSI (14)
- MACD (12, 26, 9)

**LONG Logic:**
```
RSI between 40-55 (pullback zone) ? +15 pts
MACD histogram positive OR crossing up ? +5 pts
```

**SHORT Logic:**
```
RSI between 45-60 (pullback zone) ? +15 pts
MACD histogram negative OR crossing down ? +5 pts
```

**Why this works:** Buying pullbacks in uptrends = high probability entries.

---

### Block 3: VOLUME (15 points max)
**Purpose:** Confirm move has fuel

**Settings:**
- Volume > 1.5x 20-period average = Spike

**Logic:**
```
Volume spike = 1.5x ? PASS (+15 pts)
Volume ratio 1.2-1.5x ? +10 pts
Volume ratio 1.0-1.2x ? +5 pts
Volume ratio < 1.0x ? +0 pts
```

**Why this works:** Volume confirms the move is real, not just noise.

---

### Block 4: VOLATILITY (15 points max)
**Purpose:** Ensure market is moving but not chaos

**Settings:**
- ATR percentage: 0.3% - 3.0%
- Bollinger Band width > 1.5%

**Logic:**
```
ATR 0.5-2.0% + BB width > 1.5% ? +15 pts
Outside range ? +10 pts or less
```

**Why this works:** Too low volatility = no moves. Too high = unpredictable.

---

### Block 5: STRUCTURE (15 points max)
**Purpose:** Confirm market structure supports our trade

**Settings:**
- Swing high/low detection (5-bar lookback)

**LONG Logic:**
```
Higher low formed ? +15 pts
Otherwise ? +8 pts
```

**SHORT Logic:**
```
Lower high formed ? +15 pts
Otherwise ? +8 pts
```

**Why this works:** Trading at support/resistance with confirmed structure.

---

### Block 6: DERIVATIVES (10 points max)
**Purpose:** Check funding rate alignment (simulated)

**Settings:**
- Funding Rate
- Open Interest

**Logic:**
```
LONG:  Funding not too negative (bears paying) ? +10 pts
SHORT: Funding not too positive (bulls paying) ? +10 pts
```

---

## Entry Rules (Complete Checklist)

### Pre-Check:
- [ ] In London (7-16 UTC) OR US (13:30-22 UTC) session
- [ ] ADX > 20 (trend strength)
- [ ] ADX < 15 ? SKIP (dead market)

### LONG Entry Checklist:
- [ ] EMA 50 > EMA 200 (trend)
- [ ] Price at EMA 50 or VWAP pullback
- [ ] RSI between 40-55
- [ ] MACD bullish or crossing up
- [ ] Volume spike > 1.5x
- [ ] ATR between 0.5-3.0%
- [ ] Higher low formed
- [ ] Total score = 65/100

### SHORT Entry Checklist:
- [ ] EMA 50 < EMA 200 (trend)
- [ ] Price at EMA 50 or VWAP pullback
- [ ] RSI between 45-60
- [ ] MACD bearish or crossing down
- [ ] Volume spike > 1.5x
- [ ] ATR between 0.5-3.0%
- [ ] Lower high formed
- [ ] Total score = 65/100

---

## Exit Rules

### Take Profit:
- **Target:** 0.8% (scalping style)
- **Logic:** Small, consistent wins

### Stop Loss:
- **Target:** 0.5% (tight)
- **Logic:** Cut losses fast

### Time Exit:
- **Max hold:** 2 hours
- **Logic:** Scalping, not swing trading

### Reversal Exit:
- **Trigger:** Opposite signal with 80%+ confidence
- **Logic:** Exit when proven wrong

---

## Risk Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max Leverage | 5x | Conservative |
| Risk per Trade | 1.5% | Fixed |
| Max Daily Loss | 5.0% | Auto-stop |
| Take Profit | 0.8% | Scalping |
| Stop Loss | 0.5% | Tight |
| Max Hold Time | 2 hours | Scalping |

---

## Session Filtering

| Session | Time (UTC) | Trade? |
|---------|------------|--------|
| London | 7:00 - 16:00 | ? Yes |
| US | 13:30 - 22:00 | ? Yes |
| Asian | 0:00 - 7:00 | ? Skip |
| Overlap | 13:30 - 16:00 | ? Best |

---

## Multi-Timeframe Confirmation

| Timeframe | Purpose | Required? |
|-----------|---------|-----------|
| 5m | Entry signal | Yes |
| 15m | Trend confirmation | Recommended |
| 1h | Direction bias | Recommended |

**Logic:** If 15m and 1h align with 5m signal ? Higher confidence

---

## Scoring System (Fallback)

If ALL conditions too strict, use weighted scoring:

| Score | Action |
|-------|--------|
| 85-100 | Strong buy/sell |
| 70-84 | Normal entry |
| 65-69 | Weak entry (smaller size) |
| < 65 | No trade |

---

## Expected Performance

| Metric | Target | Range |
|--------|--------|-------|
| Win Rate | 70% | 65-80% |
| Profit Factor | 1.8 | 1.5-2.5 |
| Max Drawdown | 10% | 5-15% |
| Trades/Day | 2-4 | Per pair |
| Risk/Reward | 1.6 | 1.5-2.0 |

---

## Files

- `core_strategy.py` - Main strategy implementation
- `optimized_strategy.py` - Previous version (fallback)
- `talon_sniper_strategy.py` - Original version (fallback)

---

## Usage

```bash
cd D:\Futures
start_multi.bat
```

Monitor logs:
```
[BTCUSDT] LONG @ 83200.00 | Conf: 78% | Score: 72/100 | Regime: confirmed_trend | Session: US
```

---

## Optimization Tips

### If Win Rate < 60%:
1. Increase min_score to 70
2. Require ADX > 25
3. Tighten RSI range

### If Too Few Trades:
1. Lower min_score to 60
2. Allow 15m-only confirmation
3. Expand session to Asian hours

### If Drawdown > 15%:
1. Reduce position size by 20%
2. Increase stop loss to 0.6%
3. Lower max leverage to 3x

### If Profitable but Want More:
1. Add more pairs (ETH, SOL)
2. Expand session filter
3. Slightly lower min_score

---

## Disclaimer

Trading cryptocurrency futures involves substantial risk. This software is for educational purposes only. Past performance does not guarantee future results.
