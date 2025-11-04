
# aita_system/aita_system.py
from __future__ import annotations
import math, json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ---------- Indicators & helpers

def atr(df: pd.DataFrame, n=14) -> pd.Series:
    return AverageTrueRange(df["High"], df["Low"], df["Close"], n).average_true_range()

def rsi(df: pd.DataFrame, n=14) -> pd.Series:
    return RSIIndicator(df["Close"], n).rsi()

def macd(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    return m.macd(), m.macd_signal(), m.macd_diff()

def vol_zscore(df: pd.DataFrame, n=20) -> pd.Series:
    v = df["Volume"].astype(float)
    m = v.rolling(n).mean()
    s = v.rolling(n).std(ddof=0)
    return (v - m) / s

def typical_price(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3.0

# ---------- Anchored VWAP

def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    tp = typical_price(df)
    vol = df["Volume"].astype(float)
    ctp = (tp.iloc[anchor_idx:] * vol.iloc[anchor_idx:]).cumsum()
    cv = vol.iloc[anchor_idx:].cumsum()
    avwap = (ctp / cv).reindex(df.index)
    return avwap

# ---------- VPVR

def vpvr(df: pd.DataFrame, bins=60) -> Dict[str, np.ndarray]:
    low = df["Low"].min()
    high = df["High"].max()
    hist_prices = np.linspace(low, high, bins + 1)
    tp = typical_price(df).values
    vol = df["Volume"].values.astype(float)
    inds = np.searchsorted(hist_prices, tp, side="right") - 1
    inds = np.clip(inds, 0, bins - 1)
    vbp = np.zeros(bins)
    np.add.at(vbp, inds, vol)
    centers = (hist_prices[:-1] + hist_prices[1:]) / 2
    return {"price_centers": centers, "volume": vbp}

from scipy.signal import argrelextrema

def find_pivots(df: pd.DataFrame, order=5) -> Tuple[List[int], List[int]]:
    highs = argrelextrema(df["High"].values, np.greater, order=order)[0].tolist()
    lows  = argrelextrema(df["Low"].values, np.less, order=order)[0].tolist()
    return highs, lows

def cluster_levels(prices: List[float], tolerance: float=0.003) -> List[Tuple[float,int]]:
    if not prices:
        return []
    prices = sorted(prices)
    clusters = [[prices[0]]]
    for x in prices[1:]:
        if abs(x - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    result = [(float(sum(c)/len(c)), len(c)) for c in clusters]
    result.sort(key=lambda t: (-t[1], t[0]))
    return result

def sr_levels_from_pivots(df: pd.DataFrame, order=5, tolerance=0.003) -> List[Tuple[float,int]]:
    highs, lows = find_pivots(df, order)
    levels = [float(df["High"].iloc[i]) for i in highs] + [float(df["Low"].iloc[i]) for i in lows]
    return cluster_levels(levels, tolerance)

def fib_levels_from_last_swing(df: pd.DataFrame, order=5, uptrend: Optional[bool]=None) -> Dict[str, float]:
    highs, lows = find_pivots(df, order)
    if not highs or not lows:
        return {}
    last_high = max(highs)
    last_low  = max(lows)
    if uptrend is None:
        uptrend = last_low < last_high
    if uptrend:
        start = max([i for i in lows if i < last_high], default=lows[-1])
        end = last_high
        lo = df["Low"].iloc[start]
        hi = df["High"].iloc[end]
        diff = hi - lo
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        levels = {f"{int(r*100)}%": float(hi - r*diff) for r in ratios}
        levels.update({"127.2%": float(hi + 0.272*diff), "161.8%": float(hi + 0.618*diff)})
        return {"direction":"up","start_idx": int(start), "end_idx": int(end), "levels": levels}
    else:
        start = max([i for i in highs if i < last_low], default=highs[-1])
        end = last_low
        hi = df["High"].iloc[start]
        lo = df["Low"].iloc[end]
        diff = hi - lo
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        levels = {f"{int(r*100)}%": float(lo + r*diff) for r in ratios}
        levels.update({"127.2%": float(lo - 0.272*diff), "161.8%": float(lo - 0.618*diff)})
        return {"direction":"down","start_idx": int(start), "end_idx": int(end), "levels": levels}

# ---------- Patterns

@dataclass
class DetectedPattern:
    name: str
    score: float
    meta: Dict

def detect_cup_handle(df: pd.DataFrame, order=5, min_depth_pct=0.08, max_depth_pct=0.40) -> Optional[DetectedPattern]:
    highs, lows = find_pivots(df, order)
    if len(highs) < 2 or len(lows) < 1:
        return None
    rim2 = highs[-1]
    rim1 = max([i for i in highs[:-1] if i < rim2 - 5], default=None)
    if rim1 is None:
        return None
    rim_hi1 = float(df["High"].iloc[rim1])
    rim_hi2 = float(df["High"].iloc[rim2])
    if abs(rim_hi1 - rim_hi2) / ((rim_hi1 + rim_hi2)/2) > 0.015:
        return None
    mid_lows = [i for i in lows if rim1 < i < rim2]
    if not mid_lows:
        return None
    cup_low_idx = int(min(mid_lows, key=lambda i: df["Low"].iloc[i]))
    cup_low = float(df["Low"].iloc[cup_low_idx])
    rim = (rim_hi1 + rim_hi2) / 2.0
    depth = (rim - cup_low) / rim
    if not (min_depth_pct <= depth <= max_depth_pct):
        return None
    window = max(3, (rim2 - rim1)//5)
    smooth = df["Close"].rolling(window).mean()
    curve_err = float(np.mean(np.abs(smooth.iloc[rim1:rim2] - df["Close"].iloc[rim1:rim2])) / rim)
    if curve_err > 0.03:
        return None
    handle_slice = df.iloc[rim2+1 : rim2+1+15]
    if handle_slice.shape[0] < 3:
        return None
    x = np.arange(handle_slice.shape[0])
    y = handle_slice["Close"].values
    slope, intercept = np.polyfit(x, y, 1)
    handle_depth = (handle_slice["High"].max() - handle_slice["Low"].min()) / rim
    if not (-0.8 <= slope <= 0) or handle_depth > min(0.5*depth, 0.12):
        return None
    broke = float(df["Close"].iloc[-1]) > rim
    score = 0.6 + 0.2*(1-curve_err/0.03) + 0.2*min(1, (0.5*depth)/max(min_depth_pct,1e-6))
    return DetectedPattern(
        name="CupAndHandle",
        score=float(min(1.0, max(0.0, score) + (0.1 if broke else 0))),
        meta={"rim_level": rim,"rim_idx1": int(rim1),"rim_idx2": int(rim2),
              "cup_low_idx": cup_low_idx,"cup_depth_pct": float(depth),
              "handle_start": int(rim2+1),"handle_end": int(rim2+1+len(handle_slice)-1)}
    )

def detect_bull_flag(df: pd.DataFrame, lookback=40, min_impulse_atr=1.2) -> Optional[DetectedPattern]:
    if df.shape[0] < lookback + 10:
        return None
    sub = df.iloc[-lookback:]
    atr14 = atr(sub, 14).iloc[-1]
    impulse_range = sub["High"].iloc[-10:].max() - sub["Low"].iloc[-10:].min()
    if impulse_range < min_impulse_atr * atr14:
        return None
    pull = sub.iloc[-8:]
    x = np.arange(pull.shape[0])
    slope, intercept = np.polyfit(x, pull["Close"].values, 1)
    if slope >= 0:
        return None
    volz = vol_zscore(sub, 20).iloc[-8:]
    if not (volz.mean() < 0.0):
        return None
    score = 0.55 + 0.25*min(1, impulse_range/(1.5*atr14)) + 0.2*min(1, (0 - volz.mean())/2)
    flag_start_idx = len(df) - 8
    return DetectedPattern(name="BullFlag", score=float(min(1.0, score)), meta={"flag_start_idx": flag_start_idx, "flag_len": 8})

def detect_head_shoulders(df: pd.DataFrame, order=5, symmetry_tolerance=0.03) -> Optional[DetectedPattern]:
    """Detect head and shoulders pattern (bearish reversal)"""
    highs, lows = find_pivots(df, order)
    if len(highs) < 3:
        return None

    # Look for H&S in recent swing highs
    for i in range(len(highs) - 2):
        left_shoulder_idx = highs[i]
        head_idx = highs[i + 1]
        right_shoulder_idx = highs[i + 2]

        left_shoulder = float(df["High"].iloc[left_shoulder_idx])
        head = float(df["High"].iloc[head_idx])
        right_shoulder = float(df["High"].iloc[right_shoulder_idx])

        # Check H&S conditions: head > both shoulders
        if not (head > left_shoulder and head > right_shoulder):
            continue

        # Check shoulders are roughly equal (within tolerance)
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > symmetry_tolerance:
            continue

        # Find neckline (support between shoulders)
        neckline_slice = df["Low"].iloc[left_shoulder_idx:right_shoulder_idx + 1]
        neckline_price = float(neckline_slice.min())
        neckline_idx = neckline_slice.idxmin()

        # Calculate pattern height (for target projection)
        height = head - neckline_price

        # Calculate confidence score
        symmetry_score = 1.0 - (shoulder_diff / symmetry_tolerance)
        volume_conf = 1.0 if df["Volume"].iloc[head_idx] > df["Volume"].iloc[left_shoulder_idx:right_shoulder_idx + 1].mean() else 0.5

        # Check if price has broken neckline
        current_price = float(df["Close"].iloc[-1])
        breakout = current_price < neckline_price * 0.99

        score = 0.65 + 0.2 * symmetry_score + 0.1 * volume_conf + (0.05 if breakout else 0)

        if score >= 0.70:
            return DetectedPattern(
                name="HeadAndShoulders",
                score=float(min(1.0, score)),
                meta={
                    "left_shoulder_idx": int(left_shoulder_idx),
                    "head_idx": int(head_idx),
                    "right_shoulder_idx": int(right_shoulder_idx),
                    "neckline_price": neckline_price,
                    "neckline_idx": int(df.index.get_loc(neckline_idx)),
                    "pattern_height": height,
                    "left_shoulder": left_shoulder,
                    "head": head,
                    "right_shoulder": right_shoulder
                }
            )
    return None

def detect_inverse_head_shoulders(df: pd.DataFrame, order=5, symmetry_tolerance=0.03) -> Optional[DetectedPattern]:
    """Detect inverse head and shoulders pattern (bullish reversal)"""
    highs, lows = find_pivots(df, order)
    if len(lows) < 3:
        return None

    # Look for inverse H&S in recent swing lows
    for i in range(len(lows) - 2):
        left_shoulder_idx = lows[i]
        head_idx = lows[i + 1]
        right_shoulder_idx = lows[i + 2]

        left_shoulder = float(df["Low"].iloc[left_shoulder_idx])
        head = float(df["Low"].iloc[head_idx])
        right_shoulder = float(df["Low"].iloc[right_shoulder_idx])

        # Check inverse H&S conditions: head < both shoulders (lower low)
        if not (head < left_shoulder and head < right_shoulder):
            continue

        # Check shoulders are roughly equal
        shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
        if shoulder_diff > symmetry_tolerance:
            continue

        # Find neckline (resistance between shoulders)
        neckline_slice = df["High"].iloc[left_shoulder_idx:right_shoulder_idx + 1]
        neckline_price = float(neckline_slice.max())
        neckline_idx = neckline_slice.idxmax()

        # Calculate pattern height
        height = neckline_price - head

        # Calculate confidence
        symmetry_score = 1.0 - (shoulder_diff / symmetry_tolerance)
        volume_conf = 1.0 if df["Volume"].iloc[right_shoulder_idx] > df["Volume"].iloc[left_shoulder_idx:head_idx + 1].mean() else 0.5

        # Check breakout
        current_price = float(df["Close"].iloc[-1])
        breakout = current_price > neckline_price * 1.01

        score = 0.65 + 0.2 * symmetry_score + 0.1 * volume_conf + (0.05 if breakout else 0)

        if score >= 0.70:
            return DetectedPattern(
                name="InverseHeadAndShoulders",
                score=float(min(1.0, score)),
                meta={
                    "left_shoulder_idx": int(left_shoulder_idx),
                    "head_idx": int(head_idx),
                    "right_shoulder_idx": int(right_shoulder_idx),
                    "neckline_price": neckline_price,
                    "neckline_idx": int(df.index.get_loc(neckline_idx)),
                    "pattern_height": height,
                    "left_shoulder": left_shoulder,
                    "head": head,
                    "right_shoulder": right_shoulder
                }
            )
    return None

def detect_double_bottom(df: pd.DataFrame, order=5, price_tolerance=0.02) -> Optional[DetectedPattern]:
    """Detect double bottom pattern (bullish reversal)"""
    highs, lows = find_pivots(df, order)
    if len(lows) < 2:
        return None

    # Look for two roughly equal lows
    for i in range(len(lows) - 1):
        first_low_idx = lows[i]
        second_low_idx = lows[i + 1]

        first_low = float(df["Low"].iloc[first_low_idx])
        second_low = float(df["Low"].iloc[second_low_idx])

        # Check if lows are roughly equal (within tolerance)
        price_diff = abs(first_low - second_low) / first_low
        if price_diff > price_tolerance:
            continue

        # Find peak between the two lows (resistance level)
        peak_slice = df["High"].iloc[first_low_idx:second_low_idx + 1]
        resistance = float(peak_slice.max())
        peak_idx = peak_slice.idxmax()

        # Ensure there's meaningful separation (peak should be higher than lows)
        if resistance < first_low * 1.02:
            continue

        # Calculate confidence
        price_similarity = 1.0 - (price_diff / price_tolerance)

        # Check volume (second low should have lower volume = bullish)
        vol_confirmation = 1.0 if df["Volume"].iloc[second_low_idx] < df["Volume"].iloc[first_low_idx] else 0.6

        # Check breakout above resistance
        current_price = float(df["Close"].iloc[-1])
        breakout = current_price > resistance * 1.01

        score = 0.65 + 0.2 * price_similarity + 0.1 * vol_confirmation + (0.05 if breakout else 0)

        if score >= 0.70:
            return DetectedPattern(
                name="DoubleBottom",
                score=float(min(1.0, score)),
                meta={
                    "first_low_idx": int(first_low_idx),
                    "second_low_idx": int(second_low_idx),
                    "peak_idx": int(df.index.get_loc(peak_idx)),
                    "first_low": first_low,
                    "second_low": second_low,
                    "resistance": resistance,
                    "pattern_height": resistance - min(first_low, second_low)
                }
            )
    return None

def detect_double_top(df: pd.DataFrame, order=5, price_tolerance=0.02) -> Optional[DetectedPattern]:
    """Detect double top pattern (bearish reversal)"""
    highs, lows = find_pivots(df, order)
    if len(highs) < 2:
        return None

    # Look for two roughly equal highs
    for i in range(len(highs) - 1):
        first_high_idx = highs[i]
        second_high_idx = highs[i + 1]

        first_high = float(df["High"].iloc[first_high_idx])
        second_high = float(df["High"].iloc[second_high_idx])

        # Check if highs are roughly equal
        price_diff = abs(first_high - second_high) / first_high
        if price_diff > price_tolerance:
            continue

        # Find trough between the two highs (support level)
        trough_slice = df["Low"].iloc[first_high_idx:second_high_idx + 1]
        support = float(trough_slice.min())
        trough_idx = trough_slice.idxmin()

        # Ensure meaningful separation
        if support > first_high * 0.98:
            continue

        # Calculate confidence
        price_similarity = 1.0 - (price_diff / price_tolerance)
        vol_confirmation = 1.0 if df["Volume"].iloc[second_high_idx] < df["Volume"].iloc[first_high_idx] else 0.6

        # Check breakdown below support
        current_price = float(df["Close"].iloc[-1])
        breakdown = current_price < support * 0.99

        score = 0.65 + 0.2 * price_similarity + 0.1 * vol_confirmation + (0.05 if breakdown else 0)

        if score >= 0.70:
            return DetectedPattern(
                name="DoubleTop",
                score=float(min(1.0, score)),
                meta={
                    "first_high_idx": int(first_high_idx),
                    "second_high_idx": int(second_high_idx),
                    "trough_idx": int(df.index.get_loc(trough_idx)),
                    "first_high": first_high,
                    "second_high": second_high,
                    "support": support,
                    "pattern_height": max(first_high, second_high) - support
                }
            )
    return None

# ---------- Confluence

def hvn_lvn(vp: Dict[str, np.ndarray], order: int=2) -> Tuple[List[float], List[float]]:
    v = vp["volume"]
    p = vp["price_centers"]
    highs = argrelextrema(v, np.greater_equal, order=order)[0]
    lows  = argrelextrema(v, np.less_equal, order=order)[0]
    meanv = v.mean() if v.size else 0
    hvns = [float(p[i]) for i in highs if v[i] >= meanv * 1.1]
    lvns = [float(p[i]) for i in lows  if v[i] <= meanv * 0.9]
    return hvns, lvns

def confluence_score(df: pd.DataFrame,
                     patterns: List[DetectedPattern],
                     sr_levels: List[Tuple[float,int]],
                     vp: Dict[str,np.ndarray],
                     volz: pd.Series,
                     macd_line: pd.Series,
                     rsi_line: pd.Series) -> float:
    s = 0.0
    if patterns:
        s += min(0.5, sum(p.score for p in patterns))
    if volz.iloc[-1] > 1.0:
        s += 0.15
    if macd_line.iloc[-1] > 0:
        s += 0.1
    if 40 <= rsi_line.iloc[-1] <= 70:
        s += 0.05
    price = float(df["Close"].iloc[-1])
    hvns, lvns = hvn_lvn(vp)
    if lvns:
        nearest_lvn = min(lvns, key=lambda x: abs(x - price))
        if abs(nearest_lvn - price)/price <= 0.01:
            s += 0.1
    if sr_levels and sr_levels[0][1] >= 3:
        s += 0.05
    return float(max(0.0, min(1.0, s)))

# ---------- Options

@dataclass
class OptionsPlan:
    structure: str
    dte: int
    strikes: Dict[str, float]
    notes: str

def iv_rank_from_hist_impvol_or_proxy(df: pd.DataFrame) -> float:
    try:
        logret = np.log(df["Close"]).diff()
        hv20 = logret.rolling(20).std() * np.sqrt(252)
        hv = hv20.dropna()
        if len(hv) < 60: return 0.3
        cur, lo, hi = hv.iloc[-1], hv.min(), hv.max()
        if hi - lo < 1e-9: return 0.3
        return float((cur - lo) / (hi - lo))
    except Exception:
        return 0.3

def choose_options_structure(iv_rank: float, bullish: bool, entry: float, tp1: float, sl: float) -> OptionsPlan:
    rr = abs(tp1 - entry) / max(1e-9, abs(entry - sl))
    if iv_rank < 0.3:
        structure = "Debit Call Vertical" if bullish else "Debit Put Vertical"
        dte = 21 if rr >= 1.5 else 28
        strikes = {"long": round(entry * (1.02 if bullish else 0.98), 2), "short": round(tp1, 2)}
        notes = "Low IV: favor debits; short leg near TP1."
    elif iv_rank > 0.6:
        structure = "Bull Put Credit Spread" if bullish else "Bear Call Credit Spread"
        dte = 21
        strikes = {"short": round(sl * (0.98 if bullish else 1.02), 2), "long": round(sl * (0.96 if bullish else 1.04), 2)}
        notes = "High IV: sell credit below/above invalidation."
    else:
        structure = "Call Vertical" if bullish else "Put Vertical"
        dte = 28
        strikes = {"long": round(entry * (1.01 if bullish else 0.99), 2), "short": round(tp1, 2)}
        notes = "Medium IV: defined-risk vertical."
    return OptionsPlan(structure, dte, strikes, notes)

# ---------- Trade plan

def compile_trade_plan(df: pd.DataFrame,
                       patterns: List[DetectedPattern],
                       sr_levels: List[Tuple[float,int]],
                       vp: Dict[str,np.ndarray],
                       avwap: Optional[pd.Series],
                       volz: pd.Series,
                       macd_line: pd.Series,
                       rsi_line: pd.Series) -> Dict:
    price = float(df["Close"].iloc[-1])
    rng_atr = float(atr(df,14).iloc[-1])

    # Determine trade direction based on pattern type
    bullish_patterns = ["CupAndHandle", "BullFlag", "InverseHeadAndShoulders", "DoubleBottom"]
    bearish_patterns = ["HeadAndShoulders", "DoubleTop"]

    direction = "NEUTRAL"
    if patterns:
        top_pattern = patterns[0]  # Highest confidence pattern
        if top_pattern.name in bullish_patterns:
            direction = "LONG"
        elif top_pattern.name in bearish_patterns:
            direction = "SHORT"

    # Calculate entry, stop, and targets based on pattern
    rim = None
    resistance = price
    support = price
    stop_candidates = []
    entry = price
    breakout = False

    for p in patterns:
        if p.name == "CupAndHandle":
            rim = p.meta["rim_level"]
            resistance = rim
            breakout = price > (resistance + 0.25*rng_atr)
            entry = price if breakout else resistance
            stop_candidates.append(float(df["Low"].iloc[p.meta["handle_start"]:p.meta["handle_end"]+1].min()))
        elif p.name == "InverseHeadAndShoulders":
            resistance = p.meta["neckline_price"]
            breakout = price > resistance * 1.01
            entry = resistance * 1.01
            stop_candidates.append(p.meta["right_shoulder"] * 0.98)
        elif p.name == "DoubleBottom":
            resistance = p.meta["resistance"]
            breakout = price > resistance * 1.01
            entry = resistance * 1.01
            stop_candidates.append(min(p.meta["first_low"], p.meta["second_low"]) * 0.98)
        elif p.name == "HeadAndShoulders":
            support = p.meta["neckline_price"]
            breakout = price < support * 0.99
            entry = support * 0.99
            stop_candidates.append(p.meta["right_shoulder"] * 1.02)
        elif p.name == "DoubleTop":
            support = p.meta["support"]
            breakout = price < support * 0.99
            entry = support * 0.99
            stop_candidates.append(max(p.meta["first_high"], p.meta["second_high"]) * 1.02)

    # Add general stop candidates
    swing_lows = find_pivots(df,5)[1]
    swing_highs = find_pivots(df,5)[0]

    if avwap is not None and not math.isnan(avwap.iloc[-1]):
        if direction == "LONG":
            stop_candidates.append(float(avwap.iloc[-1] - 0.1*rng_atr))
        else:
            stop_candidates.append(float(avwap.iloc[-1] + 0.1*rng_atr))

    if direction == "LONG":
        if swing_lows:
            stop_candidates.append(float(df["Low"].iloc[swing_lows[-1]]))
        if not stop_candidates:
            stop_candidates = [price - 2*rng_atr]
        stop = float(min(stop_candidates))
    else:
        if swing_highs:
            stop_candidates.append(float(df["High"].iloc[swing_highs[-1]]))
        if not stop_candidates:
            stop_candidates = [price + 2*rng_atr]
        stop = float(max(stop_candidates))

    # Calculate targets
    tp1, tp2 = price + 1.5*rng_atr, price + 3*rng_atr

    for p in patterns:
        if p.name == "CupAndHandle":
            depth = p.meta["cup_depth_pct"] * p.meta["rim_level"]
            tp1 = max(tp1, float(resistance + depth))
            tp2 = max(tp2, float(resistance + 1.5*depth))
        elif p.name == "InverseHeadAndShoulders" and "pattern_height" in p.meta:
            height = p.meta["pattern_height"]
            tp1 = max(tp1, p.meta["neckline_price"] + height)
            tp2 = max(tp2, p.meta["neckline_price"] + height * 1.5)
        elif p.name == "DoubleBottom" and "pattern_height" in p.meta:
            height = p.meta["pattern_height"]
            tp1 = max(tp1, p.meta["resistance"] + height)
            tp2 = max(tp2, p.meta["resistance"] + height * 1.5)
        elif p.name == "HeadAndShoulders" and "pattern_height" in p.meta:
            height = p.meta["pattern_height"]
            tp1 = min(tp1, p.meta["neckline_price"] - height)
            tp2 = min(tp2, p.meta["neckline_price"] - height * 1.5)
        elif p.name == "DoubleTop" and "pattern_height" in p.meta:
            height = p.meta["pattern_height"]
            tp1 = min(tp1, p.meta["support"] - height)
            tp2 = min(tp2, p.meta["support"] - height * 1.5)
    hvns, lvns = hvn_lvn(vp)
    if hvns:
        above = [h for h in hvns if h > price]
        if above:
            tp1 = min(above, key=lambda x: abs(x - tp1))
            farther = [h for h in above if h > tp1*1.02]
            if farther:
                tp2 = min(farther, key=lambda x: abs(x - tp2))
    conf = confluence_score(df, patterns, sr_levels, vp, volz, macd_line, rsi_line)
    rr = float((tp1 - entry) / max(1e-9, abs(entry - stop)))

    # Override direction with indicator confirmation if no strong pattern
    if direction == "NEUTRAL" and (breakout or (macd_line.iloc[-1] > 0 and rsi_line.iloc[-1] > 45)):
        direction = "LONG"

    ivr = iv_rank_from_hist_impvol_or_proxy(df)
    opt = choose_options_structure(ivr, bullish=(direction=="LONG"), entry=entry, tp1=tp1, sl=stop)
    plan = {
        "symbol": "TICK",
        "timeframe": "1D",
        "thesis": ", ".join(p.name for p in patterns) or "Price Action",
        "direction": direction,
        "entry": round(entry,2),
        "stop": round(stop,2),
        "targets": [round(tp1,2), round(tp2,2)],
        "quality": {"confluence": round(conf,3),
                    "volume_z": round(float(volz.iloc[-1]),2),
                    "rsi": round(float(rsi_line.iloc[-1]),1),
                    "macd": round(float(macd_line.iloc[-1]),2)},
        "risk_reward": round(rr,2),
        "options": asdict(opt),
        "notes": "Breakout confirmation requires close above resistance + volume."
    }
    return plan

# ---------- Overlay

def overlay_spec(df: pd.DataFrame,
                 patterns: List[DetectedPattern],
                 fib: Dict,
                 vp: Dict[str,np.ndarray],
                 plan: Dict,
                 avwap: Optional[pd.Series]) -> Dict:
    layers = []
    for p in patterns:
        if p.name == "CupAndHandle":
            layers.append({"type":"band","style":"rim","y":p.meta["rim_level"],"height":0.2*df['Close'].iloc[-1]/100})
            layers.append({"type":"curve","style":"cup","start_idx":p.meta["rim_idx1"],"mid_idx":p.meta["cup_low_idx"],"end_idx":p.meta["rim_idx2"]})
            layers.append({"type":"channel","style":"handle","start_idx":p.meta["handle_start"],"end_idx":p.meta["handle_end"]})
        elif p.name == "BullFlag":
            layers.append({"type":"channel","style":"flag","start_idx":p.meta["flag_start_idx"],"end_idx":len(df)-1})
        elif p.name == "HeadAndShoulders":
            layers.append({"type":"pattern_markers","style":"h_and_s","points":[
                {"idx":p.meta["left_shoulder_idx"],"y":p.meta["left_shoulder"],"label":"LS"},
                {"idx":p.meta["head_idx"],"y":p.meta["head"],"label":"H"},
                {"idx":p.meta["right_shoulder_idx"],"y":p.meta["right_shoulder"],"label":"RS"}
            ]})
            layers.append({"type":"hline","style":"neckline","label":"Neckline","y":p.meta["neckline_price"]})
        elif p.name == "InverseHeadAndShoulders":
            layers.append({"type":"pattern_markers","style":"inv_h_and_s","points":[
                {"idx":p.meta["left_shoulder_idx"],"y":p.meta["left_shoulder"],"label":"LS"},
                {"idx":p.meta["head_idx"],"y":p.meta["head"],"label":"H"},
                {"idx":p.meta["right_shoulder_idx"],"y":p.meta["right_shoulder"],"label":"RS"}
            ]})
            layers.append({"type":"hline","style":"neckline","label":"Neckline","y":p.meta["neckline_price"]})
        elif p.name == "DoubleBottom":
            layers.append({"type":"pattern_markers","style":"double_bottom","points":[
                {"idx":p.meta["first_low_idx"],"y":p.meta["first_low"],"label":"B1"},
                {"idx":p.meta["second_low_idx"],"y":p.meta["second_low"],"label":"B2"}
            ]})
            layers.append({"type":"hline","style":"resistance","label":"Resistance","y":p.meta["resistance"]})
        elif p.name == "DoubleTop":
            layers.append({"type":"pattern_markers","style":"double_top","points":[
                {"idx":p.meta["first_high_idx"],"y":p.meta["first_high"],"label":"T1"},
                {"idx":p.meta["second_high_idx"],"y":p.meta["second_high"],"label":"T2"}
            ]})
            layers.append({"type":"hline","style":"support","label":"Support","y":p.meta["support"]})
    if avwap is not None:
        layers.append({"type":"line_series","style":"avwap","y": [None if math.isnan(y) else float(y) for y in avwap.values]})
    layers.append({"type":"vpvr","centers": vp["price_centers"].tolist(),"volume": vp["volume"].tolist()})
    if fib:
        for name, y in fib["levels"].items():
            layers.append({"type":"hline","style":"fib","label":name,"y":float(y)})
    layers += [
        {"type":"marker","kind":"ENTRY","y":plan["entry"]},
        {"type":"marker","kind":"STOP","y":plan["stop"]},
        {"type":"marker","kind":"TP","y":plan["targets"][0]},
        {"type":"marker","kind":"TP","y":plan["targets"][1]},
        {"type":"riskbox","entry":plan["entry"],"stop":plan["stop"],"targets":plan["targets"]}
    ]
    return {"layers": layers}

# ---------- Orchestrator

@dataclass
class AITAConfig:
    period: str = "6mo"
    interval: str = "1d"
    pivot_order: int = 5
    vpvr_bins: int = 60

class AITA:
    def __init__(self, cfg: AITAConfig=AITAConfig()):
        self.cfg = cfg

    def analyze(self, symbol: str) -> Dict:
        data = yf.download(symbol, period=self.cfg.period, interval=self.cfg.interval, auto_adjust=True)
        if data.empty:
            raise RuntimeError(f"No data for {symbol}")
        # Flatten MultiIndex columns if present (yfinance compatibility)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data = data.dropna().copy()
        data["ATR"] = atr(data,14)
        data["RSI"] = rsi(data,14)
        m_line, m_sig, m_hist = macd(data)
        data["MACD"] = m_line
        data["VOLZ"] = vol_zscore(data,20)

        patterns: List[DetectedPattern] = []

        # Detect bullish patterns
        cnh = detect_cup_handle(data, order=self.cfg.pivot_order)
        if cnh: patterns.append(cnh)
        flag = detect_bull_flag(data)
        if flag: patterns.append(flag)
        inv_hs = detect_inverse_head_shoulders(data, order=self.cfg.pivot_order)
        if inv_hs: patterns.append(inv_hs)
        db = detect_double_bottom(data, order=self.cfg.pivot_order)
        if db: patterns.append(db)

        # Detect bearish patterns
        hs = detect_head_shoulders(data, order=self.cfg.pivot_order)
        if hs: patterns.append(hs)
        dt = detect_double_top(data, order=self.cfg.pivot_order)
        if dt: patterns.append(dt)

        # Sort by score (highest confidence first)
        patterns.sort(key=lambda p: p.score, reverse=True)

        srlvls = sr_levels_from_pivots(data, order=self.cfg.pivot_order)
        fib = fib_levels_from_last_swing(data, order=self.cfg.pivot_order)
        vp = vpvr(data, bins=self.cfg.vpvr_bins)

        anchor_idx = None
        if cnh:
            anchor_idx = cnh.meta["rim_idx2"] + 1
        else:
            lows = find_pivots(data, self.cfg.pivot_order)[1]
            if lows:
                anchor_idx = lows[-1]
        avw = anchored_vwap(data, anchor_idx) if anchor_idx is not None else None

        plan = compile_trade_plan(data, patterns, srlvls, vp, avw, data["VOLZ"], data["MACD"], data["RSI"])
        plan["symbol"] = symbol
        plan["timeframe"] = self.cfg.interval.upper()

        overlay = overlay_spec(data, patterns, fib, vp, plan, avw)

        top_sr = [{"level": round(l,2), "touches": t} for l,t in srlvls[:5]]
        summary = {
            "symbol": symbol,
            "price": round(float(data["Close"].iloc[-1]),2),
            "patterns": [{"name": p.name, "score": round(p.score,2)} for p in patterns],
            "top_sr_levels": top_sr,
            "plan": plan
        }
        return {"summary": summary, "overlay": overlay}

if __name__ == "__main__":
    a = AITA()
    out = a.analyze("AVGO")
    print(json.dumps(out["summary"], indent=2))
