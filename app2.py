"""
Crypto Tops & Bottoms Detector Bot - النسخة المتقدمة والمحسنة (v3.1)
إصدار 3.1 - تحسين الاتصال ومعالجة الأخطاء، تحديث CCXT، وإدارة الطلبات
"""

import os
import json
import time
import math
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from threading import Lock
from collections import deque

import numpy as np
from flask import Flask, render_template, jsonify, request
import ccxt
import backoff
from ratelimit import limits, RateLimitException

# فحص إصدار CCXT
logger = logging.getLogger(__name__)
logger.info(f"CCXT version: {ccxt.__version__}")
if ccxt.__version__ < '4.4.0':
    logger.warning("⚠️ CCXT version is old. Recommended to upgrade to 4.4.0+")

# ======================
# إعدادات التسجيل
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crypto_tops_bottoms.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ======================
# هياكل البيانات الأساسية
# ======================
@dataclass
class CoinConfig:
    symbol: str
    name: str
    base_asset: str
    quote_asset: str
    enabled: bool = True

@dataclass
class TopBottomSignal:
    coin_symbol: str
    coin_name: str
    signal_type: str  # "TOP" or "BOTTOM"
    confidence: float  # 0-100
    price: float
    timestamp: datetime
    indicators: Dict[str, Any]
    message: str

@dataclass
class Notification:
    id: str
    timestamp: datetime
    coin_symbol: str
    coin_name: str
    message: str
    notification_type: str
    signal_strength: float
    price: float

# ======================
# إعدادات التطبيق (محدثة)
# ======================
class AppConfig:
    @staticmethod
    def get_top_coins(limit=15):
        """جلب أفضل العملات من حيث حجم التداول من Binance"""
        try:
            exchange = ccxt.binance()
            tickers = exchange.fetch_tickers()
            usdt_pairs = {k: v for k, v in tickers.items() 
                         if k.endswith('/USDT') and v.get('quoteVolume')}
            sorted_pairs = sorted(usdt_pairs.items(), 
                                key=lambda x: x[1]['quoteVolume'] or 0, 
                                reverse=True)
            coins = []
            EXCLUDED_COINS = ['LUNA', 'UST', 'FTT', 'TERRA', 'USD1', 'USDC']
            for symbol, ticker in sorted_pairs[:limit]:
                base = symbol.replace('/USDT', '')
                if base not in EXCLUDED_COINS:
                    coins.append(CoinConfig(symbol, base, base, 'USDT'))
            if coins:
                logger.info(f"✅ تم جلب {len(coins)} عملة من Binance")
                return coins
            else:
                return AppConfig._get_default_coins()
        except Exception as e:
            logger.error(f"❌ خطأ في جلب العملات: {e}")
            return AppConfig._get_default_coins()

    @staticmethod
    def _get_default_coins():
        return [
            CoinConfig("BTC/USDT", "Bitcoin", "BTC", "USDT"),
            CoinConfig("ETH/USDT", "Ethereum", "ETH", "USDT"),
            CoinConfig("BNB/USDT", "Binance Coin", "BNB", "USDT"),
            CoinConfig("SOL/USDT", "Solana", "SOL", "USDT"),
            CoinConfig("XRP/USDT", "Ripple", "XRP", "USDT"),
            CoinConfig("ADA/USDT", "Cardano", "ADA", "USDT"),
            CoinConfig("DOGE/USDT", "Dogecoin", "DOGE", "USDT"),
            CoinConfig("AVAX/USDT", "Avalanche", "AVAX", "USDT"),
            CoinConfig("DOT/USDT", "Polkadot", "DOT", "USDT"),
            CoinConfig("MATIC/USDT", "Polygon", "MATIC", "USDT"),
        ]

    COINS = get_top_coins(15)

    TIMEFRAME = '15m'
    HIGHER_TIMEFRAME = '1h'
    MAX_CANDLES = 300
    MIN_CANDLES_REQUIRED = 30  # تم التخفيض من 50 إلى 30

    # إعدادات Pivot
    PIVOT_LEFT = 5
    PIVOT_RIGHT = 5
    MIN_PIVOT_DISTANCE_ATR = 1.5      # مضاعف ATR للمسافة بين القمم/القيعان

    # Fractal
    FRACTAL_PERIOD = 2                 # 2 شمعة على كل جانب => 5 شموع

    # عتبات الثقة (تم تخفيضها قليلاً لزيادة الحساسية)
    TOP_CONFIDENCE_THRESHOLD = 55
    BOTTOM_CONFIDENCE_THRESHOLD = 55

    UPDATE_INTERVAL = 120

    # فلتر الإشعارات
    COOLDOWN_SECONDS = 300
    MIN_PRICE_MOVE_PERCENT = 0.8
    MIN_VOLATILITY_ATR_PERCENT = 0.5   # إذا كانت ATR% أقل من هذا، السوق هادئ

    # أوزان المؤشرات (محسنة)
    WEIGHTS = {
        'trend': {
            'pivot': 0.25,
            'rsi_div': 0.30,
            'fractal': 0.15,
            'volume': 0.10,
            'msb': 0.20
        },
        'ranging': {
            'pivot': 0.30,
            'rsi_div': 0.20,
            'fractal': 0.25,
            'volume': 0.10,
            'msb': 0.15
        }
    }

    # إضافة أنماط الشموع
    ENABLE_CANDLE_PATTERNS = True

# ======================
# إعدادات APIs الخارجية
# ======================
class ExternalAPIConfig:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_tops_bottoms_advanced')
    NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 2

# ======================
# Binance Client محدث مع تحسينات
# ======================
class BinanceClient:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
            'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'rateLimit': 50,  # تم التعديل للإصدار الجديد
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
                'maxRetriesOnFailure': 3,  # ميزة جديدة
            }
        })
        self.session = requests.Session()
        self.last_request_time = {}
        self.min_request_interval = 1.5  # ثانية بين الطلبات لنفس العملة

    def _wait_for_rate_limit(self, symbol):
        """تأخير بين الطلبات لنفس العملة"""
        now = time.time()
        if symbol in self.last_request_time:
            elapsed = now - self.last_request_time[symbol]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[symbol] = time.time()

    @backoff.on_exception(
        backoff.expo,
        (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException),
        max_tries=5,
        max_time=60,
        giveup=lambda e: isinstance(e, ccxt.BadSymbol)
    )
    def fetch_ohlcv(self, symbol: str, timeframe: str = AppConfig.TIMEFRAME, limit: int = AppConfig.MAX_CANDLES) -> Optional[List]:
        """جلب بيانات OHLCV مع تأخير وإعادة محاولة"""
        try:
            self._wait_for_rate_limit(symbol)
            # استخدام fetch_ohlcv مع معالجة أفضل للأخطاء
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                return ohlcv
            else:
                logger.warning(f"No data returned for {symbol}")
                return None
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {symbol}, waiting longer...")
            time.sleep(5)
            return None
        except ccxt.NetworkError as e:
            logger.error(f"Network error for {symbol}: {e}")
            time.sleep(3)
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            return None

    @backoff.on_exception(
        backoff.expo,
        (ccxt.RateLimitExceeded, ccxt.NetworkError, requests.exceptions.RequestException),
        max_tries=3,
        max_time=30
    )
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """جلب التicker مع تأخير"""
        try:
            self._wait_for_rate_limit(symbol)
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def fetch_ohlcv_with_fallback(self, symbol: str, timeframe: str = AppConfig.TIMEFRAME, limit: int = AppConfig.MAX_CANDLES) -> Optional[List]:
        """محاولة جلب البيانات من مصادر متعددة إذا فشل المصدر الرئيسي"""
        data = self.fetch_ohlcv(symbol, timeframe, limit)
        if data:
            return data
        # المحاولة الثانية: تقليل الـ limit
        logger.info(f"Retrying {symbol} with smaller limit...")
        data = self.fetch_ohlcv(symbol, timeframe, min(limit, 100))
        if data:
            return data
        return None

    def fetch_multiple_tickers(self, symbols: List[str]) -> Dict:
        """محاولة جلب عدة tickers مرة واحدة (إن أمكن)"""
        try:
            # في الإصدارات الحديثة، fetchTickers يمكن أن تأخذ قائمة رموز
            tickers = self.exchange.fetch_tickers(symbols)
            return tickers
        except Exception as e:
            logger.error(f"Error fetching multiple tickers: {e}")
            return {}

# ======================
# المؤشرات الفنية المحسوبة يدوياً - نسخة محسنة (معالج الأخطاء)
# ======================
class TechnicalIndicators:
    """حساب المؤشرات مع معالجة الأخطاء والتحسينات"""

    @staticmethod
    def sma(values: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        result = [None] * len(values)
        for i in range(period - 1, len(values)):
            result[i] = sum(values[i - period + 1:i + 1]) / period
        return result

    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if not values:
            return []
        k = 2 / (period + 1)
        ema_values = [values[0]]
        for i in range(1, len(values)):
            ema_values.append(values[i] * k + ema_values[-1] * (1 - k))
        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """Relative Strength Index - مصحح بالكامل"""
        if len(prices) < period + 1:
            return [None] * len(prices)

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [None] * period

        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_values.append(rsi)

            if i < len(prices) - 1:
                gain = gains[i]
                loss = losses[i]
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period

        return rsi_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average True Range - مصحح"""
        length = len(closes)
        if length < period + 1:
            return [None] * length

        tr = [0.0] * length
        tr[0] = highs[0] - lows[0]
        for i in range(1, length):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        atr_values = [None] * length
        atr_values[period - 1] = sum(tr[:period]) / period

        for i in range(period, length):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period

        return atr_values

    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average Directional Index - مصحح"""
        length = len(closes)
        if length < period * 2:
            return [None] * length

        # حساب +DM و -DM
        plus_dm = [0.0] * length
        minus_dm = [0.0] * length
        for i in range(1, length):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0.0

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0.0

        # حساب TR
        tr = [0.0] * length
        tr[0] = highs[0] - lows[0]
        for i in range(1, length):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        # تجانس
        smoothed_tr = [None] * length
        smoothed_plus_dm = [None] * length
        smoothed_minus_dm = [None] * length

        smoothed_tr[period-1] = sum(tr[:period]) / period
        smoothed_plus_dm[period-1] = sum(plus_dm[:period]) / period
        smoothed_minus_dm[period-1] = sum(minus_dm[:period]) / period

        for i in range(period, length):
            smoothed_tr[i] = (smoothed_tr[i-1] * (period - 1) + tr[i]) / period
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (period - 1) + plus_dm[i]) / period
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (period - 1) + minus_dm[i]) / period

        # حساب +DI و -DI
        plus_di = [None] * length
        minus_di = [None] * length
        dx = [None] * length

        for i in range(period-1, length):
            if smoothed_tr[i] and smoothed_tr[i] != 0:
                plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX
        adx = [None] * length
        dx_values = [dx[i] for i in range(length) if dx[i] is not None]
        if len(dx_values) >= period:
            start_idx = (period - 1) + period
            if start_idx < length:
                adx[start_idx] = sum(dx_values[:period]) / period
                for i in range(start_idx + 1, length):
                    if dx[i] is not None:
                        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

        return adx

    @staticmethod
    def pivot_points(highs: List[float], lows: List[float], left: int = 5, right: int = 5) -> Tuple[List[bool], List[bool]]:
        """تحديد القمم والقيعان المحلية"""
        length = len(highs)
        pivot_highs = [False] * length
        pivot_lows = [False] * length
        for i in range(left, length - right):
            # قمة
            if all(highs[i] >= highs[i - j] for j in range(1, left + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, right + 1)):
                pivot_highs[i] = True
            # قاع
            if all(lows[i] <= lows[i - j] for j in range(1, left + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, right + 1)):
                pivot_lows[i] = True
        return pivot_highs, pivot_lows

    @staticmethod
    def fractal(highs: List[float], lows: List[float], period: int = 2) -> Tuple[List[bool], List[bool]]:
        """Fractals of Bill Williams (5 candles)"""
        length = len(highs)
        fractal_up = [False] * length
        fractal_down = [False] * length
        for i in range(period, length - period):
            # Fractal up (قمة)
            if all(highs[i] > highs[i - j] for j in range(1, period + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, period + 1)):
                fractal_up[i] = True
            # Fractal down (قاع)
            if all(lows[i] < lows[i - j] for j in range(1, period + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, period + 1)):
                fractal_down[i] = True
        return fractal_up, fractal_down

    @staticmethod
    def detect_divergence(prices: List[float], oscillator: List[Optional[float]], window: int = 30) -> Dict[str, bool]:
        """كشف الانحراف (divergence) باستخدام القمم والقيعان المحلية - محسّن"""
        recent_prices = prices[-window:]
        recent_osc = oscillator[-window:]

        valid_indices = [i for i, v in enumerate(recent_osc) if v is not None]
        if len(valid_indices) < 10:
            return {'bullish': False, 'bearish': False}

        prices_valid = [recent_prices[i] for i in valid_indices]
        osc_valid = [recent_osc[i] for i in valid_indices]

        def find_peaks(arr):
            peaks = []
            for i in range(1, len(arr) - 1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                    peaks.append((i, arr[i]))
            return peaks

        def find_troughs(arr):
            troughs = []
            for i in range(1, len(arr) - 1):
                if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                    troughs.append((i, arr[i]))
            return troughs

        price_peaks = find_peaks(prices_valid)
        price_troughs = find_troughs(prices_valid)
        osc_peaks = find_peaks(osc_valid)
        osc_troughs = find_troughs(osc_valid)

        bullish_div = False
        bearish_div = False

        # انحراف صاعد (Bullish): قاع سعري أدنى مع قاع مذبذب أعلى
        if len(price_troughs) >= 2 and len(osc_troughs) >= 2:
            last_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            last_osc_trough = osc_troughs[-1]
            prev_osc_trough = osc_troughs[-2]

            if last_price_trough[1] < prev_price_trough[1] and last_osc_trough[1] > prev_osc_trough[1]:
                bullish_div = True

        # انحراف هابط (Bearish): قمة سعرية أعلى مع قمة مذبذب أقل
        if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            last_osc_peak = osc_peaks[-1]
            prev_osc_peak = osc_peaks[-2]

            if last_price_peak[1] > prev_price_peak[1] and last_osc_peak[1] < prev_osc_peak[1]:
                bearish_div = True

        return {'bullish': bullish_div, 'bearish': bearish_div}

    @staticmethod
    def market_structure(highs: List[float], lows: List[float], closes: List[float],
                         pivot_highs: List[bool], pivot_lows: List[bool]) -> Dict[str, Any]:
        """تحليل هيكل السوق: تحديد الاتجاه، كسر الهيكل (BOS)، تغير الطابع (CHoCH)"""
        length = len(highs)

        high_indices = [i for i, v in enumerate(pivot_highs) if v]
        low_indices = [i for i, v in enumerate(pivot_lows) if v]

        if len(high_indices) < 2 or len(low_indices) < 2:
            return {
                'trend': 'unknown',
                'bos_up': False,
                'bos_down': False,
                'choch_up': False,
                'choch_down': False,
                'last_high_idx': None,
                'last_low_idx': None
            }

        last_high_idx = high_indices[-1]
        last_low_idx = low_indices[-1]
        prev_high_idx = high_indices[-2] if len(high_indices) >= 2 else None
        prev_low_idx = low_indices[-2] if len(low_indices) >= 2 else None

        uptrend = False
        downtrend = False

        if prev_high_idx is not None and prev_low_idx is not None:
            if highs[last_high_idx] > highs[prev_high_idx] and lows[last_low_idx] > lows[prev_low_idx]:
                uptrend = True
            elif highs[last_high_idx] < highs[prev_high_idx] and lows[last_low_idx] < lows[prev_low_idx]:
                downtrend = True

        bos_up = False
        bos_down = False
        if uptrend and closes[-1] > highs[last_high_idx]:
            bos_up = True
        if downtrend and closes[-1] < lows[last_low_idx]:
            bos_down = True

        choch_up = False
        choch_down = False
        if downtrend and closes[-1] > highs[last_high_idx]:
            choch_up = True
        if uptrend and closes[-1] < lows[last_low_idx]:
            choch_down = True

        trend = 'uptrend' if uptrend else 'downtrend' if downtrend else 'ranging'

        return {
            'trend': trend,
            'bos_up': bos_up,
            'bos_down': bos_down,
            'choch_up': choch_up,
            'choch_down': choch_down,
            'last_high_idx': last_high_idx,
            'last_low_idx': last_low_idx
        }

    @staticmethod
    def candle_patterns(open_prices: List[float], high: List[float], low: List[float], close: List[float]) -> Dict[str, bool]:
        """كشف أنماط الشموع الرئيسية للقمم والقيعان"""
        if len(close) < 5:
            return {'shooting_star': False, 'hammer': False, 'engulfing_bear': False, 'engulfing_bull': False}

        # الشموع الثلاث الأخيرة
        o1, o2, o3 = open_prices[-1], open_prices[-2], open_prices[-3]
        h1, h2, h3 = high[-1], high[-2], high[-3]
        l1, l2, l3 = low[-1], low[-2], low[-3]
        c1, c2, c3 = close[-1], close[-2], close[-3]

        patterns = {
            'shooting_star': False,
            'hammer': False,
            'engulfing_bear': False,
            'engulfing_bull': False
        }

        # Shooting Star (قمة محتملة)
        body1 = abs(c1 - o1)
        upper_shadow1 = h1 - max(c1, o1)
        lower_shadow1 = min(c1, o1) - l1
        if upper_shadow1 > 2 * body1 and lower_shadow1 < 0.2 * body1 and c1 < o1:
            patterns['shooting_star'] = True

        # Hammer (قاع محتمل)
        if lower_shadow1 > 2 * body1 and upper_shadow1 < 0.2 * body1 and c1 > o1:
            patterns['hammer'] = True

        # Bearish Engulfing (قمة)
        if c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2:
            patterns['engulfing_bear'] = True

        # Bullish Engulfing (قاع)
        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2:
            patterns['engulfing_bull'] = True

        return patterns

# ======================
# مدير الإشعارات (محسّن)
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}       # (coin, type) -> datetime
        self.last_notification_price = {}      # (coin, type) -> price
        self.cooldown_base = AppConfig.COOLDOWN_SECONDS
        self.cooldown_multiplier = {}           # (coin, type) -> multiplier

    def add(self, notification: Notification):
        self.history.append(notification)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, limit: int = 10) -> List[Notification]:
        return self.history[-limit:] if self.history else []

    def should_send(self, coin_symbol: str, signal_type: str, confidence: float, current_price: float) -> bool:
        now = datetime.now()
        key = (coin_symbol, signal_type)

        last_time = self.last_notification_time.get(key)
        if last_time:
            delta = (now - last_time).total_seconds()
            multiplier = self.cooldown_multiplier.get(key, 1.0)
            cooldown = self.cooldown_base * multiplier
            if delta < cooldown:
                return False

        last_price = self.last_notification_price.get(key)
        if last_price:
            price_move_pct = abs(current_price - last_price) / last_price * 100
            if price_move_pct < AppConfig.MIN_PRICE_MOVE_PERCENT:
                return False

        return True

    def send_ntfy(self, message: str, title: str = "Crypto Top/Bottom", priority: str = "3", tags: str = "chart") -> bool:
        try:
            headers = {
                "Title": title,
                "Priority": priority,
                "Tags": tags,
                "Content-Type": "text/plain; charset=utf-8"
            }
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            resp = requests.post(
                ExternalAPIConfig.NTFY_URL,
                data=safe_message.encode('utf-8'),
                headers=headers,
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"NTFY error: {e}")
            return False

    def create_notification(self, signal: TopBottomSignal) -> Optional[Notification]:
        key = (signal.coin_symbol, signal.signal_type)
        if not self.should_send(signal.coin_symbol, signal.signal_type, signal.confidence, signal.price):
            return None

        title = f"{signal.signal_type} DETECTED: {signal.coin_name}"
        message = (
            f"{title}\n"
            f"Confidence: {signal.confidence:.1f}%\n"
            f"Price: ${signal.price:,.2f}\n"
            f"Time: {signal.timestamp.strftime('%H:%M')}\n"
            f"Indicators: {json.dumps(signal.indicators, default=str)}"
        )

        tags = "arrow_up" if signal.signal_type == "TOP" else "arrow_down"
        priority = "5" if signal.confidence >= 85 else "4" if signal.confidence >= 65 else "3"

        if self.send_ntfy(message, title, priority, tags):
            notification = Notification(
                id=f"{signal.coin_symbol}_{signal.signal_type}_{int(signal.timestamp.timestamp())}",
                timestamp=signal.timestamp,
                coin_symbol=signal.coin_symbol,
                coin_name=signal.coin_name,
                message=message,
                notification_type=signal.signal_type,
                signal_strength=signal.confidence,
                price=signal.price
            )
            self.add(notification)

            self.last_notification_time[key] = datetime.now()
            self.last_notification_price[key] = signal.price
            multiplier = self.cooldown_multiplier.get(key, 1.0)
            self.cooldown_multiplier[key] = min(multiplier * 1.5, 5.0)

            return notification
        return None

    def reset_cooldown(self, coin_symbol: str, signal_type: str):
        key = (coin_symbol, signal_type)
        self.cooldown_multiplier[key] = max(1.0, self.cooldown_multiplier.get(key, 1.0) * 0.8)

# ======================
# المدير الرئيسي لاكتشاف القمم والقيعان (محسّن)
# ======================
class TopBottomDetector:
    def __init__(self):
        self.detections: List[TopBottomSignal] = []
        self.last_update: Optional[datetime] = None
        self.last_coins_update: Optional[datetime] = None
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()
        self.cached_higher_tf_data: Dict[str, Any] = {}

    def update_coins_list(self):
        """تحديث قائمة العملات كل ساعة"""
        now = datetime.now()
        if not self.last_coins_update or (now - self.last_coins_update).seconds > 3600:
            new_coins = AppConfig.get_top_coins(15)
            if new_coins:
                AppConfig.COINS = new_coins
                self.last_coins_update = now
                logger.info(f"🔄 تم تحديث قائمة العملات: {len(new_coins)} عملة")

    def update_all(self) -> bool:
        with self.lock:
            self.update_coins_list()
            logger.info(f"🔄 Scanning {len(AppConfig.COINS)} coins for tops/bottoms (advanced mode)...")

            success_count = 0
            failed_coins = []

            # محاولة جلب جميع الـ tickers دفعة واحدة لتحسين الأداء
            symbols = [coin.symbol for coin in AppConfig.COINS if coin.enabled]
            all_tickers = self.binance.fetch_multiple_tickers(symbols)

            for coin in AppConfig.COINS:
                if not coin.enabled:
                    continue
                try:
                    # تمرير الـ ticker إذا كان موجوداً في النتائج
                    ticker = all_tickers.get(coin.symbol)
                    signal = self._scan_coin_advanced(coin, ticker)
                    if signal:
                        self.detections.append(signal)
                        self.notification_manager.create_notification(signal)
                        success_count += 1
                    else:
                        failed_coins.append(coin)
                except Exception as e:
                    logger.error(f"Error on {coin.symbol}: {e}", exc_info=True)
                    failed_coins.append(coin)

            # إعادة محاولة العملات الفاشلة (مرة واحدة)
            if failed_coins:
                logger.info(f"🔄 Retrying {len(failed_coins)} failed coins...")
                time.sleep(5)
                for coin in failed_coins:
                    try:
                        signal = self._scan_coin_advanced(coin)
                        if signal:
                            self.detections.append(signal)
                            self.notification_manager.create_notification(signal)
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Retry error on {coin.symbol}: {e}")

            self.last_update = datetime.now()
            if len(self.detections) > 100:
                self.detections = self.detections[-100:]

            logger.info(f"✅ Found {success_count} potential tops/bottoms")
            return success_count > 0

    def _scan_coin_advanced(self, coin: CoinConfig, pre_fetched_ticker: Optional[Dict] = None) -> Optional[TopBottomSignal]:
        """مسح عملة واحدة مع إمكانية استخدام ticker مُسبق"""
        # استخدام fetch_ohlcv_with_fallback للحصول على البيانات
        ohlcv = self.binance.fetch_ohlcv_with_fallback(coin.symbol, AppConfig.TIMEFRAME, AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < AppConfig.MIN_CANDLES_REQUIRED:
            logger.debug(f"Insufficient data for {coin.symbol}: {len(ohlcv) if ohlcv else 0} candles")
            return None

        ohlcv_htf = self.binance.fetch_ohlcv(coin.symbol, AppConfig.HIGHER_TIMEFRAME, 100)

        opens = [c[1] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        # استخدام ticker المقدم إن وجد، وإلا جلبه
        if pre_fetched_ticker:
            ticker = pre_fetched_ticker
        else:
            ticker = self.binance.fetch_ticker(coin.symbol)

        if not ticker:
            logger.warning(f"No ticker for {coin.symbol}")
            return None

        current_price = ticker['last']
        if current_price is None:
            return None

        # حساب المؤشرات
        rsi = TechnicalIndicators.rsi(closes, 14)
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)
        adx = TechnicalIndicators.adx(highs, lows, closes, 14)

        current_rsi = rsi[-1] if rsi[-1] is not None else 50
        current_atr = atr[-1] if atr[-1] is not None else 0
        current_adx = adx[-1] if adx[-1] is not None else 20

        # تحديد نوع السوق
        market_regime = 'trend' if current_adx > 25 else 'ranging'

        # Pivot High/Low
        pivot_highs, pivot_lows = TechnicalIndicators.pivot_points(highs, lows, AppConfig.PIVOT_LEFT, AppConfig.PIVOT_RIGHT)

        # تحقق من القرب من آخر Pivot حقيقي
        near_pivot_high = False
        near_pivot_low = False

        valid_pivot_highs = [i for i, is_high in enumerate(pivot_highs) if is_high]
        valid_pivot_lows = [i for i, is_low in enumerate(pivot_lows) if is_low]

        if valid_pivot_highs and current_atr > 0:
            last_pivot_high_idx = valid_pivot_highs[-1]
            if len(highs) - last_pivot_high_idx <= 20:
                pivot_price = highs[last_pivot_high_idx]
                if abs(current_price - pivot_price) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                    near_pivot_high = True

        if valid_pivot_lows and current_atr > 0:
            last_pivot_low_idx = valid_pivot_lows[-1]
            if len(lows) - last_pivot_low_idx <= 20:
                pivot_price = lows[last_pivot_low_idx]
                if abs(current_price - pivot_price) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                    near_pivot_low = True

        # Fractals
        fractal_up, fractal_down = TechnicalIndicators.fractal(highs, lows, AppConfig.FRACTAL_PERIOD)
        last_fractal_up = any(fractal_up[-AppConfig.FRACTAL_PERIOD*3:])
        last_fractal_down = any(fractal_down[-AppConfig.FRACTAL_PERIOD*3:])

        # Divergence
        divergence = TechnicalIndicators.detect_divergence(closes, rsi, window=40)

        # Market Structure
        ms = TechnicalIndicators.market_structure(highs, lows, closes, pivot_highs, pivot_lows)

        # Multi-timeframe confirmation
        htf_confirmation = self._check_higher_timeframe(coin.symbol, ohlcv_htf)

        # Volume spike
        avg_vol = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else 0
        volume_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False

        # Volatility filter
        atr_percent = (current_atr / current_price) * 100 if current_price > 0 else 0
        if atr_percent < AppConfig.MIN_VOLATILITY_ATR_PERCENT:
            logger.debug(f"{coin.symbol} volatility too low ({atr_percent:.2f}%), skipping")
            return None

        # أنماط الشموع
        patterns = TechnicalIndicators.candle_patterns(opens, highs, lows, closes) if AppConfig.ENABLE_CANDLE_PATTERNS else {}

        # حساب الثقة
        top_confidence = self._calculate_confidence(
            'TOP', current_price, closes, rsi, atr, current_adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime, patterns
        )
        bottom_confidence = self._calculate_confidence(
            'BOTTOM', current_price, closes, rsi, atr, current_adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime, patterns
        )

        # اختيار الإشارة
        signal = None
        if top_confidence >= AppConfig.TOP_CONFIDENCE_THRESHOLD and top_confidence > bottom_confidence:
            if not htf_confirmation.get('trend_up', False) or ms.get('choch_down', False):
                signal = TopBottomSignal(
                    coin_symbol=coin.symbol,
                    coin_name=coin.name,
                    signal_type="TOP",
                    confidence=top_confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    indicators={
                        'rsi': round(current_rsi, 1),
                        'atr_percent': round(atr_percent, 2),
                        'adx': round(current_adx, 1),
                        'market_regime': market_regime,
                        'near_pivot_high': near_pivot_high,
                        'fractal_up': last_fractal_up,
                        'divergence_bearish': divergence.get('bearish', False),
                        'ms_trend': ms['trend'],
                        'ms_choch_down': ms.get('choch_down', False),
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown'),
                        'patterns': patterns
                    },
                    message=f"Top detected with {top_confidence:.1f}% confidence"
                )
        elif bottom_confidence >= AppConfig.BOTTOM_CONFIDENCE_THRESHOLD:
            if not htf_confirmation.get('trend_down', False) or ms.get('choch_up', False):
                signal = TopBottomSignal(
                    coin_symbol=coin.symbol,
                    coin_name=coin.name,
                    signal_type="BOTTOM",
                    confidence=bottom_confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    indicators={
                        'rsi': round(current_rsi, 1),
                        'atr_percent': round(atr_percent, 2),
                        'adx': round(current_adx, 1),
                        'market_regime': market_regime,
                        'near_pivot_low': near_pivot_low,
                        'fractal_down': last_fractal_down,
                        'divergence_bullish': divergence.get('bullish', False),
                        'ms_trend': ms['trend'],
                        'ms_choch_up': ms.get('choch_up', False),
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown'),
                        'patterns': patterns
                    },
                    message=f"Bottom detected with {bottom_confidence:.1f}% confidence"
                )

        return signal

    def _check_higher_timeframe(self, symbol: str, ohlcv_htf: Optional[List]) -> Dict[str, Any]:
        if not ohlcv_htf or len(ohlcv_htf) < 20:
            return {'trend': 'unknown', 'trend_up': False, 'trend_down': False, 'last_high': None, 'last_low': None}
        closes_htf = [c[4] for c in ohlcv_htf]
        highs_htf = [c[2] for c in ohlcv_htf]
        lows_htf = [c[3] for c in ohlcv_htf]

        sma20 = TechnicalIndicators.sma(closes_htf, 20)
        current_sma = sma20[-1] if sma20[-1] is not None else closes_htf[-1]
        trend_up = closes_htf[-1] > current_sma
        trend_down = closes_htf[-1] < current_sma

        last_high = max(highs_htf[-5:])
        last_low = min(lows_htf[-5:])

        return {
            'trend': 'up' if trend_up else 'down' if trend_down else 'sideways',
            'trend_up': trend_up,
            'trend_down': trend_down,
            'last_high': last_high,
            'last_low': last_low
        }

    def _calculate_confidence(self, signal_type: str, price: float, closes: List[float],
                              rsi: List[Optional[float]], atr: List[Optional[float]], adx: float,
                              near_pivot_high: bool, near_pivot_low: bool,
                              fractal_up: bool, fractal_down: bool,
                              divergence: Dict[str, bool],
                              ms: Dict[str, Any],
                              volume_spike: bool,
                              htf_conf: Dict[str, Any],
                              market_regime: str,
                              patterns: Dict[str, bool]) -> float:
        """حساب الثقة باستخدام الأوزان وإضافة أنماط الشموع"""
        weights = AppConfig.WEIGHTS[market_regime]

        confidence = 0.0

        # 1. Pivot / near pivot
        if signal_type == 'TOP' and near_pivot_high:
            confidence += weights['pivot'] * 100
        if signal_type == 'BOTTOM' and near_pivot_low:
            confidence += weights['pivot'] * 100

        # 2. RSI Divergence
        if signal_type == 'TOP' and divergence.get('bearish', False):
            confidence += weights['rsi_div'] * 100
        if signal_type == 'BOTTOM' and divergence.get('bullish', False):
            confidence += weights['rsi_div'] * 100

        # 3. Fractal
        if signal_type == 'TOP' and fractal_up:
            confidence += weights['fractal'] * 100
        if signal_type == 'BOTTOM' and fractal_down:
            confidence += weights['fractal'] * 100

        # 4. Volume spike
        if volume_spike:
            confidence += weights['volume'] * 100

        # 5. Market Structure Break / Change of Character
        if signal_type == 'TOP' and (ms.get('bos_down', False) or ms.get('choch_down', False)):
            confidence += weights['msb'] * 100
        if signal_type == 'BOTTOM' and (ms.get('bos_up', False) or ms.get('choch_up', False)):
            confidence += weights['msb'] * 100

        # 6. RSI extremes
        current_rsi = rsi[-1] if rsi[-1] is not None else 50
        if signal_type == 'TOP' and current_rsi > 70:
            confidence += 10
        if signal_type == 'BOTTOM' and current_rsi < 30:
            confidence += 10

        # 7. Multi-timeframe confirmation
        if signal_type == 'TOP' and htf_conf.get('trend_down', False):
            confidence += 15
        if signal_type == 'BOTTOM' and htf_conf.get('trend_up', False):
            confidence += 15

        # 8. Candle patterns
        if signal_type == 'TOP':
            if patterns.get('shooting_star', False):
                confidence += 10
            if patterns.get('engulfing_bear', False):
                confidence += 15
        if signal_type == 'BOTTOM':
            if patterns.get('hammer', False):
                confidence += 10
            if patterns.get('engulfing_bull', False):
                confidence += 15

        return min(confidence, 100)

    def get_recent_detections(self, limit: int = 20) -> List[Dict]:
        recent = self.detections[-limit:] if self.detections else []
        return [asdict(d) for d in recent]

    def get_stats(self) -> Dict:
        now = datetime.now()
        last_up = self.last_update
        status = 'healthy'
        if last_up and (now - last_up).total_seconds() > 600:
            status = 'warning'

        tops = sum(1 for d in self.detections if d.signal_type == "TOP")
        bottoms = sum(1 for d in self.detections if d.signal_type == "BOTTOM")

        return {
            'status': status,
            'last_update': last_up.isoformat() if last_up else None,
            'coins_tracked': len(AppConfig.COINS),
            'total_detections': len(self.detections),
            'tops': tops,
            'bottoms': bottoms,
            'notifications_sent': len(self.notification_manager.history)
        }

# ======================
# المحدّث الخلفي
# ======================
def background_updater():
    while True:
        try:
            detector.update_all()
            time.sleep(AppConfig.UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Update error: {e}")
            time.sleep(60)

# ======================
# تطبيق Flask
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-tops-bottoms-advanced-secret')
detector = TopBottomDetector()
start_time = time.time()

updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

detector.update_all()

# ======================
# المسارات (Routes)
# ======================
@app.route('/')
def index():
    detections = detector.get_recent_detections(10)
    stats = detector.get_stats()
    return render_template('index_tops_bottoms.html', detections=detections, stats=stats)

@app.route('/api/detections')
def api_detections():
    limit = request.args.get('limit', 20, type=int)
    return jsonify({
        'status': 'success',
        'data': detector.get_recent_detections(limit),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/update', methods=['POST'])
def manual_update():
    success = detector.update_all()
    return jsonify({
        'status': 'success' if success else 'warning',
        'message': 'Scan completed',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health')
def health():
    stats = detector.get_stats()
    stats['uptime'] = time.time() - start_time
    return jsonify(stats)

@app.route('/api/notifications')
def get_notifications():
    limit = request.args.get('limit', 10, type=int)
    nots = detector.notification_manager.get_recent(limit)
    return jsonify({
        'notifications': [asdict(n) for n in nots],
        'total': len(detector.notification_manager.history)
    })

@app.route('/api/test_ntfy')
def test_ntfy():
    msg = "Test notification - Advanced Tops & Bottoms detector is working"
    success = detector.notification_manager.send_ntfy(msg, "Test", "3", "test_tube")
    return jsonify({'success': success})

def send_startup_notification():
    try:
        msg = (
            f"Crypto Tops & Bottoms Detector Advanced Started\n"
            f"Tracking {len(AppConfig.COINS)} coins\n"
            f"Update interval: {AppConfig.UPDATE_INTERVAL//60} minutes\n"
            f"Threshold: {AppConfig.TOP_CONFIDENCE_THRESHOLD}%"
        )
        detector.notification_manager.send_ntfy(msg, "System Started", "3", "rocket")
    except Exception as e:
        logger.error(f"Startup notification error: {e}")

def delayed_startup():
    time.sleep(5)
    send_startup_notification()

threading.Thread(target=delayed_startup, daemon=True).start()

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 Crypto Tops & Bottoms Detector Advanced v3.1 (Enhanced Connection)")
    logger.info(f"📊 Coins: {len(AppConfig.COINS)}")
    logger.info(f"🔄 Update every {AppConfig.UPDATE_INTERVAL//60} minutes")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
