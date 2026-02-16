"""
Crypto Tops & Bottoms Detector Bot - الإصدار المتقدم v4.0
تحسينات شاملة: Heiken Ashi, ZigZag pivots, مؤشرات متعددة, انحراف متقدم, تأكيد الإشارات, إشارات متدرجة
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

# إعدادات التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crypto_tops_bottoms_advanced.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ======================
# هياكل البيانات
# ======================
@dataclass
class CoinConfig:
    symbol: str
    name: str
    base_asset: str
    quote_asset: str
    enabled: bool = True

class SignalStrength(Enum):
    WEAK = "ضعيف"
    MODERATE = "متوسط"
    STRONG = "قوي"

@dataclass
class TopBottomSignal:
    coin_symbol: str
    coin_name: str
    signal_type: str  # "TOP" or "BOTTOM"
    strength: SignalStrength
    confidence: float  # 0-100
    price: float
    timestamp: datetime
    indicators: Dict[str, Any]
    message: str
    confirmed: bool = False  # للإشارات المؤكدة فقط

@dataclass
class PendingSignal:
    """إشارة محتملة في انتظار التأكيد"""
    coin_symbol: str
    coin_name: str
    signal_type: str
    confidence: float
    price: float
    first_seen: datetime
    indicators: Dict[str, Any]
    retest_count: int = 0

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
# إعدادات التطبيق (موسعة)
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

    # الإطارات الزمنية
    TIMEFRAME = '15m'
    HIGHER_TIMEFRAMES = ['1h', '4h']  # إطارات متعددة للتأكيد
    MAX_CANDLES = 500  # زيادة لتغذية المؤشرات
    MIN_CANDLES_REQUIRED = 100

    # إعدادات Heiken Ashi
    ENABLE_HEIKEN_ASHI = True

    # إعدادات Pivot بطريقة ZigZag
    PIVOT_MIN_ATR_MULTIPLE = 2.0      # المسافة الدنيا بين القمم/القيعان (مضاعف ATR)
    PIVOT_LOOKBACK = 30                # عدد الشموع للبحث عن القمم/القيعان

    # Fractal (يبقى كما هو للتوافق)
    FRACTAL_PERIOD = 2

    # عتبات الثقة (تعديل بعد التحسينات)
    TOP_CONFIDENCE_THRESHOLD = 65
    BOTTOM_CONFIDENCE_THRESHOLD = 65

    UPDATE_INTERVAL = 180  # ثواني

    # فلتر الإشعارات
    COOLDOWN_SECONDS = 300
    MIN_PRICE_MOVE_PERCENT = 0.8
    MIN_VOLATILITY_ATR_PERCENT = 0.4

    # إعدادات التأكيد
    CONFIRMATION_CANDLES = 1          # عدد الشموع للتأكيد
    PENDING_SIGNAL_TIMEOUT = 3600     # ثواني (ساعة واحدة)

    # مستويات الثقة للإشارات المتدرجة
    CONFIDENCE_LEVELS = {
        SignalStrength.WEAK: (50, 70),
        SignalStrength.MODERATE: (70, 85),
        SignalStrength.STRONG: (85, 100)
    }

    # أوزان المؤشرات (سيتم استخدامها في الدمج غير الخطي)
    INDICATOR_WEIGHTS = {
        'pivot': 0.25,
        'rsi_div': 0.20,
        'macd_div': 0.15,
        'stoch_div': 0.10,
        'fractal': 0.10,
        'volume': 0.10,
        'msb': 0.20,
        'bb': 0.10,
        'obv': 0.05,
        'mfi': 0.10,
        'choppiness': 0.05
    }

    # أنماط الشموع
    ENABLE_CANDLE_PATTERNS = True

# ======================
# إعدادات APIs
# ======================
class ExternalAPIConfig:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_tops_bottoms_v4')
    NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 2

# ======================
# Binance Client محدث
# ======================
class BinanceClient:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
            'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'rateLimit': 50,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
                'maxRetriesOnFailure': 3,
            }
        })
        self.session = requests.Session()
        self.last_request_time = {}
        self.min_request_interval = 1.5

    def _wait_for_rate_limit(self, symbol):
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
        try:
            self._wait_for_rate_limit(symbol)
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

    def fetch_ohlcv_with_fallback(self, symbol: str, timeframe: str = AppConfig.TIMEFRAME, limit: int = AppConfig.MAX_CANDLES) -> Optional[List]:
        data = self.fetch_ohlcv(symbol, timeframe, limit)
        if data:
            return data
        logger.info(f"Retrying {symbol} with smaller limit...")
        data = self.fetch_ohlcv(symbol, timeframe, min(limit, 200))
        if data:
            return data
        return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            self._wait_for_rate_limit(symbol)
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None

    def fetch_multiple_tickers(self, symbols: List[str]) -> Dict:
        try:
            tickers = self.exchange.fetch_tickers(symbols)
            return tickers
        except Exception as e:
            logger.error(f"Error fetching multiple tickers: {e}")
            return {}

# ======================
# المؤشرات الفنية المتقدمة
# ======================
class TechnicalIndicators:
    """مكتبة شاملة للمؤشرات الفنية"""

    @staticmethod
    def heiken_ashi(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """تحويل الشموع العادية إلى Heiken Ashi"""
        ha_close = [(o + h + l + c) / 4 for o, h, l, c in zip(opens, highs, lows, closes)]
        ha_open = [opens[0]]  # أول شمعة HA تفتح بسعر افتتاح الشمعة العادية
        for i in range(1, len(closes)):
            ha_open.append((ha_open[-1] + ha_close[i-1]) / 2)
        ha_high = [max(ha_open[i], highs[i], ha_close[i]) for i in range(len(closes))]
        ha_low = [min(ha_open[i], lows[i], ha_close[i]) for i in range(len(closes))]
        return ha_open, ha_high, ha_low, ha_close

    @staticmethod
    def sma(values: List[float], period: int) -> List[Optional[float]]:
        result = [None] * len(values)
        for i in range(period - 1, len(values)):
            result[i] = sum(values[i - period + 1:i + 1]) / period
        return result

    @staticmethod
    def ema(values: List[float], period: int) -> List[float]:
        if not values:
            return []
        k = 2 / (period + 1)
        ema_values = [values[0]]
        for i in range(1, len(values)):
            ema_values.append(values[i] * k + ema_values[-1] * (1 - k))
        return ema_values

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
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
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """MACD line, signal line, histogram"""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = [None] * len(prices)
        for i in range(len(prices)):
            if ema_fast[i] is not None and ema_slow[i] is not None:
                macd_line[i] = ema_fast[i] - ema_slow[i]
        signal_line = TechnicalIndicators.ema([x if x is not None else 0 for x in macd_line], signal)
        # تصحيح signal_line للحفاظ على القيم None في البداية
        signal_line = [None if macd_line[i] is None else signal_line[i] for i in range(len(prices))]
        histogram = [None] * len(prices)
        for i in range(len(prices)):
            if macd_line[i] is not None and signal_line[i] is not None:
                histogram[i] = macd_line[i] - signal_line[i]
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """Stochastic %K and %D"""
        k_values = [None] * len(closes)
        for i in range(k_period - 1, len(closes)):
            high_max = max(highs[i - k_period + 1:i + 1])
            low_min = min(lows[i - k_period + 1:i + 1])
            if high_max - low_min != 0:
                k = 100 * (closes[i] - low_min) / (high_max - low_min)
                k_values[i] = k
            else:
                k_values[i] = 50
        d_values = TechnicalIndicators.sma([k if k is not None else 0 for k in k_values], d_period)
        d_values = [None if k_values[i] is None else d_values[i] for i in range(len(closes))]
        return k_values, d_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
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
    def obv(closes: List[float], volumes: List[float]) -> List[float]:
        """On-Balance Volume"""
        obv_values = [0.0] * len(closes)
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv_values[i] = obv_values[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv_values[i] = obv_values[i-1] - volumes[i]
            else:
                obv_values[i] = obv_values[i-1]
        return obv_values

    @staticmethod
    def mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 14) -> List[Optional[float]]:
        """Money Flow Index"""
        length = len(closes)
        if length < period + 1:
            return [None] * length
        typical_price = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        money_flow = [tp * v for tp, v in zip(typical_price, volumes)]
        mfi_values = [None] * length
        for i in range(period, length):
            pos_flow = 0
            neg_flow = 0
            for j in range(i - period, i):
                if typical_price[j] > typical_price[j-1]:
                    pos_flow += money_flow[j]
                elif typical_price[j] < typical_price[j-1]:
                    neg_flow += money_flow[j]
            if neg_flow != 0:
                mfi = 100 - (100 / (1 + pos_flow / neg_flow))
            else:
                mfi = 100
            mfi_values[i] = mfi
        return mfi_values

    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        """Middle, Upper, Lower bands"""
        middle = TechnicalIndicators.sma(prices, period)
        upper = [None] * len(prices)
        lower = [None] * len(prices)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            std = np.std(window)
            upper[i] = middle[i] + std_dev * std
            lower[i] = middle[i] - std_dev * std
        return middle, upper, lower

    @staticmethod
    def choppiness_index(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Choppiness Index - يقيس تذبذب السوق (قيم عالية = سوق متذبذب)"""
        length = len(closes)
        if length < period + 1:
            return [None] * length
        atr_sum = [0.0] * length
        for i in range(period - 1, length):
            atr_sum[i] = sum(TechnicalIndicators.atr(highs[i-period+1:i+1], lows[i-period+1:i+1], closes[i-period+1:i+1], period)[-period:])
        high_max = [0.0] * length
        low_min = [0.0] * length
        for i in range(period - 1, length):
            high_max[i] = max(highs[i-period+1:i+1])
            low_min[i] = min(lows[i-period+1:i+1])
        choppiness = [None] * length
        for i in range(period - 1, length):
            if high_max[i] - low_min[i] != 0:
                choppiness[i] = 100 * math.log10(atr_sum[i] / (high_max[i] - low_min[i])) / math.log10(period)
            else:
                choppiness[i] = 0
        return choppiness

    @staticmethod
    def zigzag_pivots(highs: List[float], lows: List[float], atr: List[Optional[float]], min_atr_multiple: float = 2.0, lookback: int = 30) -> Tuple[List[bool], List[bool]]:
        """كشف القمم والقيعان بطريقة ZigZag مع مسافة ATR دنيا"""
        length = len(highs)
        pivot_highs = [False] * length
        pivot_lows = [False] * length
        if length < lookback:
            return pivot_highs, pivot_lows

        last_pivot_idx = None
        last_pivot_type = None  # 'high' or 'low'
        last_pivot_price = None

        for i in range(lookback, length - lookback):
            # البحث عن قمة محلية
            window_highs = highs[i - lookback:i + lookback + 1]
            if highs[i] == max(window_highs) and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                # تأكد من المسافة عن آخر قمة
                if last_pivot_type == 'high' and last_pivot_idx is not None:
                    if abs(highs[i] - last_pivot_price) < atr[i] * min_atr_multiple if atr[i] else 0:
                        continue
                pivot_highs[i] = True
                last_pivot_idx = i
                last_pivot_type = 'high'
                last_pivot_price = highs[i]

            # البحث عن قاع محلي
            window_lows = lows[i - lookback:i + lookback + 1]
            if lows[i] == min(window_lows) and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                if last_pivot_type == 'low' and last_pivot_idx is not None:
                    if abs(lows[i] - last_pivot_price) < atr[i] * min_atr_multiple if atr[i] else 0:
                        continue
                pivot_lows[i] = True
                last_pivot_idx = i
                last_pivot_type = 'low'
                last_pivot_price = lows[i]

        return pivot_highs, pivot_lows

    @staticmethod
    def detect_divergence(price: List[float], oscillator: List[Optional[float]], window: int = 40, hidden: bool = False) -> Dict[str, bool]:
        """كشف الانحراف العادي والخفي"""
        recent_price = price[-window:]
        recent_osc = oscillator[-window:]

        valid_indices = [i for i, v in enumerate(recent_osc) if v is not None]
        if len(valid_indices) < 10:
            return {'bullish': False, 'bearish': False}

        prices_valid = [recent_price[i] for i in valid_indices]
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

        bullish = False
        bearish = False

        if not hidden:
            # انحراف عادي
            if len(price_troughs) >= 2 and len(osc_troughs) >= 2:
                last_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]
                last_osc_trough = osc_troughs[-1]
                prev_osc_trough = osc_troughs[-2]
                if last_price_trough[1] < prev_price_trough[1] and last_osc_trough[1] > prev_osc_trough[1]:
                    bullish = True
            if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
                last_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]
                last_osc_peak = osc_peaks[-1]
                prev_osc_peak = osc_peaks[-2]
                if last_price_peak[1] > prev_price_peak[1] and last_osc_peak[1] < prev_osc_peak[1]:
                    bearish = True
        else:
            # انحراف خفي
            if len(price_troughs) >= 2 and len(osc_troughs) >= 2:
                last_price_trough = price_troughs[-1]
                prev_price_trough = price_troughs[-2]
                last_osc_trough = osc_troughs[-1]
                prev_osc_trough = osc_troughs[-2]
                if last_price_trough[1] > prev_price_trough[1] and last_osc_trough[1] < prev_osc_trough[1]:
                    bullish = True  # انحراف صاعد خفي
            if len(price_peaks) >= 2 and len(osc_peaks) >= 2:
                last_price_peak = price_peaks[-1]
                prev_price_peak = price_peaks[-2]
                last_osc_peak = osc_peaks[-1]
                prev_osc_peak = osc_peaks[-2]
                if last_price_peak[1] < prev_price_peak[1] and last_osc_peak[1] > prev_osc_peak[1]:
                    bearish = True  # انحراف هابط خفي

        return {'bullish': bullish, 'bearish': bearish}

    @staticmethod
    def fractal(highs: List[float], lows: List[float], period: int = 2) -> Tuple[List[bool], List[bool]]:
        length = len(highs)
        fractal_up = [False] * length
        fractal_down = [False] * length
        for i in range(period, length - period):
            if all(highs[i] > highs[i - j] for j in range(1, period + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, period + 1)):
                fractal_up[i] = True
            if all(lows[i] < lows[i - j] for j in range(1, period + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, period + 1)):
                fractal_down[i] = True
        return fractal_up, fractal_down

    @staticmethod
    def market_structure(highs: List[float], lows: List[float], closes: List[float],
                         pivot_highs: List[bool], pivot_lows: List[bool]) -> Dict[str, Any]:
        """تحليل هيكل السوق باستخدام نقاط الارتكاز"""
        high_indices = [i for i, v in enumerate(pivot_highs) if v]
        low_indices = [i for i, v in enumerate(pivot_lows) if v]

        if len(high_indices) < 2 or len(low_indices) < 2:
            return {'trend': 'unknown', 'bos_up': False, 'bos_down': False, 'choch_up': False, 'choch_down': False}

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

        bos_up = uptrend and closes[-1] > highs[last_high_idx]
        bos_down = downtrend and closes[-1] < lows[last_low_idx]
        choch_up = downtrend and closes[-1] > highs[last_high_idx]
        choch_down = uptrend and closes[-1] < lows[last_low_idx]

        trend = 'uptrend' if uptrend else 'downtrend' if downtrend else 'ranging'
        return {
            'trend': trend,
            'bos_up': bos_up,
            'bos_down': bos_down,
            'choch_up': choch_up,
            'choch_down': choch_down
        }

    @staticmethod
    def candle_patterns(open_prices: List[float], high: List[float], low: List[float], close: List[float]) -> Dict[str, bool]:
        """أنماط الشموع الأساسية"""
        if len(close) < 5:
            return {}
        o1, o2, o3 = open_prices[-1], open_prices[-2], open_prices[-3]
        h1, h2, h3 = high[-1], high[-2], high[-3]
        l1, l2, l3 = low[-1], low[-2], low[-3]
        c1, c2, c3 = close[-1], close[-2], close[-3]

        patterns = {}
        body1 = abs(c1 - o1)
        upper_shadow1 = h1 - max(c1, o1)
        lower_shadow1 = min(c1, o1) - l1

        patterns['shooting_star'] = (upper_shadow1 > 2 * body1 and lower_shadow1 < 0.2 * body1 and c1 < o1)
        patterns['hammer'] = (lower_shadow1 > 2 * body1 and upper_shadow1 < 0.2 * body1 and c1 > o1)
        patterns['engulfing_bear'] = (c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2)
        patterns['engulfing_bull'] = (c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2)
        patterns['doji'] = (abs(c1 - o1) <= (h1 - l1) * 0.1)
        return patterns

# ======================
# مدير الإشعارات
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}
        self.last_notification_price = {}
        self.cooldown_base = AppConfig.COOLDOWN_SECONDS
        self.cooldown_multiplier = {}

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
        if not signal.confirmed:
            return None  # نرسل فقط الإشارات المؤكدة

        key = (signal.coin_symbol, signal.signal_type)
        if not self.should_send(signal.coin_symbol, signal.signal_type, signal.confidence, signal.price):
            return None

        title = f"{signal.signal_type} {signal.strength.value}: {signal.coin_name}"
        message = (
            f"{title}\n"
            f"الثقة: {signal.confidence:.1f}%\n"
            f"السعر: ${signal.price:,.2f}\n"
            f"الوقت: {signal.timestamp.strftime('%H:%M')}\n"
            f"المؤشرات: {json.dumps(signal.indicators, default=str)}"
        )

        tags = "arrow_up" if signal.signal_type == "TOP" else "arrow_down"
        priority = "5" if signal.strength == SignalStrength.STRONG else "4" if signal.strength == SignalStrength.MODERATE else "3"

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
# المدير الرئيسي للكشف (نسخة متطورة)
# ======================
class TopBottomDetector:
    def __init__(self):
        self.detections: List[TopBottomSignal] = []  # الإشارات المؤكدة فقط
        self.pending_signals: Dict[str, PendingSignal] = {}  # إشارات في انتظار التأكيد
        self.last_update: Optional[datetime] = None
        self.last_coins_update: Optional[datetime] = None
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()
        self.cached_higher_tf_data: Dict[str, Any] = {}

    def update_coins_list(self):
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
            logger.info(f"🔄 فحص {len(AppConfig.COINS)} عملة لاكتشاف القمم والقيعان (نسخة متطورة)...")

            success_count = 0
            failed_coins = []

            symbols = [coin.symbol for coin in AppConfig.COINS if coin.enabled]
            all_tickers = self.binance.fetch_multiple_tickers(symbols)

            for coin in AppConfig.COINS:
                if not coin.enabled:
                    continue
                try:
                    ticker = all_tickers.get(coin.symbol)
                    signals = self._scan_coin_advanced(coin, ticker)
                    for signal in signals:
                        if signal.confirmed:
                            self.detections.append(signal)
                            self.notification_manager.create_notification(signal)
                            success_count += 1
                        else:
                            # تخزين الإشارة المحتملة للتأكيد لاحقاً
                            self._store_pending_signal(coin, signal)
                except Exception as e:
                    logger.error(f"خطأ في {coin.symbol}: {e}", exc_info=True)
                    failed_coins.append(coin)

            # تنظيف الإشارات المعلقة القديمة
            self._cleanup_pending_signals()

            self.last_update = datetime.now()
            if len(self.detections) > 100:
                self.detections = self.detections[-100:]

            logger.info(f"✅ تم اكتشاف {success_count} إشارة مؤكدة")
            return success_count > 0

    def _store_pending_signal(self, coin: CoinConfig, signal: TopBottomSignal):
        """تخزين إشارة غير مؤكدة"""
        key = f"{coin.symbol}_{signal.signal_type}"
        existing = self.pending_signals.get(key)
        if existing:
            # تحديث إذا كانت الثقة أعلى
            if signal.confidence > existing.confidence:
                existing.confidence = signal.confidence
                existing.indicators = signal.indicators
                existing.retest_count += 1
        else:
            self.pending_signals[key] = PendingSignal(
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                price=signal.price,
                first_seen=datetime.now(),
                indicators=signal.indicators
            )

    def _cleanup_pending_signals(self):
        """حذف الإشارات المعلقة القديمة"""
        now = datetime.now()
        expired_keys = []
        for key, pending in self.pending_signals.items():
            if (now - pending.first_seen).total_seconds() > AppConfig.PENDING_SIGNAL_TIMEOUT:
                expired_keys.append(key)
        for key in expired_keys:
            del self.pending_signals[key]

    def _confirm_pending_signals(self, coin: CoinConfig, new_signal: Optional[TopBottomSignal]):
        """التحقق من تأكيد الإشارات المعلقة بناءً على الشمعة الجديدة"""
        for key in list(self.pending_signals.keys()):
            if key.startswith(coin.symbol):
                pending = self.pending_signals[key]
                # إذا كانت الإشارة المعلقة لا تزال صالحة والشروط الجديدة تدعمها
                if new_signal and new_signal.signal_type == pending.signal_type and new_signal.confidence >= pending.confidence * 0.8:
                    # تأكيد الإشارة
                    confirmed_signal = TopBottomSignal(
                        coin_symbol=pending.coin_symbol,
                        coin_name=pending.coin_name,
                        signal_type=pending.signal_type,
                        strength=self._confidence_to_strength(pending.confidence),
                        confidence=pending.confidence,
                        price=pending.price,
                        timestamp=datetime.now(),
                        indicators=pending.indicators,
                        message=f"تأكيد {pending.signal_type} بعد {pending.retest_count+1} شمعة",
                        confirmed=True
                    )
                    self.detections.append(confirmed_signal)
                    self.notification_manager.create_notification(confirmed_signal)
                    logger.info(f"✅ تم تأكيد إشارة {pending.signal_type} لـ {pending.coin_symbol}")
                    del self.pending_signals[key]
                else:
                    # إذا لم يتم التأكيد، نزيد عدد المحاولات
                    pending.retest_count += 1
                    if pending.retest_count > AppConfig.CONFIRMATION_CANDLES:
                        # فشل التأكيد بعد العدد المطلوب من الشموع
                        logger.debug(f"❌ إشارة {pending.signal_type} لـ {pending.coin_symbol} لم تتأكد بعد {AppConfig.CONFIRMATION_CANDLES} شموع")
                        del self.pending_signals[key]

    def _confidence_to_strength(self, confidence: float) -> SignalStrength:
        for strength, (low, high) in AppConfig.CONFIDENCE_LEVELS.items():
            if low <= confidence < high:
                return strength
        return SignalStrength.MODERATE

    def _scan_coin_advanced(self, coin: CoinConfig, pre_fetched_ticker: Optional[Dict] = None) -> List[TopBottomSignal]:
        """مسح عملة واحدة وإرجاع قائمة بالإشارات (مؤكدة وغير مؤكدة)"""
        signals = []

        # جلب البيانات للإطار الرئيسي
        ohlcv = self.binance.fetch_ohlcv_with_fallback(coin.symbol, AppConfig.TIMEFRAME, AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < AppConfig.MIN_CANDLES_REQUIRED:
            return signals

        # استخراج الشموع
        opens = [c[1] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        # تطبيق Heiken Ashi إذا كان مفعلاً
        if AppConfig.ENABLE_HEIKEN_ASHI:
            ha_open, ha_high, ha_low, ha_close = TechnicalIndicators.heiken_ashi(opens, highs, lows, closes)
            # استخدام بيانات Heiken Ashi للمؤشرات التي تعتمد على السعر
            price_for_indicators = ha_close
            highs_for_indicators = ha_high
            lows_for_indicators = ha_low
            closes_for_indicators = ha_close
        else:
            price_for_indicators = closes
            highs_for_indicators = highs
            lows_for_indicators = lows
            closes_for_indicators = closes

        # المؤشرات الأساسية
        rsi = TechnicalIndicators.rsi(price_for_indicators, 14)
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)  # ATR على الشموع الأصلية أفضل
        macd_line, signal_line, macd_hist = TechnicalIndicators.macd(price_for_indicators)
        stoch_k, stoch_d = TechnicalIndicators.stochastic(highs_for_indicators, lows_for_indicators, closes_for_indicators)
        obv = TechnicalIndicators.obv(closes, volumes)
        mfi = TechnicalIndicators.mfi(highs, lows, closes, volumes)
        bb_mid, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(price_for_indicators)
        choppiness = TechnicalIndicators.choppiness_index(highs, lows, closes)

        # نقاط الارتكاز ZigZag
        pivot_highs, pivot_lows = TechnicalIndicators.zigzag_pivots(
            highs, lows, atr,
            min_atr_multiple=AppConfig.PIVOT_MIN_ATR_MULTIPLE,
            lookback=AppConfig.PIVOT_LOOKBACK
        )

        # كشف الانحراف على RSI, MACD, Stochastic
        rsi_div = TechnicalIndicators.detect_divergence(price_for_indicators, rsi)
        macd_div = TechnicalIndicators.detect_divergence(price_for_indicators, macd_line)
        stoch_div = TechnicalIndicators.detect_divergence(price_for_indicators, stoch_k)

        # Fractal
        fractal_up, fractal_down = TechnicalIndicators.fractal(highs, lows, AppConfig.FRACTAL_PERIOD)

        # هيكل السوق
        ms = TechnicalIndicators.market_structure(highs, lows, closes, pivot_highs, pivot_lows)

        # أنماط الشموع
        patterns = TechnicalIndicators.candle_patterns(opens, highs, lows, closes) if AppConfig.ENABLE_CANDLE_PATTERNS else {}

        # تحليل الإطار الزمني الأعلى
        htf_conf = self._check_multiple_timeframes(coin.symbol)

        # التايكر والسعر الحالي
        if pre_fetched_ticker:
            ticker = pre_fetched_ticker
        else:
            ticker = self.binance.fetch_ticker(coin.symbol)
        if not ticker or ticker.get('last') is None:
            return signals
        current_price = ticker['last']

        # فلتر التقلب
        atr_percent = (atr[-1] / current_price * 100) if atr[-1] else 0
        if atr_percent < AppConfig.MIN_VOLATILITY_ATR_PERCENT:
            return signals

        # حساب الثقة للقمة والقاع
        top_confidence = self._calculate_confidence(
            'TOP', current_price, closes_for_indicators,
            rsi, macd_line, stoch_k, obv, mfi, bb_upper, bb_lower, choppiness,
            pivot_highs, pivot_lows,
            fractal_up, fractal_down,
            rsi_div, macd_div, stoch_div,
            ms, htf_conf, patterns, volumes
        )
        bottom_confidence = self._calculate_confidence(
            'BOTTOM', current_price, closes_for_indicators,
            rsi, macd_line, stoch_k, obv, mfi, bb_upper, bb_lower, choppiness,
            pivot_highs, pivot_lows,
            fractal_up, fractal_down,
            rsi_div, macd_div, stoch_div,
            ms, htf_conf, patterns, volumes
        )

        # إنشاء الإشارات
        if top_confidence >= AppConfig.TOP_CONFIDENCE_THRESHOLD:
            signal = TopBottomSignal(
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                signal_type="TOP",
                strength=self._confidence_to_strength(top_confidence),
                confidence=top_confidence,
                price=current_price,
                timestamp=datetime.now(),
                indicators={
                    'rsi': round(rsi[-1], 1) if rsi[-1] else None,
                    'atr_percent': round(atr_percent, 2),
                    'choppiness': round(choppiness[-1], 1) if choppiness[-1] else None,
                    'div_rsi_bear': rsi_div.get('bearish', False),
                    'div_macd_bear': macd_div.get('bearish', False),
                    'div_stoch_bear': stoch_div.get('bearish', False),
                    'pivot_high': pivot_highs[-5:].count(True) > 0,
                    'fractal_up': fractal_up[-5:].count(True) > 0,
                    'ms_trend': ms['trend'],
                    'ms_choch_down': ms.get('choch_down', False),
                    'htf_conf': htf_conf,
                    'patterns': patterns
                },
                message=f"قمة محتملة بثقة {top_confidence:.1f}%",
                confirmed=False  # تحتاج تأكيد
            )
            signals.append(signal)

        if bottom_confidence >= AppConfig.BOTTOM_CONFIDENCE_THRESHOLD:
            signal = TopBottomSignal(
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                signal_type="BOTTOM",
                strength=self._confidence_to_strength(bottom_confidence),
                confidence=bottom_confidence,
                price=current_price,
                timestamp=datetime.now(),
                indicators={
                    'rsi': round(rsi[-1], 1) if rsi[-1] else None,
                    'atr_percent': round(atr_percent, 2),
                    'choppiness': round(choppiness[-1], 1) if choppiness[-1] else None,
                    'div_rsi_bull': rsi_div.get('bullish', False),
                    'div_macd_bull': macd_div.get('bullish', False),
                    'div_stoch_bull': stoch_div.get('bullish', False),
                    'pivot_low': pivot_lows[-5:].count(True) > 0,
                    'fractal_down': fractal_down[-5:].count(True) > 0,
                    'ms_trend': ms['trend'],
                    'ms_choch_up': ms.get('choch_up', False),
                    'htf_conf': htf_conf,
                    'patterns': patterns
                },
                message=f"قاع محتمل بثقة {bottom_confidence:.1f}%",
                confirmed=False
            )
            signals.append(signal)

        # التحقق من الإشارات المعلقة لهذه العملة
        self._confirm_pending_signals(coin, signals[0] if signals else None)

        return signals

    def _check_multiple_timeframes(self, symbol: str) -> Dict[str, Any]:
        """جلب وتحليل عدة إطارات زمنية أعلى"""
        result = {'trend': 'unknown', 'trend_up': False, 'trend_down': False}
        for tf in AppConfig.HIGHER_TIMEFRAMES:
            ohlcv = self.binance.fetch_ohlcv(symbol, tf, 50)
            if ohlcv and len(ohlcv) > 20:
                closes = [c[4] for c in ohlcv]
                sma20 = TechnicalIndicators.sma(closes, 20)
                current_sma = sma20[-1] if sma20[-1] else closes[-1]
                trend_up = closes[-1] > current_sma
                trend_down = closes[-1] < current_sma
                result[f'trend_{tf}'] = 'up' if trend_up else 'down' if trend_down else 'sideways'
                result[f'trend_up_{tf}'] = trend_up
                result[f'trend_down_{tf}'] = trend_down
                # تبسيط: إذا كان أي إطار أعلى متوافق، نعتبره تأكيداً
                if trend_up:
                    result['trend_up'] = True
                if trend_down:
                    result['trend_down'] = True
        return result

    def _calculate_confidence(self, signal_type: str, price: float, closes: List[float],
                              rsi: List[Optional[float]], macd: List[Optional[float]], stoch: List[Optional[float]],
                              obv: List[float], mfi: List[Optional[float]],
                              bb_upper: List[Optional[float]], bb_lower: List[Optional[float]],
                              choppiness: List[Optional[float]],
                              pivot_highs: List[bool], pivot_lows: List[bool],
                              fractal_up: List[bool], fractal_down: List[bool],
                              rsi_div: Dict[str, bool], macd_div: Dict[str, bool], stoch_div: Dict[str, bool],
                              ms: Dict[str, Any], htf_conf: Dict[str, Any],
                              patterns: Dict[str, bool], volumes: List[float]) -> float:
        """حساب الثقة باستخدام الدمج غير الخطي (معادلة الاحتمال المركب)"""
        weights = AppConfig.INDICATOR_WEIGHTS
        prob = 0.0
        total_weight = 0.0

        # دالة مساعدة لإضافة مؤشر مع وزنه
        def add_indicator(condition: bool, weight_key: str):
            nonlocal prob, total_weight
            if condition:
                prob += weights.get(weight_key, 0) * 100
                total_weight += weights.get(weight_key, 0)

        # 1. Pivot قريب
        if signal_type == 'TOP' and any(pivot_highs[-5:]):
            add_indicator(True, 'pivot')
        if signal_type == 'BOTTOM' and any(pivot_lows[-5:]):
            add_indicator(True, 'pivot')

        # 2. RSI Divergence
        if signal_type == 'TOP' and rsi_div.get('bearish', False):
            add_indicator(True, 'rsi_div')
        if signal_type == 'BOTTOM' and rsi_div.get('bullish', False):
            add_indicator(True, 'rsi_div')

        # 3. MACD Divergence
        if signal_type == 'TOP' and macd_div.get('bearish', False):
            add_indicator(True, 'macd_div')
        if signal_type == 'BOTTOM' and macd_div.get('bullish', False):
            add_indicator(True, 'macd_div')

        # 4. Stochastic Divergence
        if signal_type == 'TOP' and stoch_div.get('bearish', False):
            add_indicator(True, 'stoch_div')
        if signal_type == 'BOTTOM' and stoch_div.get('bullish', False):
            add_indicator(True, 'stoch_div')

        # 5. Fractal
        if signal_type == 'TOP' and any(fractal_up[-5:]):
            add_indicator(True, 'fractal')
        if signal_type == 'BOTTOM' and any(fractal_down[-5:]):
            add_indicator(True, 'fractal')

        # 6. Volume spike
        avg_vol = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else 0
        volume_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
        if volume_spike:
            add_indicator(True, 'volume')

        # 7. Market Structure Break
        if signal_type == 'TOP' and (ms.get('bos_down', False) or ms.get('choch_down', False)):
            add_indicator(True, 'msb')
        if signal_type == 'BOTTOM' and (ms.get('bos_up', False) or ms.get('choch_up', False)):
            add_indicator(True, 'msb')

        # 8. Bollinger Bands
        if signal_type == 'TOP' and bb_upper[-1] and price >= bb_upper[-1]:
            add_indicator(True, 'bb')
        if signal_type == 'BOTTOM' and bb_lower[-1] and price <= bb_lower[-1]:
            add_indicator(True, 'bb')

        # 9. OBV confirmation
        obv_trend = obv[-1] > obv[-5] if len(obv) >= 5 else False
        if signal_type == 'TOP' and not obv_trend:  # OBV يضعف
            add_indicator(True, 'obv')
        if signal_type == 'BOTTOM' and obv_trend:
            add_indicator(True, 'obv')

        # 10. MFI
        if signal_type == 'TOP' and mfi[-1] and mfi[-1] > 80:
            add_indicator(True, 'mfi')
        if signal_type == 'BOTTOM' and mfi[-1] and mfi[-1] < 20:
            add_indicator(True, 'mfi')

        # 11. Choppiness (if low, better for trends)
        if choppiness[-1] and choppiness[-1] < 40:
            add_indicator(True, 'choppiness')

        # 12. Candle patterns
        if signal_type == 'TOP' and (patterns.get('shooting_star') or patterns.get('engulfing_bear')):
            add_indicator(True, 'fractal')  # نستخدم وزن fractal كبديل
        if signal_type == 'BOTTOM' and (patterns.get('hammer') or patterns.get('engulfing_bull')):
            add_indicator(True, 'fractal')

        # 13. Higher timeframe confirmation
        if signal_type == 'TOP' and htf_conf.get('trend_down', False):
            add_indicator(True, 'pivot')  # نضيف وزناً إضافياً
        if signal_type == 'BOTTOM' and htf_conf.get('trend_up', False):
            add_indicator(True, 'pivot')

        # إذا لم يتحقق أي شرط، نعيد 0
        if total_weight == 0:
            return 0.0

        # المعادلة غير الخطية: prob هي مجموع الأوزان * 100، لكنها قد تتجاوز 100
        # نقوم بتطبيع باستخدام صيغة تشبه الاحتمال المركب
        # confidence = 100 * (1 - exp(-prob/50)) على سبيل المثال
        # أو ببساطة نحدها بـ 100
        confidence = min(prob, 100.0)
        return confidence

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
            'pending_signals': len(self.pending_signals),
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
            logger.error(f"خطأ في التحديث: {e}")
            time.sleep(60)

# ======================
# تطبيق Flask
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-tops-bottoms-advanced-v4-secret')
detector = TopBottomDetector()
start_time = time.time()

updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# تحديث أولي
detector.update_all()

# ======================
# المسارات
# ======================
@app.route('/')
def index():
    detections = detector.get_recent_detections(10)
    stats = detector.get_stats()
    return render_template('index_tops_bottoms_v4.html', detections=detections, stats=stats)

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
        'message': 'تم الفحص',
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
    msg = "اختبار - بوت الكشف المتقدم عن القمم والقيعان يعمل"
    success = detector.notification_manager.send_ntfy(msg, "اختبار", "3", "test_tube")
    return jsonify({'success': success})

def send_startup_notification():
    try:
        msg = (
            f"تم تشغيل البوت المتقدم v4.0\n"
            f"يتتبع {len(AppConfig.COINS)} عملة\n"
            f"فترة التحديث: {AppConfig.UPDATE_INTERVAL//60} دقيقة\n"
            f"عتبة الثقة: {AppConfig.TOP_CONFIDENCE_THRESHOLD}%"
        )
        detector.notification_manager.send_ntfy(msg, "بدء التشغيل", "3", "rocket")
    except Exception as e:
        logger.error(f"خطأ في إشعار البدء: {e}")

def delayed_startup():
    time.sleep(5)
    send_startup_notification()

threading.Thread(target=delayed_startup, daemon=True).start()

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 بوت اكتشاف القمم والقيعان - الإصدار المتقدم v4.0")
    logger.info(f"📊 العملات: {len(AppConfig.COINS)}")
    logger.info(f"🔄 تحديث كل {AppConfig.UPDATE_INTERVAL//60} دقيقة")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port)
