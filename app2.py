"""
Crypto Tops & Bottoms Detector Bot - النسخة المتقدمة (مصححة ومحسنة)
إصدار 2.1 - إصلاح شامل للأخطاء وتحسين الدقة
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
# إعدادات التطبيق
# ======================
class AppConfig:
    COINS = [
        CoinConfig("BTC/USDT", "Bitcoin"),
        CoinConfig("ETH/USDT", "Ethereum"),
        CoinConfig("BNB/USDT", "Binance Coin"),
        CoinConfig("SOL/USDT", "Solana"),
        CoinConfig("XRP/USDT", "Ripple"),
    ]

    TIMEFRAME = '15m'
    HIGHER_TIMEFRAME = '1h'
    MAX_CANDLES = 300

    # إعدادات Pivot
    PIVOT_LEFT = 5
    PIVOT_RIGHT = 5
    MIN_PIVOT_DISTANCE_ATR = 1.5      # مضاعف ATR للمسافة بين القمم/القيعان

    # Fractal
    FRACTAL_PERIOD = 2                 # 2 شمعة على كل جانب => 5 شموع

    # عتبات الثقة
    TOP_CONFIDENCE_THRESHOLD = 60
    BOTTOM_CONFIDENCE_THRESHOLD = 60

    UPDATE_INTERVAL = 120

    # فلتر الإشعارات
    COOLDOWN_SECONDS = 300
    MIN_PRICE_MOVE_PERCENT = 0.8
    MIN_VOLATILITY_ATR_PERCENT = 0.5   # إذا كانت ATR% أقل من هذا، السوق هادئ

    # أوزان المؤشرات (تختلف حسب نوع السوق)
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
# Binance Client مع إعادة محاولة ذكية
# ======================
class BinanceClient:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
            'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.session = requests.Session()

    @backoff.on_exception(
        backoff.expo,
        (ccxt.RateLimitExceeded, ccxt.NetworkError, requests.exceptions.RequestException),
        max_tries=3,
        max_time=30
    )
    def fetch_ohlcv(self, symbol: str, timeframe: str = AppConfig.TIMEFRAME, limit: int = AppConfig.MAX_CANDLES) -> Optional[List]:
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Binance OHLCV error {symbol} {timeframe}: {e}")
            return None

    @backoff.on_exception(
        backoff.expo,
        (ccxt.RateLimitExceeded, ccxt.NetworkError, requests.exceptions.RequestException),
        max_tries=3,
        max_time=30
    )
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Binance ticker error {symbol}: {e}")
            return None

# ======================
# المؤشرات الفنية المحسوبة يدوياً (بدون مكتبات إضافية) - مصححة
# ======================
class TechnicalIndicators:
    """حساب المؤشرات باستخدام عمليات أساسية على قوائم - نسخة مصححة"""

    @staticmethod
    def sma(values: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        result = [None] * len(values)
        for i in range(period - 1, len(values)):
            result[i] = sum(values[i - period + 1:i + 1]) / period
        return result

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """Relative Strength Index - مصحح"""
        if len(prices) < period + 1:
            return [None] * len(prices)

        # حساب التغيرات
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

        # متوسط المكاسب والخسائر الأولي
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = [None] * period  # أول period قيمة غير متاحة

        for i in range(period, len(prices)):
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_values.append(rsi)

            # تحديث المتوسطات للفترة التالية
            if i < len(prices) - 1:
                gain = gains[i]  # لاحظ أن deltas[i] يقابل الشمعة i+1
                loss = losses[i]
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period

        return rsi_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average True Range - مصحح بالكامل"""
        length = len(closes)
        if length < period + 1:
            return [None] * length

        # حساب True Range لكل شمعة (بدءاً من الشمعة 1)
        tr = [0.0] * length
        tr[0] = highs[0] - lows[0]  # أول شمعة: فقط high-low
        for i in range(1, length):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)

        # حساب ATR باستخدام Wilder's smoothing
        atr_values = [None] * length
        atr_values[period - 1] = sum(tr[:period]) / period

        for i in range(period, length):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period

        return atr_values

    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average Directional Index - مصحح بالكامل"""
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

        # تجانس TR, +DM, -DM باستخدام Wilder's smoothing
        smoothed_tr = [None] * length
        smoothed_plus_dm = [None] * length
        smoothed_minus_dm = [None] * length

        # أول قيمة متجانسة هي متوسط الفترة الأولى
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

        # حساب ADX كمتوسط DX خلال الفترة
        adx = [None] * length
        # نحتاج أول period من DX لنبدأ ADX
        dx_values = [dx[i] for i in range(length) if dx[i] is not None]
        if len(dx_values) >= period:
            # أول ADX في المؤشر (period-1 + period) لأننا نحتاج period من DX بعد أن يصبح DX متاحاً
            start_idx = (period - 1) + period
            if start_idx < length:
                adx[start_idx] = sum(dx_values[:period]) / period
                for i in range(start_idx + 1, length):
                    if dx[i] is not None:
                        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

        return adx

    @staticmethod
    def pivot_points(highs: List[float], lows: List[float], left: int = 5, right: int = 5) -> Tuple[List[bool], List[bool]]:
        """تحديد القمم والقيعان المحلية بناءً على left/right"""
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
        # نأخذ آخر window شمعة مع تجاهل القيم الخالية
        recent_prices = prices[-window:]
        recent_osc = oscillator[-window:]

        # إزالة القيم الخالية من المذبذب
        valid_indices = [i for i, v in enumerate(recent_osc) if v is not None]
        if len(valid_indices) < 10:  # نحتاج عدد كافٍ من النقاط
            return {'bullish': False, 'bearish': False}

        prices_valid = [recent_prices[i] for i in valid_indices]
        osc_valid = [recent_osc[i] for i in valid_indices]

        # البحث عن القمم المحلية في السعر والمذبذب
        # نستخدم نافذة صغيرة للبحث عن القمم المحلية
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
            # آخر قاعين
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
        """تحليل هيكل السوق: تحديد الاتجاه، كسر الهيكل (BOS)، تغير الطابع (CHoCH) - محسّن"""
        length = len(highs)

        # الحصول على آخر 5 قمم وقيعان صالحة
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

        # آخر قمة وقاع
        last_high_idx = high_indices[-1]
        last_low_idx = low_indices[-1]

        # القمة والقبل الأخيرة
        prev_high_idx = high_indices[-2] if len(high_indices) >= 2 else None
        prev_low_idx = low_indices[-2] if len(low_indices) >= 2 else None

        # تحديد الاتجاه الأساسي بناءً على سلاسل القمم والقيعان
        uptrend = False
        downtrend = False

        if prev_high_idx is not None and prev_low_idx is not None:
            # اتجاه صاعد: قمم أعلى وقيعان أعلى
            if highs[last_high_idx] > highs[prev_high_idx] and lows[last_low_idx] > lows[prev_low_idx]:
                uptrend = True
            # اتجاه هابط: قمم أدنى وقيعان أدنى
            elif highs[last_high_idx] < highs[prev_high_idx] and lows[last_low_idx] < lows[prev_low_idx]:
                downtrend = True

        # كسر الهيكل (BOS): كسر آخر قمة في اتجاه صاعد، أو كسر آخر قاع في اتجاه هابط
        bos_up = False
        bos_down = False
        if uptrend and closes[-1] > highs[last_high_idx]:
            bos_up = True
        if downtrend and closes[-1] < lows[last_low_idx]:
            bos_down = True

        # تغير الطابع (CHoCH): كسر القمة في اتجاه هابط (انعكاس صاعد) أو كسر القاع في اتجاه صاعد (انعكاس هابط)
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

# ======================
# مدير الإشعارات (محدث)
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}       # (coin, type) -> datetime
        self.last_notification_price = {}      # (coin, type) -> price
        self.cooldown_base = AppConfig.COOLDOWN_SECONDS
        self.cooldown_multiplier = {}           # (coin, type) -> multiplier for dynamic cooldown

    def add(self, notification: Notification):
        self.history.append(notification)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, limit: int = 10) -> List[Notification]:
        return self.history[-limit:] if self.history else []

    def should_send(self, coin_symbol: str, signal_type: str, confidence: float, current_price: float) -> bool:
        now = datetime.now()
        key = (coin_symbol, signal_type)

        # 1. Cooldown زمني
        last_time = self.last_notification_time.get(key)
        if last_time:
            delta = (now - last_time).total_seconds()
            multiplier = self.cooldown_multiplier.get(key, 1.0)
            cooldown = self.cooldown_base * multiplier
            if delta < cooldown:
                return False

        # 2. Minimum price move
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

        # إنشاء رسالة
        title = f"{signal.signal_type} Detected: {signal.coin_name}"
        message = (
            f"{title}\n"
            f"Confidence: {signal.confidence:.1f}%\n"
            f"Price: ${signal.price:,.2f}\n"
            f"Time: {signal.timestamp.strftime('%H:%M')}\n"
            f"Indicators: {json.dumps(signal.indicators, default=str)}"
        )

        tags = "arrow_up" if signal.signal_type == "TOP" else "arrow_down"
        priority = "4" if signal.confidence >= 85 else "3"

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

            # تحديث السجلات
            self.last_notification_time[key] = datetime.now()
            self.last_notification_price[key] = signal.price

            # تحديث cooldown multiplier
            multiplier = self.cooldown_multiplier.get(key, 1.0)
            self.cooldown_multiplier[key] = min(multiplier * 1.5, 5.0)

            return notification
        return None

    def reset_cooldown(self, coin_symbol: str, signal_type: str):
        key = (coin_symbol, signal_type)
        self.cooldown_multiplier[key] = max(1.0, self.cooldown_multiplier.get(key, 1.0) * 0.8)

# ======================
# المدير الرئيسي لاكتشاف القمم والقيعان (محسن)
# ======================
class TopBottomDetector:
    def __init__(self):
        self.detections: List[TopBottomSignal] = []
        self.last_update: Optional[datetime] = None
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()
        self.cached_higher_tf_data: Dict[str, Any] = {}

    def update_all(self) -> bool:
        with self.lock:
            logger.info(f"🔄 Scanning {len(AppConfig.COINS)} coins for tops/bottoms (advanced mode)...")
            success_count = 0

            for coin in AppConfig.COINS:
                if not coin.enabled:
                    continue
                try:
                    signal = self._scan_coin_advanced(coin)
                    if signal:
                        self.detections.append(signal)
                        self.notification_manager.create_notification(signal)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error on {coin.symbol}: {e}", exc_info=True)

            self.last_update = datetime.now()
            if len(self.detections) > 100:
                self.detections = self.detections[-100:]
            logger.info(f"✅ Found {success_count} potential tops/bottoms")
            return success_count > 0

    def _scan_coin_advanced(self, coin: CoinConfig) -> Optional[TopBottomSignal]:
        # جلب البيانات للإطار الزمني الرئيسي والإطار الأعلى
        ohlcv = self.binance.fetch_ohlcv(coin.symbol, AppConfig.TIMEFRAME, AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < 50:
            return None

        ohlcv_htf = self.binance.fetch_ohlcv(coin.symbol, AppConfig.HIGHER_TIMEFRAME, 100)

        opens = [c[1] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]
        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]

        ticker = self.binance.fetch_ticker(coin.symbol)
        if not ticker:
            return None
        current_price = ticker['last']

        # حساب المؤشرات
        rsi = TechnicalIndicators.rsi(closes, 14)
        atr = TechnicalIndicators.atr(highs, lows, closes, 14)
        adx = TechnicalIndicators.adx(highs, lows, closes, 14)

        current_rsi = rsi[-1] if rsi[-1] is not None else 50
        current_atr = atr[-1] if atr[-1] is not None else 0
        current_adx = adx[-1] if adx[-1] is not None else 20

        # تحديد نوع السوق (رينج أم ترند) بناءً على ADX
        market_regime = 'trend' if current_adx > 25 else 'ranging'

        # Pivot High/Low
        pivot_highs, pivot_lows = TechnicalIndicators.pivot_points(highs, lows, AppConfig.PIVOT_LEFT, AppConfig.PIVOT_RIGHT)

        # تحقق مما إذا كانت الشمعة الحالية قريبة من آخر Pivot حقيقي (ضمن ATR)
        near_pivot_high = False
        near_pivot_low = False

        # آخر 5 pivots
        valid_pivot_highs = [i for i, is_high in enumerate(pivot_highs) if is_high]
        valid_pivot_lows = [i for i, is_low in enumerate(pivot_lows) if is_low]

        if valid_pivot_highs and current_atr > 0:
            last_pivot_high_idx = valid_pivot_highs[-1]
            # التأكد من أن الـ pivot ليس قديماً جداً (آخر 20 شمعة)
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
        last_fractal_up = any(fractal_up[-AppConfig.FRACTAL_PERIOD*3:])  # نافذة أوسع قليلاً
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

        # حساب الثقة للقمم والقيعان
        top_confidence = self._calculate_confidence(
            'TOP', current_price, closes, rsi, atr, current_adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime
        )
        bottom_confidence = self._calculate_confidence(
            'BOTTOM', current_price, closes, rsi, atr, current_adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime
        )

        # اختيار الإشارة
        signal = None
        if top_confidence >= AppConfig.TOP_CONFIDENCE_THRESHOLD and top_confidence > bottom_confidence:
            # تحقق إضافي: إذا كان الإطار الأعلى في اتجاه صاعد لا نرسل قمة (إلا إذا كان هناك انعكاس واضح)
            if not htf_confirmation.get('trend_up', False) or ms.get('choch_down', False):
                signal = TopBottomSignal(
                    coin_symbol=coin.symbol,
                    coin_name=coin.name,
                    signal_type="TOP",
                    confidence=top_confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    indicators={
                        'rsi': current_rsi,
                        'atr_percent': atr_percent,
                        'adx': current_adx,
                        'market_regime': market_regime,
                        'near_pivot_high': near_pivot_high,
                        'fractal_up': last_fractal_up,
                        'divergence_bearish': divergence.get('bearish', False),
                        'ms_trend': ms['trend'],
                        'ms_choch_down': ms.get('choch_down', False),
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown')
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
                        'rsi': current_rsi,
                        'atr_percent': atr_percent,
                        'adx': current_adx,
                        'market_regime': market_regime,
                        'near_pivot_low': near_pivot_low,
                        'fractal_down': last_fractal_down,
                        'divergence_bullish': divergence.get('bullish', False),
                        'ms_trend': ms['trend'],
                        'ms_choch_up': ms.get('choch_up', False),
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown')
                    },
                    message=f"Bottom detected with {bottom_confidence:.1f}% confidence"
                )

        return signal

    def _check_higher_timeframe(self, symbol: str, ohlcv_htf: Optional[List]) -> Dict[str, Any]:
        """تحليل الإطار الأعلى لتأكيد الاتجاه"""
        if not ohlcv_htf or len(ohlcv_htf) < 20:
            return {'trend': 'unknown', 'trend_up': False, 'trend_down': False, 'last_high': None, 'last_low': None}
        closes_htf = [c[4] for c in ohlcv_htf]
        highs_htf = [c[2] for c in ohlcv_htf]
        lows_htf = [c[3] for c in ohlcv_htf]

        # SMA 20 لتحديد الاتجاه
        sma20 = TechnicalIndicators.sma(closes_htf, 20)
        current_sma = sma20[-1] if sma20[-1] is not None else closes_htf[-1]
        trend_up = closes_htf[-1] > current_sma
        trend_down = closes_htf[-1] < current_sma

        # آخر قمة وقاع
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
                              market_regime: str) -> float:
        """حساب الثقة باستخدام نظام الأوزان الديناميكي"""
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

# تشغيل المحدّث الخلفي
updater_thread = threading.Thread(target=background_updater, daemon=True)
updater_thread.start()

# تحديث أولي
detector.update_all()

# ======================
# المسارات (Routes)
# ======================
@app.route('/')
def index():
    detections = detector.get_recent_detections(10)
    stats = detector.get_stats()
    return render_template('index.html', detections=detections, stats=stats)

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

# ======================
# إشعار بدء التشغيل
# ======================
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

# ======================
# التشغيل الرئيسي
# ======================
if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 Crypto Tops & Bottoms Detector Advanced v2.1 (Fixed)")
    logger.info(f"📊 Coins: {len(AppConfig.COINS)}")
    logger.info(f"🔄 Update every {AppConfig.UPDATE_INTERVAL//60} minutes")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

"""
إضافة باكتيست متكامل لاختبار أداء الإشارات على بيانات تاريخية
يضاف إلى نهاية الملف app1.py
"""

import itertools
from collections import defaultdict

# ======================
# إعدادات الباكتيست
# ======================
class BacktestConfig:
    MONTHS = 6                      # عدد الأشهر للاختبار
    TIMEFRAME = AppConfig.TIMEFRAME  # نفس الإطار المستخدم في الكشف
    PROFIT_TARGET_PCT = 2.0          # النسبة المئوية للحركة المتوقعة بعد الإشارة (2%)
    MAX_BARS_AHEAD = 10               # عدد الشموع المستقبلية للتحقق من النجاح
    CONFIDENCE_THRESHOLD = 50         # حد الثقة الأدنى للإشارات المشمولة في الاختبار
    EXIT_ON_OPPOSITE_SIGNAL = False   # هل نخرج عند إشارة معاكسة؟

# ======================
# باكتيست
# ======================
class Backtester:
    def __init__(self, detector):
        self.detector = detector
        self.binance = detector.binance
        self.results = {}
        self.running = False

    def fetch_historical_data(self, symbol: str, months: int, timeframe: str) -> Optional[List]:
        """جلب بيانات تاريخية لعدد محدد من الأشهر مع التعامل مع pagination"""
        all_candles = []
        # حساب وقت البدء: months شهر مضت
        since = self.binance.exchange.parse8601((datetime.now() - timedelta(days=30*months)).isoformat())
        limit = 1000  # الحد الأقصى لكل طلب في binance

        while True:
            try:
                candles = self.binance.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not candles:
                    break
                all_candles.extend(candles)
                if len(candles) < limit:
                    break
                # تحديث since إلى وقت آخر شمعة + 1 ms
                since = candles[-1][0] + 1
                # تجنب rate limit
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                break

        logger.info(f"Fetched {len(all_candles)} candles for {symbol}")
        return all_candles if all_candles else None

    def run_backtest_for_coin(self, coin: CoinConfig) -> Dict:
        """تشغيل الباكتيست لعملة واحدة"""
        logger.info(f"Running backtest for {coin.symbol}...")
        candles = self.fetch_historical_data(coin.symbol, BacktestConfig.MONTHS, BacktestConfig.TIMEFRAME)
        if not candles or len(candles) < 100:
            return {"error": "Insufficient data"}

        # تحويل البيانات إلى قوائم منفصلة
        timestamps = [c[0] for c in candles]
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]

        total_candles = len(candles)
        signals = []  # قائمة بالإشارات التي سنختبرها

        # محاكاة الوقت الحقيقي: نمر على كل شمعة (نبدأ بعدد كافٍ من الشموع لحساب المؤشرات)
        min_bars_needed = 60  # عدد كافٍ لمعظم المؤشرات
        for i in range(min_bars_needed, total_candles):
            # استخراج البيانات حتى الشمعة i (الشمعة الحالية)
            current_highs = highs[:i+1]
            current_lows = lows[:i+1]
            current_closes = closes[:i+1]
            current_volumes = volumes[:i+1]

            # حساب المؤشرات يدوياً (نحتاج إلى إعادة استخدام دوال TechnicalIndicators)
            # هذا قد يكون مكلفاً، لكن للباك تست يمكن تحمله
            rsi = TechnicalIndicators.rsi(current_closes, 14)
            atr = TechnicalIndicators.atr(current_highs, current_lows, current_closes, 14)
            adx = TechnicalIndicators.adx(current_highs, current_lows, current_closes, 14)

            # نحتاج فقط آخر القيم
            current_rsi = rsi[-1] if rsi and rsi[-1] is not None else 50
            current_atr = atr[-1] if atr and atr[-1] is not None else 0
            current_adx = adx[-1] if adx and adx[-1] is not None else 20

            # تحديد market regime
            market_regime = 'trend' if current_adx > 25 else 'ranging'

            # Pivot points
            pivot_highs, pivot_lows = TechnicalIndicators.pivot_points(
                current_highs, current_lows, AppConfig.PIVOT_LEFT, AppConfig.PIVOT_RIGHT
            )

            # Near pivot
            near_pivot_high = False
            near_pivot_low = False
            valid_pivot_highs = [idx for idx, v in enumerate(pivot_highs) if v]
            valid_pivot_lows = [idx for idx, v in enumerate(pivot_lows) if v]

            if valid_pivot_highs and current_atr > 0:
                last_pivot_idx = valid_pivot_highs[-1]
                if i - last_pivot_idx <= 20:
                    if abs(closes[i] - highs[last_pivot_idx]) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                        near_pivot_high = True
            if valid_pivot_lows and current_atr > 0:
                last_pivot_idx = valid_pivot_lows[-1]
                if i - last_pivot_idx <= 20:
                    if abs(closes[i] - lows[last_pivot_idx]) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                        near_pivot_low = True

            # Fractals
            fractal_up, fractal_down = TechnicalIndicators.fractal(
                current_highs, current_lows, AppConfig.FRACTAL_PERIOD
            )
            last_fractal_up = any(fractal_up[-AppConfig.FRACTAL_PERIOD*3:])
            last_fractal_down = any(fractal_down[-AppConfig.FRACTAL_PERIOD*3:])

            # Divergence
            divergence = TechnicalIndicators.detect_divergence(
                current_closes, rsi, window=40
            )

            # Market structure
            ms = TechnicalIndicators.market_structure(
                current_highs, current_lows, current_closes, pivot_highs, pivot_lows
            )

            # Volume spike
            if i >= 20:
                avg_vol = sum(current_volumes[-20:-1]) / 19
                volume_spike = current_volumes[-1] > avg_vol * 1.5
            else:
                volume_spike = False

            # ATR percent
            atr_percent = (current_atr / closes[i]) * 100 if closes[i] > 0 else 0

            # Multi-timeframe confirmation (لا نستخدمها في الباك تست لعدم وجود بيانات الإطار الأعلى)
            htf_conf = {'trend_up': False, 'trend_down': False}

            # حساب الثقة
            top_conf = self.detector._calculate_confidence(
                'TOP', closes[i], current_closes, rsi, atr, current_adx,
                near_pivot_high, near_pivot_low,
                last_fractal_up, last_fractal_down,
                divergence, ms, volume_spike, htf_conf, market_regime
            )
            bottom_conf = self.detector._calculate_confidence(
                'BOTTOM', closes[i], current_closes, rsi, atr, current_adx,
                near_pivot_high, near_pivot_low,
                last_fractal_up, last_fractal_down,
                divergence, ms, volume_spike, htf_conf, market_regime
            )

            # تطبيق عتبة الثقة
            if top_conf >= BacktestConfig.CONFIDENCE_THRESHOLD and top_conf > bottom_conf:
                signals.append({
                    'index': i,
                    'type': 'TOP',
                    'price': closes[i],
                    'timestamp': timestamps[i],
                    'confidence': top_conf,
                    'atr': current_atr
                })
            elif bottom_conf >= BacktestConfig.CONFIDENCE_THRESHOLD:
                signals.append({
                    'index': i,
                    'type': 'BOTTOM',
                    'price': closes[i],
                    'timestamp': timestamps[i],
                    'confidence': bottom_conf,
                    'atr': current_atr
                })

        logger.info(f"Generated {len(signals)} signals for {coin.symbol}")

        # تقييم الإشارات
        results = []
        for sig in signals:
            idx = sig['index']
            entry_price = sig['price']
            signal_type = sig['type']
            atr_val = sig['atr'] or 0

            # نحدد نافذة التحقق: عدد محدود من الشموع المستقبلية
            end_idx = min(idx + BacktestConfig.MAX_BARS_AHEAD, total_candles - 1)
            success = False
            exit_price = entry_price
            exit_reason = "timeout"

            for j in range(idx + 1, end_idx + 1):
                current_price = closes[j]
                if signal_type == 'TOP':
                    # نتوقع هبوط: نبحث عن انخفاض بنسبة PROFIT_TARGET_PCT
                    if (entry_price - current_price) / entry_price * 100 >= BacktestConfig.PROFIT_TARGET_PCT:
                        success = True
                        exit_price = current_price
                        exit_reason = "target_hit"
                        break
                    # إذا ارتفع أكثر من نسبة معينة (مثلاً 1%)، نعتبر فشلاً (وقف خسارة)
                    elif (current_price - entry_price) / entry_price * 100 > 1.0:
                        success = False
                        exit_price = current_price
                        exit_reason = "stop_loss"
                        break
                else:  # BOTTOM
                    if (current_price - entry_price) / entry_price * 100 >= BacktestConfig.PROFIT_TARGET_PCT:
                        success = True
                        exit_price = current_price
                        exit_reason = "target_hit"
                        break
                    elif (entry_price - current_price) / entry_price * 100 > 1.0:
                        success = False
                        exit_price = current_price
                        exit_reason = "stop_loss"
                        break

            # حساب الربح/الخسارة
            if signal_type == 'TOP':
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            else:
                pnl_pct = (exit_price - entry_price) / entry_price * 100

            results.append({
                'timestamp': datetime.fromtimestamp(sig['timestamp']/1000).isoformat(),
                'type': signal_type,
                'entry': entry_price,
                'exit': exit_price,
                'pnl_pct': pnl_pct,
                'success': success,
                'confidence': sig['confidence'],
                'exit_reason': exit_reason
            })

        # تحليل النتائج
        total = len(results)
        if total == 0:
            return {"symbol": coin.symbol, "total_signals": 0, "message": "No signals generated"}

        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        win_rate = successful / total * 100 if total > 0 else 0
        avg_pnl = sum(r['pnl_pct'] for r in results) / total
        total_pnl = sum(r['pnl_pct'] for r in results)

        # تفصيل حسب النوع
        tops = [r for r in results if r['type'] == 'TOP']
        bottoms = [r for r in results if r['type'] == 'BOTTOM']
        tops_win = sum(1 for r in tops if r['success'])
        bottoms_win = sum(1 for r in bottoms if r['success'])

        return {
            "symbol": coin.symbol,
            "total_signals": total,
            "successful": successful,
            "failed": failed,
            "win_rate": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "tops": len(tops),
            "tops_win": tops_win,
            "tops_win_rate": round(tops_win/len(tops)*100, 2) if tops else 0,
            "bottoms": len(bottoms),
            "bottoms_win": bottoms_win,
            "bottoms_win_rate": round(bottoms_win/len(bottoms)*100, 2) if bottoms else 0,
            "details": results[-20:]  # آخر 20 إشارة للعرض
        }

    def run_all(self) -> Dict:
        """تشغيل الباكتيست على جميع العملات"""
        self.running = True
        all_results = {}
        for coin in AppConfig.COINS:
            if not coin.enabled:
                continue
            try:
                res = self.run_backtest_for_coin(coin)
                all_results[coin.symbol] = res
            except Exception as e:
                logger.error(f"Backtest error for {coin.symbol}: {e}")
                all_results[coin.symbol] = {"error": str(e)}
        self.running = False
        self.results = all_results
        return all_results

    def generate_report(self) -> str:
        """توليد تقرير نصي للنتائج"""
        if not self.results:
            return "No backtest results available."

        lines = []
        lines.append("📊 **Backtest Results (Last 6 Months)**")
        lines.append("")

        total_signals = 0
        total_success = 0
        total_pnl = 0.0

        for symbol, res in self.results.items():
            if "error" in res:
                lines.append(f"❌ {symbol}: {res['error']}")
                continue
            lines.append(f"**{symbol}**")
            lines.append(f"- Signals: {res['total_signals']}")
            lines.append(f"- Win Rate: {res['win_rate']}% ({res['successful']}/{res['total_signals']})")
            lines.append(f"- Avg PnL: {res['avg_pnl']}%")
            lines.append(f"- Total PnL: {res['total_pnl']}%")
            lines.append(f"  Tops: {res['tops']} (win: {res['tops_win_rate']}%)")
            lines.append(f"  Bottoms: {res['bottoms']} (win: {res['bottoms_win_rate']}%)")
            lines.append("")

            total_signals += res['total_signals']
            total_success += res['successful']
            total_pnl += res['total_pnl']

        if total_signals > 0:
            overall_win = total_success / total_signals * 100
            lines.append("**Overall**")
            lines.append(f"- Total Signals: {total_signals}")
            lines.append(f"- Overall Win Rate: {overall_win:.2f}%")
            lines.append(f"- Total PnL (sum): {total_pnl:.2f}%")

        return "\n".join(lines)

    def send_report(self):
        """إرسال التقرير إلى ntfy"""
        report = self.generate_report()
        title = "Backtest Results"
        self.detector.notification_manager.send_ntfy(report, title, priority="4", tags="bar_chart")

# ======================
# إضافة route جديد لتشغيل الباكتيست
# ======================
@app.route('/api/run_backtest', methods=['POST'])
def run_backtest():
    """تشغيل الباكتيست بشكل غير متزامن وإرسال النتائج إلى ntfy"""
    if hasattr(app, 'backtester_running') and app.backtester_running:
        return jsonify({"status": "error", "message": "Backtest already running"}), 409

    def backtest_thread():
        app.backtester_running = True
        try:
            backtester = Backtester(detector)
            results = backtester.run_all()
            backtester.send_report()
            logger.info("Backtest completed and report sent.")
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
        finally:
            app.backtester_running = False

    app.backtester_running = False
    thread = threading.Thread(target=backtest_thread, daemon=True)
    thread.start()
    return jsonify({"status": "success", "message": "Backtest started. Results will be sent to ntfy."})

# يمكن أيضاً إضافة route لاستعراض النتائج المخزنة
@app.route('/api/backtest_results')
def get_backtest_results():
    backtester = Backtester(detector)
    return jsonify(backtester.results)

# ======================
# ملاحظة: يجب إضافة import itertools في أعلى الملف (إذا لم يكن موجوداً)
# ======================
