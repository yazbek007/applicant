"""
Crypto Tops & Bottoms Detector Bot - النسخة المتقدمة (بدون مكتبات خارجية ثقيلة)
إصدار 2.0 - يكتشف القمم والقيعان باستخدام تحليل متقدم (Pivot Points، ATR، Fractals، Divergence، Market Structure)
ويرسل إشعارات NTFY باللغة الإنجليزية
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

import numpy as np  # متاح على Render عادة
from flask import Flask, render_template, jsonify, request
import ccxt

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
    HIGHER_TIMEFRAME = '1h'          # للتأكيد متعدد الأطراف
    MAX_CANDLES = 300

    # إعدادات Pivot
    PIVOT_LEFT = 5
    PIVOT_RIGHT = 5
    MIN_PIVOT_DISTANCE_ATR = 1.5      # مضاعف ATR للمسافة بين القمم/القيعان

    # Fractal
    FRACTAL_PERIOD = 2                 # 2 شمعة على كل جانب => 5 شموع

    # عتبات الثقة
    TOP_CONFIDENCE_THRESHOLD = 65
    BOTTOM_CONFIDENCE_THRESHOLD = 65

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
# Binance Client
# ======================
class BinanceClient:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': ExternalAPIConfig.BINANCE_API_KEY,
            'secret': ExternalAPIConfig.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str = AppConfig.TIMEFRAME, limit: int = AppConfig.MAX_CANDLES) -> Optional[List]:
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Binance OHLCV error {symbol} {timeframe}: {e}")
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Binance ticker error {symbol}: {e}")
            return None

# ======================
# المؤشرات الفنية المحسوبة يدوياً (بدون مكتبات إضافية)
# ======================
class TechnicalIndicators:
    """حساب المؤشرات باستخدام عمليات أساسية على قوائم."""

    @staticmethod
    def sma(values: List[float], period: int) -> List[Optional[float]]:
        """Simple Moving Average"""
        result = [None] * len(values)
        for i in range(period - 1, len(values)):
            result[i] = sum(values[i - period + 1:i + 1]) / period
        return result

    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[Optional[float]]:
        """Relative Strength Index"""
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
                rs = float('inf')
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
            if i < len(prices) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        return rsi_values

    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average True Range"""
        if len(closes) < period + 1:
            return [None] * len(closes)
        tr = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr.append(max(hl, hc, lc))
        # أول period-1 قيم None، ثم نبدأ حساب ATR
        atr_values = [None] * period
        # أول ATR هو متوسط أول period من TR
        first_atr = sum(tr[:period]) / period
        atr_values.append(first_atr)
        for i in range(period, len(tr)):
            atr_values.append((atr_values[-1] * (period - 1) + tr[i]) / period)
        # التأكد من أن الطول مطابق لـ closes
        # نظراً لأن TR أقصر بواحد، نحتاج لملء بداية القائمة
        # سيكون atr_values بطول len(closes) إذا بدأنا من i=period في closes
        # لكننا بدأنا من i=period+1؟ دعنا ننشئ قائمة بالطول الصحيح.
        # طريقة مبسطة: نعيد قائمة ATR تتوافق مع الفهارس حيث i >= period
        full_atr = [None] * len(closes)
        # أول ATR حقيقي عند index period (لأننا نحتاج period من TR، و TR يبدأ من index 1)
        # لذلك full_atr[period] = first_atr
        # نضع القيم من هناك
        for i in range(period, len(closes)):
            if i == period:
                full_atr[i] = first_atr
            else:
                # نحتاج TR[i-1] لأن TR لكل شمعة i (باستثناء الأولى) محسوب بين i و i-1
                # عند i>period، نستخدم معادلة ATR السلسة
                tr_index = i - 1  # TR للشمعة i (بين i و i-1) موجود في tr[tr_index-1]؟ لنتأكد.
                # tr[0] يتوافق مع الشمعة 1، tr[1] مع الشمعة 2، ... tr[k] مع الشمعة k+1
                # لذا للشمعة i (i>=1) TR index = i-1
                if i-1 < len(tr):
                    full_atr[i] = (full_atr[i-1] * (period - 1) + tr[i-1]) / period
        return full_atr

    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[Optional[float]]:
        """Average Directional Index"""
        if len(closes) < period + 1:
            return [None] * len(closes)
        # حساب +DM و -DM
        plus_dm = [0.0]
        minus_dm = [0.0]
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0.0)
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0.0)
        # حساب TR (كما في ATR)
        tr = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr.append(max(hl, hc, lc))
        # تجانس TR, +DM, -DM باستخدام Wilder's smoothing (period)
        smoothed_tr = TechnicalIndicators._wilder_smooth(tr, period)
        smoothed_plus_dm = TechnicalIndicators._wilder_smooth(plus_dm[1:], period)  # نتجاهل أول عنصر
        smoothed_minus_dm = TechnicalIndicators._wilder_smooth(minus_dm[1:], period)
        # حساب +DI و -DI
        plus_di = [None] * len(closes)
        minus_di = [None] * len(closes)
        dx = [None] * len(closes)
        adx = [None] * len(closes)
        # أول قيمة متاحة بعد period
        for i in range(period, len(closes)):
            tr_val = smoothed_tr[i - period]  # smoothed_tr يبدأ من 0 للفترة period
            if tr_val != 0:
                plus_di[i] = 100 * smoothed_plus_dm[i - period] / tr_val
                minus_di[i] = 100 * smoothed_minus_dm[i - period] / tr_val
                di_diff = abs(plus_di[i] - minus_di[i])
                di_sum = plus_di[i] + minus_di[i]
                if di_sum != 0:
                    dx[i] = 100 * di_diff / di_sum
        # حساب ADX كـ smoothed average of DX
        dx_vals = [dx[i] for i in range(len(dx)) if dx[i] is not None]
        if len(dx_vals) >= period:
            adx_values = TechnicalIndicators._wilder_smooth(dx_vals, period)
            # وضعها في المواضع المناسبة
            start = period + period  # لأننا نحتاج period لـ +DI ثم period لـ ADX
            for j, val in enumerate(adx_values):
                if start + j < len(adx):
                    adx[start + j] = val
        return adx

    @staticmethod
    def _wilder_smooth(values: List[float], period: int) -> List[float]:
        """Wilder's smoothing (used in RSI, ADX)"""
        if len(values) < period:
            return []
        smoothed = [sum(values[:period]) / period]
        for i in range(period, len(values)):
            smoothed.append((smoothed[-1] * (period - 1) + values[i]) / period)
        return smoothed

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
    def detect_divergence(prices: List[float], oscillator: List[Optional[float]], window: int = 20) -> Dict[str, bool]:
        """كشف الانحراف (divergence) بين السعر والمذبذب (مثل RSI)"""
        # نحتاج آخر window شمعة
        recent_prices = prices[-window:]
        recent_osc = oscillator[-window:]
        # إزالة القيم الخالية
        valid_idx = [i for i, v in enumerate(recent_osc) if v is not None]
        if len(valid_idx) < 5:
            return {'bullish': False, 'bearish': False}
        prices_valid = [recent_prices[i] for i in valid_idx]
        osc_valid = [recent_osc[i] for i in valid_idx]
        # العثور على القمم والقيعان في السعر والمذبذب
        # طريقة مبسطة: نستخدم max/min خلال النافذة
        price_max_idx = np.argmax(prices_valid)
        price_min_idx = np.argmin(prices_valid)
        osc_max_idx = np.argmax(osc_valid)
        osc_min_idx = np.argmin(osc_valid)

        bullish_div = False
        bearish_div = False

        # انحراف صاعد: قاع سعري أدنى لكن قاع مذبذب أعلى
        if price_min_idx == len(prices_valid)-1 and osc_min_idx == len(osc_valid)-1:
            # قاع حديث
            prev_price_min = min(prices_valid[:-1])
            prev_osc_min = min(osc_valid[:-1])
            if prices_valid[-1] < prev_price_min and osc_valid[-1] > prev_osc_min:
                bullish_div = True

        # انحراف هابط: قمة سعرية أعلى لكن قمة مذبذب أقل
        if price_max_idx == len(prices_valid)-1 and osc_max_idx == len(osc_valid)-1:
            prev_price_max = max(prices_valid[:-1])
            prev_osc_max = max(osc_valid[:-1])
            if prices_valid[-1] > prev_price_max and osc_valid[-1] < prev_osc_max:
                bearish_div = True

        return {'bullish': bullish_div, 'bearish': bearish_div}

    @staticmethod
    def market_structure(highs: List[float], lows: List[float], closes: List[float], pivot_highs: List[bool], pivot_lows: List[bool]) -> Dict[str, Any]:
        """تحليل هيكل السوق: تحديد آخر قمة/قاع صاعدة/هابطة، وكشف MSB و CHoCH"""
        # العثور على آخر قمة وقاع معتبرة (pivot)
        last_high_idx = None
        last_low_idx = None
        for i in range(len(highs)-1, -1, -1):
            if pivot_highs[i] and (last_high_idx is None):
                last_high_idx = i
            if pivot_lows[i] and (last_low_idx is None):
                last_low_idx = i
            if last_high_idx is not None and last_low_idx is not None:
                break

        # الاتجاه الحالي بناءً على آخر قمة وقاع
        trend = 'unknown'
        msb_detected = False
        choch_detected = False

        if last_high_idx is not None and last_low_idx is not None:
            if last_high_idx > last_low_idx:  # آخر قمة بعد آخر قاع
                # تحقق مما إذا كانت القمة أعلى من القمة التي قبلها
                # نحتاج قمة سابقة
                prev_high_idx = None
                for i in range(last_high_idx-1, -1, -1):
                    if pivot_highs[i]:
                        prev_high_idx = i
                        break
                if prev_high_idx is not None:
                    if highs[last_high_idx] > highs[prev_high_idx]:
                        trend = 'uptrend'
                    else:
                        trend = 'downtrend'  # قمة أدنى
                # قاع سابق
                prev_low_idx = None
                for i in range(last_low_idx-1, -1, -1):
                    if pivot_lows[i]:
                        prev_low_idx = i
                        break
                if prev_low_idx is not None:
                    if lows[last_low_idx] < lows[prev_low_idx]:
                        # قاع أدنى، يتوافق مع downtrend
                        if trend == 'uptrend':
                            # هذا قد يكون MSB إذا كسر القاع السابق في اتجاه صاعد؟ يحتاج تعريف أدق
                            msb_detected = True
                    else:
                        # قاع أعلى
                        if trend == 'downtrend':
                            choch_detected = True
        # تبسيط: نستخدم السعر الحالي بالنسبة للقمة والقاع الأخيرين
        current_close = closes[-1]
        if last_high_idx is not None and current_close > highs[last_high_idx]:
            choch_detected = True  # كسر القمة الأخيرة
        if last_low_idx is not None and current_close < lows[last_low_idx]:
            choch_detected = True  # كسر القاع الأخير (عكس الاتجاه)

        return {
            'trend': trend,
            'msb': msb_detected,
            'choch': choch_detected,
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
            # cooldown ديناميكي: يزيد مع تكرار الإشارات
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

        # 3. تحديث cooldown multiplier (إذا تم إرسال الإشارة، نزيده)
        # سنقوم بذلك بعد الإرسال الفعلي

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

            # تحديث cooldown multiplier: زيادة بعد كل إشارة
            multiplier = self.cooldown_multiplier.get(key, 1.0)
            self.cooldown_multiplier[key] = min(multiplier * 1.5, 5.0)  # حد أقصى 5x

            return notification
        return None

    def reset_cooldown(self, coin_symbol: str, signal_type: str):
        """يمكن استدعاؤها بعد فترة هدوء طويلة لتقليل المضاعف"""
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
        self.cached_higher_tf_data: Dict[str, Any] = {}  # تخزين مؤقت للبيانات عالية الإطار

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

        ohlcv_htf = self.binance.fetch_ohlcv(coin.symbol, AppConfig.HIGHER_TIMEFRAME, 100)  # نأخذ 100 شمعة كافية

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

        # تصفية Pivot باستخدام ATR: يجب أن تكون المسافة بين القمم/القيعان > min_atr_distance
        # نأخذ آخر قمة وقاع حقيقيين
        valid_pivot_highs = [i for i, is_high in enumerate(pivot_highs) if is_high]
        valid_pivot_lows = [i for i, is_low in enumerate(pivot_lows) if is_low]

        # فحص ما إذا كانت آخر شمعة قريبة من أي Pivot (ضمن ATR)
        near_pivot_high = False
        near_pivot_low = False
        if valid_pivot_highs and current_atr > 0:
            last_pivot_high_idx = valid_pivot_highs[-1]
            if len(highs) - last_pivot_high_idx <= AppConfig.PIVOT_RIGHT * 2:
                # إذا كان آخر pivot قريب (خلال ضعف right)
                pivot_price = highs[last_pivot_high_idx]
                if abs(current_price - pivot_price) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                    near_pivot_high = True
        if valid_pivot_lows and current_atr > 0:
            last_pivot_low_idx = valid_pivot_lows[-1]
            if len(lows) - last_pivot_low_idx <= AppConfig.PIVOT_RIGHT * 2:
                pivot_price = lows[last_pivot_low_idx]
                if abs(current_price - pivot_price) <= current_atr * AppConfig.MIN_PIVOT_DISTANCE_ATR:
                    near_pivot_low = True

        # Fractals
        fractal_up, fractal_down = TechnicalIndicators.fractal(highs, lows, AppConfig.FRACTAL_PERIOD)
        last_fractal_up = any(fractal_up[-AppConfig.FRACTAL_PERIOD*2:])
        last_fractal_down = any(fractal_down[-AppConfig.FRACTAL_PERIOD*2:])

        # Divergence بين السعر و RSI
        divergence = TechnicalIndicators.detect_divergence(closes, rsi, window=30)

        # Market Structure
        ms = TechnicalIndicators.market_structure(highs, lows, closes, pivot_highs, pivot_lows)

        # Multi-timeframe confirmation (باستخدام الإطار الأعلى)
        htf_confirmation = self._check_higher_timeframe(coin.symbol, ohlcv_htf, signal_type=None)  # سنمرر النوع لاحقاً

        # Volume spike
        avg_vol = sum(volumes[-20:-1]) / 19
        volume_spike = volumes[-1] > avg_vol * 1.5

        # Volatility filter: نتحقق من أن ATR% أعلى من الحد الأدنى
        atr_percent = (current_atr / current_price) * 100 if current_price > 0 else 0
        if atr_percent < AppConfig.MIN_VOLATILITY_ATR_PERCENT:
            logger.debug(f"{coin.symbol} volatility too low ({atr_percent:.2f}%), skipping")
            return None

        # تجاهل الإشارات ضد اتجاه الإطار الأعلى
        # سنقوم بذلك داخل حساب الثقة

        # حساب الثقة للقمم والقيعان
        top_confidence = self._calculate_confidence(
            'TOP', current_price, closes, rsi, atr, adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime
        )
        bottom_confidence = self._calculate_confidence(
            'BOTTOM', current_price, closes, rsi, atr, adx,
            near_pivot_high, near_pivot_low,
            last_fractal_up, last_fractal_down,
            divergence, ms, volume_spike, htf_confirmation, market_regime
        )

        # اختيار الإشارة
        signal = None
        if top_confidence >= AppConfig.TOP_CONFIDENCE_THRESHOLD and top_confidence > bottom_confidence:
            # تحقق إضافي: ضد اتجاه الإطار الأعلى
            if not htf_confirmation.get('trend_down', False):  # إذا كان الإطار الأعلى غير هابط بقوة
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
                        'ms_choch': ms['choch'],
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown')
                    },
                    message=f"Top detected with {top_confidence:.1f}% confidence"
                )
        elif bottom_confidence >= AppConfig.BOTTOM_CONFIDENCE_THRESHOLD:
            if not htf_confirmation.get('trend_up', False):
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
                        'ms_choch': ms['choch'],
                        'volume_spike': volume_spike,
                        'htf_trend': htf_confirmation.get('trend', 'unknown')
                    },
                    message=f"Bottom detected with {bottom_confidence:.1f}% confidence"
                )

        return signal

    def _check_higher_timeframe(self, symbol: str, ohlcv_htf: Optional[List], signal_type: Optional[str] = None) -> Dict[str, Any]:
        """تحليل الإطار الأعلى لتأكيد الاتجاه"""
        if not ohlcv_htf or len(ohlcv_htf) < 20:
            return {'trend': 'unknown', 'trend_up': False, 'trend_down': False}
        closes_htf = [c[4] for c in ohlcv_htf]
        highs_htf = [c[2] for c in ohlcv_htf]
        lows_htf = [c[3] for c in ohlcv_htf]

        # SMA بسيط لتحديد الاتجاه
        sma20 = TechnicalIndicators.sma(closes_htf, 20)
        current_sma = sma20[-1] if sma20[-1] is not None else closes_htf[-1]
        trend_up = closes_htf[-1] > current_sma
        trend_down = closes_htf[-1] < current_sma

        # قمم وقيعان سريعة
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
        reasons = []

        # 1. Pivot / near pivot
        if signal_type == 'TOP' and near_pivot_high:
            confidence += weights['pivot'] * 100
            reasons.append('pivot_high')
        if signal_type == 'BOTTOM' and near_pivot_low:
            confidence += weights['pivot'] * 100
            reasons.append('pivot_low')

        # 2. RSI Divergence
        if signal_type == 'TOP' and divergence.get('bearish', False):
            confidence += weights['rsi_div'] * 100
            reasons.append('bearish_divergence')
        if signal_type == 'BOTTOM' and divergence.get('bullish', False):
            confidence += weights['rsi_div'] * 100
            reasons.append('bullish_divergence')

        # 3. Fractal
        if signal_type == 'TOP' and fractal_up:
            confidence += weights['fractal'] * 100
            reasons.append('fractal_up')
        if signal_type == 'BOTTOM' and fractal_down:
            confidence += weights['fractal'] * 100
            reasons.append('fractal_down')

        # 4. Volume spike (يدعم كلا الاتجاهين)
        if volume_spike:
            confidence += weights['volume'] * 100
            reasons.append('volume_spike')

        # 5. Market Structure Break / Change of Character
        if signal_type == 'TOP' and ms.get('choch', False) and ms.get('trend') == 'uptrend':
            confidence += weights['msb'] * 100
            reasons.append('choch_uptrend')
        if signal_type == 'BOTTOM' and ms.get('choch', False) and ms.get('trend') == 'downtrend':
            confidence += weights['msb'] * 100
            reasons.append('choch_downtrend')

        # 6. RSI extremes (إضافة دعم إضافي)
        current_rsi = rsi[-1] if rsi[-1] is not None else 50
        if signal_type == 'TOP' and current_rsi > 70:
            confidence += 10  # مكافأة
            reasons.append('rsi_overbought')
        if signal_type == 'BOTTOM' and current_rsi < 30:
            confidence += 10
            reasons.append('rsi_oversold')

        # 7. Multi-timeframe confirmation (يضاف إذا كان متوافقاً)
        if signal_type == 'TOP' and htf_conf.get('trend_down', False):
            confidence += 15  # الإطار الأعلى هابط يدعم القمة
            reasons.append('htf_downtrend')
        if signal_type == 'BOTTOM' and htf_conf.get('trend_up', False):
            confidence += 15
            reasons.append('htf_uptrend')

        # تطبيق decay زمني؟ يمكن إضافته لاحقاً إذا أردنا خفض الثقة مع تقدم الوقت منذ آخر إشارة
        # لكننا نتعامل مع الإشارة الحالية فقط

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
    logger.info("🚀 Crypto Tops & Bottoms Detector Advanced v2.0")
    logger.info(f"📊 Coins: {len(AppConfig.COINS)}")
    logger.info(f"🔄 Update every {AppConfig.UPDATE_INTERVAL//60} minutes")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
