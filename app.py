"""
Crypto Tops & Bottoms Detector Bot - النسخة الخفيفة والمحسنة للذاكرة
إصدار 1.0 - يكتشف القمم والقيعان لأهم 5 عملات ويرسل إشعارات NTFY باللغة الإنجليزية
"""

import os
import json
import time
import math
import logging
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock

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
    indicators: Dict[str, Any]  # تفاصيل المؤشرات المستخدمة
    message: str

@dataclass
class Notification:
    id: str
    timestamp: datetime
    coin_symbol: str
    coin_name: str
    message: str
    notification_type: str  # "TOP" or "BOTTOM"
    signal_strength: float  # confidence
    price: float

# ======================
# إعدادات التطبيق
# ======================
class AppConfig:
    # العملات المدعومة (نختار أول 5 عملات من القائمة)
    COINS = [
        CoinConfig("BTC/USDT", "Bitcoin"),
        CoinConfig("ETH/USDT", "Ethereum"),
        CoinConfig("BNB/USDT", "Binance Coin"),
        CoinConfig("SOL/USDT", "Solana"),
        CoinConfig("XRP/USDT", "Ripple"),
    ]

    # إعدادات اكتشاف القمم والقيعان
    SWING_LEFT = 5
    SWING_RIGHT = 5
    TOP_CONFIDENCE_THRESHOLD = 60
    BOTTOM_CONFIDENCE_THRESHOLD = 60
    COOLDOWN_SECONDS = 300  # ساعة واحدة بين الإشعارات لنفس العملة ونفس النوع

    UPDATE_INTERVAL = 120  # 2 دقيقة
    MAX_CANDLES = 200

# ======================
# إعدادات APIs الخارجية
# ======================
class ExternalAPIConfig:
    BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', '')
    NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'crypto_tops_bottoms')
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

    def fetch_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 200) -> Optional[List]:
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Binance OHLCV error {symbol}: {e}")
            return None

    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Binance ticker error {symbol}: {e}")
            return None

# ======================
# المؤشرات الفنية
# ======================
class IndicatorCalculator:
    @staticmethod
    def sma(prices: List[float], period: int) -> List[Optional[float]]:
        result = []
        for i in range(len(prices)):
            if i < period - 1:
                result.append(None)
            else:
                result.append(sum(prices[i - period + 1:i + 1]) / period)
        return result

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
    def support_resistance_score(high: List[float], low: List[float], close: List[float]) -> float:
        """يعيد درجة قرب السعر من الدعم (قريبة من 1) أو المقاومة (قريبة من 0)"""
        if len(high) < 40:
            return 0.5
        highs = high[-40:]
        lows = low[-40:]
        resistance_candidates = []
        support_candidates = []

        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_candidates.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_candidates.append(lows[i])

        if not resistance_candidates and not support_candidates:
            return 0.5

        current = close[-1]
        closest_resistance = min([r for r in resistance_candidates if r > current], default=None)
        closest_support = max([s for s in support_candidates if s < current], default=None)

        if closest_resistance and closest_support:
            dist_to_res = (closest_resistance - current) / current
            dist_to_sup = (current - closest_support) / current
            if dist_to_sup < 0.02:
                return 0.9
            if dist_to_res < 0.02:
                return 0.1
            if dist_to_sup < 0.05:
                return 0.7
            if dist_to_res < 0.05:
                return 0.3
        elif closest_resistance:
            dist_to_res = (closest_resistance - current) / current
            if dist_to_res < 0.02:
                return 0.1
            if dist_to_res < 0.05:
                return 0.3
        elif closest_support:
            dist_to_sup = (current - closest_support) / current
            if dist_to_sup < 0.02:
                return 0.9
            if dist_to_sup < 0.05:
                return 0.7
        return 0.5

    @staticmethod
    def detect_candlestick_pattern(open_prices: List[float], high_prices: List[float],
                                   low_prices: List[float], close_prices: List[float]) -> Dict[str, bool]:
        """يكشف أنماط الشموع الانعكاسية: shooting star, hammer, engulfing"""
        if len(open_prices) < 2:
            return {}
        patterns = {}
        current = {
            'open': open_prices[-1],
            'high': high_prices[-1],
            'low': low_prices[-1],
            'close': close_prices[-1]
        }
        prev = {
            'open': open_prices[-2],
            'high': high_prices[-2],
            'low': low_prices[-2],
            'close': close_prices[-2]
        }

        body_current = abs(current['close'] - current['open'])
        upper_wick_current = current['high'] - max(current['open'], current['close'])
        lower_wick_current = min(current['open'], current['close']) - current['low']

        # Shooting Star (علوي طويل، جسم صغير، افتتاح قريب من القاع)
        if upper_wick_current > 2 * body_current and lower_wick_current < 0.3 * body_current and current['close'] < current['open']:
            patterns['shooting_star'] = True

        # Hammer (سفلي طويل، جسم صغير، افتتاح قريب من القمة)
        if lower_wick_current > 2 * body_current and upper_wick_current < 0.3 * body_current and current['close'] > current['open']:
            patterns['hammer'] = True

        # Bullish Engulfing (السابق هابط، الحالي صاعد ويبتلع جسم السابق)
        if prev['close'] < prev['open'] and current['close'] > current['open'] and \
           current['open'] < prev['close'] and current['close'] > prev['open']:
            patterns['bullish_engulfing'] = True

        # Bearish Engulfing (السابق صاعد، الحالي هابط ويبتلع جسم السابق)
        if prev['close'] > prev['open'] and current['close'] < current['open'] and \
           current['open'] > prev['close'] and current['close'] < prev['open']:
            patterns['bearish_engulfing'] = True

        return patterns

    @staticmethod
    def volume_spike(volumes: List[float], multiplier: float = 1.5) -> bool:
        """يكشف ما إذا كان الحجم الحالي أعلى من المتوسط بمقدار multiplier"""
        if len(volumes) < 20:
            return False
        avg_vol = sum(volumes[-20:-1]) / 19  # متوسط آخر 19 شمعة (نستثني الحالية)
        current_vol = volumes[-1]
        return current_vol > avg_vol * multiplier

# ======================
# مدير الإشعارات (بدون تغيير)
# ======================
class NotificationManager:
    def __init__(self):
        self.history: List[Notification] = []
        self.max_history = 50
        self.last_notification_time = {}  # مفتاح: (coin_symbol, type)
        self.min_interval = AppConfig.COOLDOWN_SECONDS

    def add(self, notification: Notification):
        self.history.append(notification)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(self, limit: int = 10) -> List[Notification]:
        return self.history[-limit:] if self.history else []

    def should_send(self, coin_symbol: str, signal_type: str, confidence: float) -> bool:
        now = datetime.now()
        key = (coin_symbol, signal_type)
        if key in self.last_notification_time:
            delta = now - self.last_notification_time[key]
            if delta.total_seconds() < self.min_interval:
                return False
        # يمكن إضافة شرط إضافي على الثقة إذا أردت
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
        if not self.should_send(signal.coin_symbol, signal.signal_type, signal.confidence):
            return None

        # إنشاء رسالة بالإنجليزية
        title = f"{signal.signal_type} Detected: {signal.coin_name}"
        message = (
            f"{title}\n"
            f"Confidence: {signal.confidence:.1f}%\n"
            f"Price: ${signal.price:,.2f}\n"
            f"Time: {signal.timestamp.strftime('%H:%M')}\n"
            f"Indicators: {signal.indicators}"
        )

        # اختيار tags و priority بناءً على النوع
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
            key = (signal.coin_symbol, signal.signal_type)
            self.last_notification_time[key] = datetime.now()
            return notification
        return None

# ======================
# المدير الرئيسي لاكتشاف القمم والقيعان
# ======================
class TopBottomDetector:
    def __init__(self):
        self.detections: List[TopBottomSignal] = []
        self.last_update: Optional[datetime] = None
        self.notification_manager = NotificationManager()
        self.binance = BinanceClient()
        self.lock = Lock()

    def update_all(self) -> bool:
        with self.lock:
            logger.info(f"🔄 Scanning {len(AppConfig.COINS)} coins for tops/bottoms...")
            success_count = 0

            for coin in AppConfig.COINS:
                if not coin.enabled:
                    continue
                try:
                    signal = self._scan_coin(coin)
                    if signal:
                        self.detections.append(signal)
                        self.notification_manager.create_notification(signal)
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error on {coin.symbol}: {e}")

            self.last_update = datetime.now()
            # الحفاظ على آخر 100 اكتشاف فقط
            if len(self.detections) > 100:
                self.detections = self.detections[-100:]
            logger.info(f"✅ Found {success_count} potential tops/bottoms")
            return success_count > 0

    def _scan_coin(self, coin: CoinConfig) -> Optional[TopBottomSignal]:
        ohlcv = self.binance.fetch_ohlcv(coin.symbol, '15m', AppConfig.MAX_CANDLES)
        if not ohlcv or len(ohlcv) < 50:
            return None

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
        rsi_vals = IndicatorCalculator.rsi(closes, 14)
        current_rsi = rsi_vals[-1] if rsi_vals[-1] is not None else 50

        sma_20_vals = IndicatorCalculator.sma(closes, 20)
        current_sma_20 = sma_20_vals[-1] if sma_20_vals[-1] is not None else current_price

        support_resistance = IndicatorCalculator.support_resistance_score(highs, lows, closes)

        patterns = IndicatorCalculator.detect_candlestick_pattern(opens, highs, lows, closes)

        volume_spike = IndicatorCalculator.volume_spike(volumes, 1.5)

        # تحليل القمة
        top_confidence = self._calculate_top_confidence(
            current_price, current_sma_20, current_rsi,
            support_resistance, patterns, volume_spike
        )

        # تحليل القاع
        bottom_confidence = self._calculate_bottom_confidence(
            current_price, current_sma_20, current_rsi,
            support_resistance, patterns, volume_spike
        )

        # اختيار الإشارة الأقوى
        signal = None
        if top_confidence >= AppConfig.TOP_CONFIDENCE_THRESHOLD and top_confidence > bottom_confidence:
            signal = TopBottomSignal(
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                signal_type="TOP",
                confidence=top_confidence,
                price=current_price,
                timestamp=datetime.now(),
                indicators={
                    'rsi': current_rsi,
                    'vs_sma20': current_price / current_sma_20 - 1,
                    'support_resistance_score': support_resistance,
                    'patterns': patterns,
                    'volume_spike': volume_spike
                },
                message=f"Top detected with {top_confidence:.1f}% confidence"
            )
        elif bottom_confidence >= AppConfig.BOTTOM_CONFIDENCE_THRESHOLD:
            signal = TopBottomSignal(
                coin_symbol=coin.symbol,
                coin_name=coin.name,
                signal_type="BOTTOM",
                confidence=bottom_confidence,
                price=current_price,
                timestamp=datetime.now(),
                indicators={
                    'rsi': current_rsi,
                    'vs_sma20': current_price / current_sma_20 - 1,
                    'support_resistance_score': support_resistance,
                    'patterns': patterns,
                    'volume_spike': volume_spike
                },
                message=f"Bottom detected with {bottom_confidence:.1f}% confidence"
            )

        return signal

    def _calculate_top_confidence(self, price: float, sma20: float, rsi: float,
                                  sr_score: float, patterns: Dict[str, bool], volume_spike: bool) -> float:
        """حساب درجة الثقة في وجود قمة (0-100)"""
        confidence = 0.0

        # 1. السعر أعلى من SMA20 (اتجاه صاعد)
        if price > sma20:
            confidence += 15

        # 2. RSI فوق 70 (منطقة تشبع شرائي)
        if rsi > 70:
            confidence += 25
        elif rsi > 65:
            confidence += 15

        # 3. قرب المقاومة (sr_score منخفض يعني قرب مقاومة)
        if sr_score < 0.3:
            confidence += 20
        elif sr_score < 0.4:
            confidence += 10

        # 4. أنماط شموع هابطة
        if patterns.get('shooting_star'):
            confidence += 20
        if patterns.get('bearish_engulfing'):
            confidence += 25

        # 5. حجم مرتفع
        if volume_spike:
            confidence += 15

        return min(confidence, 100)

    def _calculate_bottom_confidence(self, price: float, sma20: float, rsi: float,
                                     sr_score: float, patterns: Dict[str, bool], volume_spike: bool) -> float:
        """حساب درجة الثقة في وجود قاع (0-100)"""
        confidence = 0.0

        # 1. السعر أقل من SMA20 (اتجاه هابط)
        if price < sma20:
            confidence += 15

        # 2. RSI تحت 30 (منطقة تشبع بيعي)
        if rsi < 30:
            confidence += 25
        elif rsi < 35:
            confidence += 15

        # 3. قرب الدعم (sr_score مرتفع يعني قرب دعم)
        if sr_score > 0.7:
            confidence += 20
        elif sr_score > 0.6:
            confidence += 10

        # 4. أنماط شموع صاعدة
        if patterns.get('hammer'):
            confidence += 20
        if patterns.get('bullish_engulfing'):
            confidence += 25

        # 5. حجم مرتفع
        if volume_spike:
            confidence += 15

        return min(confidence, 100)

    def get_recent_detections(self, limit: int = 20) -> List[Dict]:
        """إرجاع آخر الاكتشافات بشكل منسق للواجهة"""
        recent = self.detections[-limit:] if self.detections else []
        return [asdict(d) for d in recent]

    def get_stats(self) -> Dict:
        """إحصائيات مبسطة"""
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
app.secret_key = os.environ.get('SECRET_KEY', 'crypto-tops-bottoms-secret')
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
    """صفحة بسيطة تعرض آخر الاكتشافات (يمكن تطويرها لاحقًا)"""
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
    msg = "Test notification - Tops & Bottoms detector is working"
    success = detector.notification_manager.send_ntfy(msg, "Test", "3", "test_tube")
    return jsonify({'success': success})

# ======================
# إشعار بدء التشغيل
# ======================
def send_startup_notification():
    try:
        msg = (
            f"Crypto Tops & Bottoms Detector Started\n"
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
# قالب HTML بسيط (سيتم تحسينه لاحقًا)
# ======================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Tops & Bottoms Detector</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; margin: 20px; background: #111; color: #eee; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #333; padding: 8px; text-align: left; }
        th { background: #222; }
        .TOP { color: #ff6b6b; }
        .BOTTOM { color: #51cf66; }
    </style>
</head>
<body>
    <h1>Crypto Tops & Bottoms Detector</h1>
    <p>Last update: {{ stats.last_update or 'Never' }} | Status: {{ stats.status }}</p>
    <p>Total detections: {{ stats.total_detections }} (Tops: {{ stats.tops }}, Bottoms: {{ stats.bottoms }})</p>

    <h2>Recent Detections</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Coin</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Price</th>
            <th>Indicators</th>
        </tr>
        {% for d in detections %}
        <tr>
            <td>{{ d.timestamp }}</td>
            <td>{{ d.coin_name }} ({{ d.coin_symbol }})</td>
            <td class="{{ d.signal_type }}">{{ d.signal_type }}</td>
            <td>{{ d.confidence|round(1) }}%</td>
            <td>${{ d.price|round(2) }}</td>
            <td><pre>{{ d.indicators }}</pre></td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

# حفظ القالب في ملف (للاستخدام)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

# ======================
# التشغيل الرئيسي
# ======================
if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("🚀 Crypto Tops & Bottoms Detector v1.0")
    logger.info(f"📊 Coins: {len(AppConfig.COINS)}")
    logger.info(f"🔄 Update every {AppConfig.UPDATE_INTERVAL//60} minutes")
    logger.info(f"📢 NTFY: {ExternalAPIConfig.NTFY_URL}")
    logger.info("=" * 50)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
