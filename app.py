import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import ta  # استخدام المكتبة القديمة الموثوقة

# ========== الإعدادات ==========
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
INTERVAL = os.getenv('INTERVAL', '1h')
NTFY_TOPIC = os.getenv('NTFY_TOPIC', 'crypto_signals')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))
PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '5.0'))

# أوزان المؤشرات
WEIGHTS = {
    'rsi': float(os.getenv('WEIGHT_RSI', '0.30')),
    'bb': float(os.getenv('WEIGHT_BB', '0.20')),
    'macd': float(os.getenv('WEIGHT_MACD', '0.25')),
    'sr': float(os.getenv('WEIGHT_SR', '0.15')),
    'div': float(os.getenv('WEIGHT_DIV', '0.10'))
}
# ===============================

class CryptoSignalBot:
    def __init__(self):
        self.last_signal = None
        self.signal_price = None
        self.signal_direction = None
        self.signal_strength_pct = None
        self.last_notification_time = None

    def get_klines(self, limit=100):
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': SYMBOL, 'interval': INTERVAL, 'limit': limit}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"خطأ في جلب البيانات: {e}")
            return None

    def calculate_indicators(self, df):
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # الدعم والمقاومة
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        return df

    def score_rsi(self, rsi_value):
        if pd.isna(rsi_value):
            return 0
        if rsi_value > 70:
            score = min(100, (rsi_value - 70) * 5)
        elif rsi_value < 30:
            score = min(100, (30 - rsi_value) * 5)
        else:
            score = 0
        return score

    def score_bb(self, close, upper, lower):
        if pd.isna(upper) or pd.isna(lower):
            return 0
        if close >= upper:
            return 100
        if close <= lower:
            return 100
        if close >= upper * 0.99:
            return 70
        if close <= lower * 1.01:
            return 70
        return 0

    def score_macd(self, macd, signal, histogram, prev_hist):
        if pd.isna(macd) or pd.isna(signal):
            return 0
        score = 0
        if macd > signal and prev_hist <= 0:
            score += 60
        elif macd < signal and prev_hist >= 0:
            score += 60
        hist_strength = abs(histogram) / (abs(macd) + 0.001) * 100
        score += min(40, hist_strength)
        return min(100, score)

    def score_sr(self, close, support, resistance):
        if pd.isna(support) or pd.isna(resistance):
            return 0
        if close <= support * 1.01:
            return 100
        if close >= resistance * 0.99:
            return 100
        return 0

    def detect_signals(self, df):
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        rsi_score = self.score_rsi(latest['rsi'])
        bb_score = self.score_bb(latest['close'], latest['bb_upper'], latest['bb_lower'])
        macd_score = self.score_macd(latest['macd'], latest['macd_signal'], latest['macd_histogram'], prev['macd_histogram'])
        sr_score = self.score_sr(latest['close'], latest['support'], latest['resistance'])

        bull_score = 0
        bear_score = 0

        # تجميع النقاط
        if latest['rsi'] < 30:
            bull_score += rsi_score * WEIGHTS['rsi']
        elif latest['rsi'] > 70:
            bear_score += rsi_score * WEIGHTS['rsi']

        if latest['close'] <= latest['bb_lower']:
            bull_score += bb_score * WEIGHTS['bb']
        elif latest['close'] >= latest['bb_upper']:
            bear_score += bb_score * WEIGHTS['bb']

        if latest['macd'] > latest['macd_signal'] and prev['macd_histogram'] <= 0:
            bull_score += macd_score * WEIGHTS['macd']
        elif latest['macd'] < latest['macd_signal'] and prev['macd_histogram'] >= 0:
            bear_score += macd_score * WEIGHTS['macd']

        if latest['close'] <= latest['support'] * 1.01:
            bull_score += sr_score * WEIGHTS['sr']
        elif latest['close'] >= latest['resistance'] * 0.99:
            bear_score += sr_score * WEIGHTS['sr']

        if bull_score > bear_score:
            signal_type = "قاع محتمل"
            strength_pct = bull_score
        elif bear_score > bull_score:
            signal_type = "قمة محتملة"
            strength_pct = bear_score
        else:
            signal_type = None
            strength_pct = 0

        if strength_pct < 20:
            signal_type = None

        return signal_type, strength_pct

    def send_ntfy_notification(self, title, message, tags=[], priority=3):
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        headers = {"Title": title, "Priority": str(priority), "Tags": ",".join(tags)}
        try:
            requests.post(url, data=message.encode('utf-8'), headers=headers)
            print(f"✅ إشعار: {title}")
        except Exception as e:
            print(f"❌ خطأ في الإشعار: {e}")

    def run(self):
        print(f"🚀 بوت الكشف عن القمم والقيعان - {SYMBOL} ({INTERVAL})")
        print(f"📱 إشعارات: https://ntfy.sh/{NTFY_TOPIC}")
        print("-" * 50)

        while True:
            try:
                df = self.get_klines()
                if df is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                df = self.calculate_indicators(df)
                signal_type, strength_pct = self.detect_signals(df)

                current_price = df.iloc[-1]['close']
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if signal_type:
                    title = f"{'🔺' if 'قمة' in signal_type else '🔻'} {signal_type} على {SYMBOL}"
                    message = f"""
📈 السعر: {current_price:.4f} USDT
⏱ الوقت: {current_time}
💪 القوة: {strength_pct:.1f}%
                    """
                    self.send_ntfy_notification(title, message, ["warning"], 4)
                    print(f"[{current_time}] ✅ {signal_type} بقوة {strength_pct:.1f}%")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"خطأ: {e}")
                time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    bot = CryptoSignalBot()
    bot.run()
