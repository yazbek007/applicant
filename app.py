import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import pandas_ta as ta  # استبدال ta بـ pandas_ta

# ========== الإعدادات من متغيرات البيئة (تُضبط في Render) ==========
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
INTERVAL = os.getenv('INTERVAL', '1h')
NTFY_TOPIC = os.getenv('NTFY_TOPIC', 'crypto_signals')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))  # ثانية
PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '5.0'))  # %

# أوزان المؤشرات
WEIGHTS = {
    'rsi': float(os.getenv('WEIGHT_RSI', '0.30')),
    'bb': float(os.getenv('WEIGHT_BB', '0.20')),
    'macd': float(os.getenv('WEIGHT_MACD', '0.25')),
    'sr': float(os.getenv('WEIGHT_SR', '0.15')),
    'div': float(os.getenv('WEIGHT_DIV', '0.10'))
}
# ===================================================================

class CryptoSignalBot:
    def __init__(self):
        self.last_signal = None
        self.signal_price = None
        self.signal_direction = None
        self.signal_strength_pct = None
        self.last_notification_time = None

    def get_klines(self, limit=100):
        url = f"https://api.binance.com/api/v3/klines"
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
        # RSI (المدة 14)
        df['rsi'] = ta.rsi(df['close'], length=14)

        # MACD
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd_df], axis=1)  # يضيف أعمدة MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # نعيد تسميتها للاختصار
        df.rename(columns={
            'MACD_12_26_9': 'macd',
            'MACDs_12_26_9': 'macd_signal',
            'MACDh_12_26_9': 'macd_histogram'
        }, inplace=True)

        # Bollinger Bands (20,2)
        bb_df = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bb_df], axis=1)
        df.rename(columns={
            'BBU_20_2.0': 'bb_upper',
            'BBM_20_2.0': 'bb_middle',
            'BBL_20_2.0': 'bb_lower'
        }, inplace=True)

        # الدعم والمقاومة (آخر 20 شمعة)
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
        band_width = upper - lower
        if band_width == 0:
            return 0
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
        dist_to_support = abs(close - support) / close * 100
        dist_to_resistance = abs(close - resistance) / close * 100
        if close <= support * 1.01:
            return 100 - min(100, dist_to_support * 10)
        if close >= resistance * 0.99:
            return 100 - min(100, dist_to_resistance * 10)
        return 0

    def score_divergence(self, df):
        if len(df) < 10:
            return 0
        recent = df.iloc[-5:]
        price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
        rsi_change = recent['rsi'].iloc[-1] - recent['rsi'].iloc[0]
        if price_change < 0 and rsi_change > 5:
            return 100
        if price_change > 0 and rsi_change < -5:
            return 100
        if (price_change < 0 and rsi_change > 2) or (price_change > 0 and rsi_change < -2):
            return 50
        return 0

    def detect_signals(self, df):
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        rsi_score = self.score_rsi(latest['rsi'])
        bb_score = self.score_bb(latest['close'], latest['bb_upper'], latest['bb_lower'])
        macd_score = self.score_macd(latest['macd'], latest['macd_signal'], latest['macd_histogram'], prev['macd_histogram'])
        sr_score = self.score_sr(latest['close'], latest['support'], latest['resistance'])
        div_score = self.score_divergence(df)

        bull_score = 0
        bear_score = 0

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

        if div_score > 50:
            bull_score += div_score * WEIGHTS['div']
        elif div_score > 0:
            bear_score += div_score * WEIGHTS['div']

        total_bull = bull_score
        total_bear = bear_score

        if total_bull > total_bear:
            signal_type = "قاع محتمل"
            strength_pct = total_bull
            reasons = self._build_reasons(rsi_score, bb_score, macd_score, sr_score, div_score, latest)
        elif total_bear > total_bull:
            signal_type = "قمة محتملة"
            strength_pct = total_bear
            reasons = self._build_reasons(rsi_score, bb_score, macd_score, sr_score, div_score, latest)
        else:
            signal_type = None
            strength_pct = 0
            reasons = []

        if strength_pct < 20:
            signal_type = None

        return signal_type, strength_pct, reasons

    def _build_reasons(self, rsi_score, bb_score, macd_score, sr_score, div_score, latest):
        reasons = []
        if rsi_score > 30:
            reasons.append(f"RSI {latest['rsi']:.1f} (قوة {rsi_score:.0f}%)")
        if bb_score > 30:
            reasons.append(f"Bollinger Bands (قوة {bb_score:.0f}%)")
        if macd_score > 30:
            reasons.append(f"MACD (قوة {macd_score:.0f}%)")
        if sr_score > 30:
            reasons.append(f"دعم/مقاومة (قوة {sr_score:.0f}%)")
        if div_score > 30:
            reasons.append(f"دايفرجنس (قوة {div_score:.0f}%)")
        return reasons

    def check_price_update(self, current_price):
        if not self.signal_price:
            return False, 0
        change_percent = ((current_price - self.signal_price) / self.signal_price) * 100
        if self.signal_direction == "قمة" and change_percent >= PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        elif self.signal_direction == "قاع" and change_percent <= -PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        elif abs(change_percent) >= PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        return False, change_percent

    def send_ntfy_notification(self, title, message, tags=[], priority=3):
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        headers = {"Title": title, "Priority": str(priority), "Tags": ",".join(tags)}
        try:
            requests.post(url, data=message.encode('utf-8'), headers=headers)
            print(f"✅ إشعار: {title}")
        except Exception as e:
            print(f"❌ خطإشعار: {e}")

    def run(self):
        print(f"🚀 بوت الكشف عن القمم والقيعان - {SYMBOL} ({INTERVAL})")
        print(f"📱 إشعارات: https://ntfy.sh/{NTFY_TOPIC}")
        print(f"📊 أوزان المؤشرات: {WEIGHTS}")
        print("-" * 50)

        while True:
            try:
                df = self.get_klines(100)
                if df is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                df = self.calculate_indicators(df)
                signal_type, strength_pct, reasons = self.detect_signals(df)

                current_price = df.iloc[-1]['close']
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if signal_type:
                    if self.last_signal != f"{signal_type}_{df.iloc[-1]['timestamp']}":
                        if strength_pct >= 80:
                            priority = 5
                            tags = ["rotating_light", "warning"]
                        elif strength_pct >= 60:
                            priority = 4
                            tags = ["chart_increasing" if "قاع" in signal_type else "chart_decreasing"]
                        elif strength_pct >= 40:
                            priority = 3
                            tags = ["information_source"]
                        else:
                            priority = 2
                            tags = ["grey_question"]

                        title = f"{'🔺' if 'قمة' in signal_type else '🔻'} {signal_type} على {SYMBOL} بقوة {strength_pct:.0f}%"
                        message = f"""
📈 السعر: {current_price:.4f} USDT
⏱ الوقت: {current_time}
💪 القوة الإجمالية: {strength_pct:.1f}%

🔍 تفاصيل المؤشرات:
{chr(10).join(['• ' + r for r in reasons]) if reasons else '• لا توجد أسباب مفصلة'}

⚡ عتبة التحديث: {PRICE_CHANGE_THRESHOLD}%
                        """
                        self.send_ntfy_notification(title, message, tags, priority)

                        self.last_signal = f"{signal_type}_{df.iloc[-1]['timestamp']}"
                        self.signal_price = current_price
                        self.signal_direction = signal_type.split()[0]
                        self.signal_strength_pct = strength_pct
                        self.last_notification_time = current_time

                        print(f"[{current_time}] ✅ إشارة {signal_type} بقوة {strength_pct:.1f}%")

                if self.signal_price:
                    should_update, change_percent = self.check_price_update(current_price)
                    if should_update and self.last_notification_time:
                        last_time = datetime.strptime(self.last_notification_time, "%Y-%m-%d %H:%M:%S")
                        now = datetime.now()
                        if (now - last_time).total_seconds() > 3600:
                            direction = "صعد" if change_percent > 0 else "هبط"
                            title = f"🔄 تحديث {SYMBOL}: {direction} {abs(change_percent):.1f}%"
                            message = f"""
📊 آخر إشارة: {self.signal_direction} بقوة {self.signal_strength_pct:.0f}% @ {self.signal_price:.4f}
💰 السعر الآن: {current_price:.4f} ({change_percent:+.1f}%)
⏱ {current_time}
                            """
                            self.send_ntfy_notification(title, message, ["arrow_up" if change_percent>0 else "arrow_down"], 3)
                            self.last_notification_time = current_time

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 إيقاف البوت")
                break
            except Exception as e:
                print(f"خطأ: {e}")
                time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    bot = CryptoSignalBot()
    bot.run()
