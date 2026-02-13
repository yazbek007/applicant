import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import ta
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler

# ========== Environment Variables ==========
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
INTERVAL = os.getenv('INTERVAL', '1h')
NTFY_TOPIC = os.getenv('NTFY_TOPIC', 'crypto_signals')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))          # seconds
PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '5.0'))  # %
PORT = int(os.getenv('PORT', '10000'))  # health check port

# Indicator weights
WEIGHTS = {
    'rsi': float(os.getenv('WEIGHT_RSI', '0.30')),
    'bb': float(os.getenv('WEIGHT_BB', '0.20')),
    'macd': float(os.getenv('WEIGHT_MACD', '0.25')),
    'sr': float(os.getenv('WEIGHT_SR', '0.15')),
    'div': float(os.getenv('WEIGHT_DIV', '0.10'))
}
# ============================================

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')
    def log_message(self, format, *args):
        pass  # suppress logs

def run_health_server():
    server = HTTPServer(('0.0.0.0', PORT), HealthCheckHandler)
    print(f"✅ Health check server running on port {PORT}")
    server.serve_forever()

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
            print(f"❌ Error fetching data: {e}")
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
        
        # Support/Resistance (last 20 candles)
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        return df

    # Scoring functions
    def score_rsi(self, rsi_value):
        if pd.isna(rsi_value):
            return 0
        if rsi_value > 70:
            return min(100, (rsi_value - 70) * 5)
        elif rsi_value < 30:
            return min(100, (30 - rsi_value) * 5)
        return 0

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

        # RSI
        if latest['rsi'] < 30:
            bull_score += rsi_score * WEIGHTS['rsi']
        elif latest['rsi'] > 70:
            bear_score += rsi_score * WEIGHTS['rsi']

        # Bollinger
        if latest['close'] <= latest['bb_lower']:
            bull_score += bb_score * WEIGHTS['bb']
        elif latest['close'] >= latest['bb_upper']:
            bear_score += bb_score * WEIGHTS['bb']

        # MACD
        if latest['macd'] > latest['macd_signal'] and prev['macd_histogram'] <= 0:
            bull_score += macd_score * WEIGHTS['macd']
        elif latest['macd'] < latest['macd_signal'] and prev['macd_histogram'] >= 0:
            bear_score += macd_score * WEIGHTS['macd']

        # Support/Resistance
        if latest['close'] <= latest['support'] * 1.01:
            bull_score += sr_score * WEIGHTS['sr']
        elif latest['close'] >= latest['resistance'] * 0.99:
            bear_score += sr_score * WEIGHTS['sr']

        # Divergence
        if div_score > 50:
            bull_score += div_score * WEIGHTS['div']
        elif div_score > 0:
            bear_score += div_score * WEIGHTS['div']

        if bull_score > bear_score:
            signal_type = "Potential Bottom"
            strength_pct = bull_score
        elif bear_score > bull_score:
            signal_type = "Potential Top"
            strength_pct = bear_score
        else:
            signal_type = None
            strength_pct = 0

        # Ignore weak signals (<20%)
        if strength_pct < 20:
            signal_type = None

        return signal_type, strength_pct

    def check_price_update(self, current_price):
        if not self.signal_price:
            return False, 0
        change_percent = ((current_price - self.signal_price) / self.signal_price) * 100
        if self.signal_direction == "Top" and change_percent >= PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        elif self.signal_direction == "Bottom" and change_percent <= -PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        elif abs(change_percent) >= PRICE_CHANGE_THRESHOLD:
            return True, change_percent
        return False, change_percent

    def send_ntfy_notification(self, title, message, tags=[], priority=3):
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        headers = {"Title": title, "Priority": str(priority), "Tags": ",".join(tags)}
        try:
            response = requests.post(url, data=message.encode('utf-8'), headers=headers)
            if response.status_code == 200:
                print(f"✅ Notification sent: {title}")
            else:
                print(f"❌ Failed to send notification: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending notification: {e}")

    def send_startup_notification(self):
        """Send a startup notification to ntfy"""
        title = f"🚀 Bot Started - {SYMBOL}"
        message = f"""
Bot is now running and monitoring {SYMBOL} ({INTERVAL})

⚙️ Settings:
• Interval: {INTERVAL}
• Price change threshold: {PRICE_CHANGE_THRESHOLD}%
• Check interval: {CHECK_INTERVAL}s
• Weights: {WEIGHTS}

You will receive notifications for:
• Strong potential tops/bottoms (strength >20%)
• Price movements >{PRICE_CHANGE_THRESHOLD}% after a signal
        """
        self.send_ntfy_notification(title, message, tags=["rocket"], priority=3)

    def run(self):
        print(f"🚀 Bot started for {SYMBOL} ({INTERVAL})")
        print(f"📱 Notifications to: https://ntfy.sh/{NTFY_TOPIC}")
        print(f"⚡ Price update threshold: {PRICE_CHANGE_THRESHOLD}%")
        print(f"📊 Indicator weights: {WEIGHTS}")
        print("-" * 50)

        # Send startup notification
        self.send_startup_notification()

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

                # New signal
                if signal_type:
                    last_signal_key = f"{signal_type}_{df.iloc[-1]['timestamp']}"
                    if self.last_signal != last_signal_key:
                        # Priority based on strength
                        if strength_pct >= 80:
                            priority = 5
                            tags = ["rotating_light", "warning"]
                        elif strength_pct >= 60:
                            priority = 4
                            tags = ["chart_increasing" if "Bottom" in signal_type else "chart_decreasing"]
                        elif strength_pct >= 40:
                            priority = 3
                            tags = ["information_source"]
                        else:
                            priority = 2
                            tags = ["grey_question"]

                        emoji = "🔺" if "Top" in signal_type else "🔻"
                        title = f"{emoji} {signal_type} on {SYMBOL}"
                        message = f"""
📈 Price: {current_price:.4f} USDT
⏱ Time: {current_time}
💪 Strength: {strength_pct:.1f}%
⚡ Will update if price moves >{PRICE_CHANGE_THRESHOLD}%
                        """
                        self.send_ntfy_notification(title, message, tags, priority)

                        self.last_signal = last_signal_key
                        self.signal_price = current_price
                        self.signal_direction = signal_type.split()[1]  # "Top" or "Bottom"
                        self.signal_strength_pct = strength_pct
                        self.last_notification_time = current_time

                        print(f"[{current_time}] ✅ New signal: {signal_type} with strength {strength_pct:.1f}%")

                # Price update check (5% move)
                if self.signal_price:
                    should_update, change_percent = self.check_price_update(current_price)
                    if should_update and self.last_notification_time:
                        # Avoid spamming: at most once per hour
                        last_time = datetime.strptime(self.last_notification_time, "%Y-%m-%d %H:%M:%S")
                        now = datetime.now()
                        if (now - last_time).total_seconds() > 3600:  # 1 hour
                            direction = "up" if change_percent > 0 else "down"
                            title = f"🔄 Update {SYMBOL}: moved {direction} {abs(change_percent):.1f}%"
                            message = f"""
📊 Last signal: {self.signal_direction} (strength {self.signal_strength_pct:.0f}%) @ {self.signal_price:.4f}
💰 Current price: {current_price:.4f} ({change_percent:+.1f}%)
⏱ {current_time}
                            """
                            self.send_ntfy_notification(title, message, tags=["arrow_up" if change_percent>0 else "arrow_down"], priority=3)
                            self.last_notification_time = current_time

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n🛑 Bot stopped")
                break
            except Exception as e:
                print(f"⚠️ Unexpected error: {e}")
                time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    # Start health check server in a separate thread
    health_thread = Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    # Start the bot
    bot = CryptoSignalBot()
    bot.run()
