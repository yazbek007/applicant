from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import Config

class BinanceFuturesTrader:
    def __init__(self):
        self.client = Client(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_API_SECRET
        )
        # استخدام Futures API
        self.client.FUTURES_URL = Config.BINANCE_BASE_URL

    def set_leverage(self, symbol: str, leverage: int):
        """تعيين الرافعة المالية"""
        try:
            return self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except BinanceAPIException as e:
            raise Exception(f"خطأ في تعيين الرافعة: {e.message}")

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                    price: float = None, time_in_force: str = 'GTC'):
        """
        تنفيذ أمر في سوق العقود الآجلة
        side: BUY أو SELL
        order_type: MARKET أو LIMIT
        """
        try:
            if order_type == 'MARKET':
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            else:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    timeInForce=time_in_force,
                    quantity=quantity,
                    price=str(price)
                )
            return order
        except BinanceAPIException as e:
            raise Exception(f"فشل تنفيذ الأمر: {e.message}")

    def place_stop_loss_take_profit(self, symbol: str, side: str, quantity: float,
                                    stop_price: float = None, take_profit_price: float = None):
        """إضافة أوامر وقف الخسارة وجني الأرباح"""
        orders = {}
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        
        if stop_price:
            try:
                sl = self.client.futures_create_order(
                    symbol=symbol,
                    side=opposite_side,
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=str(stop_price),
                    workingType='MARK_PRICE'
                )
                orders['stopLoss'] = sl
            except BinanceAPIException as e:
                raise Exception(f"فشل إعداد وقف الخسارة: {e.message}")
        
        if take_profit_price:
            try:
                tp = self.client.futures_create_order(
                    symbol=symbol,
                    side=opposite_side,
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=str(take_profit_price),
                    workingType='MARK_PRICE'
                )
                orders['takeProfit'] = tp
            except BinanceAPIException as e:
                raise Exception(f"فشل إعداد جني الأرباح: {e.message}")
        
        return orders

    def get_symbol_price(self, symbol: str) -> float:
        """الحصول على سعر الرمز الحالي"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except:
            return 0.0

    def get_exchange_info(self):
        """الحصول على معلومات السوق (لخطوة الكمية)"""
        return self.client.futures_exchange_info()
