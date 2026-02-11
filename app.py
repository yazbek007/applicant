from flask import Flask, request, jsonify
from flask_cors import CORS
import hmac
import hashlib
from binance_futures import BinanceFuturesTrader
from config import Config

app = Flask(__name__)
CORS(app)

trader = BinanceFuturesTrader()

def verify_api_key(auth_header):
    """التحقق من صحة مفتاح API"""
    if not auth_header:
        return False
    try:
        received_key = auth_header.replace('Bearer ', '')
        return hmac.compare_digest(received_key, Config.EXECUTOR_API_KEY)
    except:
        return False

@app.route('/api/execute', methods=['POST'])
def execute_order():
    """نقطة النهاية الوحيدة لتنفيذ الصفقات"""
    # 1. التحقق من المصادقة
    if not verify_api_key(request.headers.get('Authorization')):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    data = request.json
    symbol = data.get('symbol')
    side = data.get('side')
    order_type = data.get('type', 'MARKET')
    quantity = data.get('quantity')
    quantity_type = data.get('quantityType')
    leverage = data.get('leverage', 1)
    stop_loss = data.get('stopLoss')
    stop_loss_type = data.get('stopLossType')
    take_profit = data.get('takeProfit')
    take_profit_type = data.get('takeProfitType')
    time_in_force = data.get('timeInForce', 'GTC')
    price = data.get('price') if order_type == 'LIMIT' else None

    # التحقق من البيانات الأساسية
    if not all([symbol, side, quantity]):
        return jsonify({'success': False, 'error': 'بيانات غير مكتملة'}), 400

    try:
        # تحويل الكمية إذا كانت بقيمة USDT
        if quantity_type == 'quote':
            current_price = trader.get_symbol_price(symbol)
            if current_price <= 0:
                return jsonify({'success': False, 'error': 'لا يمكن الحصول على السعر'}), 400
            quantity = quantity / current_price
            # تقريب الكمية حسب stepSize
            info = trader.get_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            quantity = round(quantity // step_size * step_size, 8)
                            break
                    break

        # تعيين الرافعة
        if leverage > 1:
            trader.set_leverage(symbol, leverage)

        # تنفيذ الأمر الأساسي
        order = trader.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )

        # حساب سعر الدخول
        entry_price = float(order.get('avgPrice', 0)) or float(order.get('price', 0))

        # تحويل وقف الخسارة وجني الأرباح من نسبة مئوية إلى سعر
        sl_price = None
        tp_price = None

        if stop_loss:
            if stop_loss_type == 'percent':
                if side == 'BUY':
                    sl_price = entry_price * (1 - stop_loss / 100)
                else:
                    sl_price = entry_price * (1 + stop_loss / 100)
            else:
                sl_price = stop_loss

        if take_profit:
            if take_profit_type == 'percent':
                if side == 'BUY':
                    tp_price = entry_price * (1 + take_profit / 100)
                else:
                    tp_price = entry_price * (1 - take_profit / 100)
            else:
                tp_price = take_profit

        # إضافة أوامر وقف الخسارة/جني الأرباح
        if sl_price or tp_price:
            trader.place_stop_loss_take_profit(
                symbol=symbol,
                side=side,
                quantity=quantity,
                stop_price=sl_price,
                take_profit_price=tp_price
            )

        return jsonify({
            'success': True,
            'order_id': order['orderId'],
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['executedQty'],
            'price': order['price'],
            'avg_price': order.get('avgPrice', '0'),
            'cost': float(order.get('cumQuote', 0)),
            'status': order['status']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخدمة"""
    return jsonify({'status': 'healthy', 'service': 'crypto-executor'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
