from flask import Flask, request, jsonify
from flask_cors import CORS
import hmac
import hashlib
import uuid
from datetime import datetime
from binance_futures import BinanceFuturesTrader
from config import Config

app = Flask(__name__)
CORS(app)

trader = BinanceFuturesTrader()

# تخزين مؤقت للأوامر المجدولة (في الإنتاج استخدم قاعدة بيانات)
scheduled_orders = {}

def verify_api_key(auth_header):
    """التحقق من صحة مفتاح API"""
    if not auth_header:
        return False
    try:
        received_key = auth_header.replace('Bearer ', '')
        return hmac.compare_digest(received_key, Config.EXECUTOR_API_KEY)
    except:
        return False

def save_scheduled_order(order_data, scheduled_time):
    """حفظ أمر مجدول وإرجاع معرف فريد"""
    order_id = str(uuid.uuid4())[:8]  # معرف قصير
    scheduled_orders[order_id] = {
        'data': order_data,
        'scheduled_time': scheduled_time,
        'created_at': datetime.now().isoformat(),
        'status': 'scheduled',
        'executed_at': None,
        'order_result': None
    }
    return order_id

@app.route('/api/execute', methods=['POST'])
def execute_order():
    """نقطة النهاية لتنفيذ الصفقات (فوري أو مجدول)"""
    # 1. التحقق من المصادقة
    if not verify_api_key(request.headers.get('Authorization')):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    data = request.json
    execution_type = data.get('executionType', 'IMMEDIATE')
    scheduled_time = data.get('scheduledTime')
    
    # ===== حالة التنفيذ الآجل (المجدول) =====
    if execution_type == 'SCHEDULED' and scheduled_time:
        try:
            order_id = save_scheduled_order(data, scheduled_time)
            return jsonify({
                'success': True,
                'order_id': order_id,
                'scheduled': True,
                'message': 'تم جدولة الأمر بنجاح',
                'scheduled_time': scheduled_time
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'فشل جدولة الأمر: {str(e)}'}), 500
    
    # ===== حالة التنفيذ الفوري =====
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
            'scheduled': False,
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

@app.route('/api/order/<order_id>', methods=['GET'])
def get_order_status(order_id):
    """الاستعلام عن حالة أمر (فوري أو مجدول)"""
    if not verify_api_key(request.headers.get('Authorization')):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    # البحث في الأوامر المجدولة أولاً
    if order_id in scheduled_orders:
        return jsonify({
            'success': True,
            'scheduled': True,
            'order': scheduled_orders[order_id]
        })
    
    # إذا لم يكن مجدولاً، ابحث في Binance (أمر فوري)
    try:
        symbol = request.args.get('symbol')
        if not symbol:
            return jsonify({'success': False, 'error': 'مطلوب رمز الزوج'}), 400
            
        order = trader.client.futures_get_order(
            symbol=symbol,
            orderId=order_id
        )
        return jsonify({
            'success': True,
            'scheduled': False,
            'order': {
                'orderId': order['orderId'],
                'symbol': order['symbol'],
                'status': order['status'],
                'executedQty': order['executedQty'],
                'avgPrice': order.get('avgPrice', '0'),
                'cumQuote': order.get('cumQuote', '0'),
                'side': order['side'],
                'type': order['type']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'الأمر غير موجود: {str(e)}'}), 404

@app.route('/api/scheduled-orders', methods=['GET'])
def list_scheduled_orders():
    """عرض جميع الأوامر المجدولة"""
    if not verify_api_key(request.headers.get('Authorization')):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    return jsonify({
        'success': True,
        'count': len(scheduled_orders),
        'orders': scheduled_orders
    })

@app.route('/api/cancel-scheduled/<order_id>', methods=['DELETE'])
def cancel_scheduled_order(order_id):
    """إلغاء أمر مجدول"""
    if not verify_api_key(request.headers.get('Authorization')):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    if order_id in scheduled_orders:
        scheduled_orders[order_id]['status'] = 'cancelled'
        del scheduled_orders[order_id]
        return jsonify({'success': True, 'message': 'تم إلغاء الأمر المجدول'})
    else:
        return jsonify({'success': False, 'error': 'الأمر غير موجود'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """فحص صحة الخدمة"""
    return jsonify({
        'status': 'healthy',
        'service': 'crypto-executor',
        'timestamp': datetime.now().isoformat(),
        'scheduled_orders_count': len(scheduled_orders)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
