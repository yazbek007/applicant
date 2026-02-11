import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance Futures (حقيقي - غير testnet)
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    BINANCE_BASE_URL = 'https://fapi.binance.com'  # USDⓈ-M Futures
    
    # مفتاح API للمصادقة (يُستخدم للتحقق من الطلبات الواردة)
    EXECUTOR_API_KEY = os.getenv('EXECUTOR_API_KEY')
    
    # إعدادات إضافية
    DEFAULT_LEVERAGE = 1
    ORDER_TIMEOUT = 10  # ثواني
