from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# ----- TEKNİK ANALİZ KÜTÜPHANELERİ -----
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# ----- App Setup -----
app = FastAPI(title="BIST Stock Scanner - KAMA/ST/WT/EMA Strategy")
api_router = APIRouter(prefix="/api")

# Paralel işlem için ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

# ----- Logger -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----- Models (Veri Yapıları) -----
class StockSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    price: float
    signal: str  # "AL", "SAT", "GÜÇLÜ AL", "TUT"
    signal_strength: float  # 0-100 (Sinyal Gücü)
    conditions_met: List[str] # Sinyali tetikleyen koşullar
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    volume: Optional[float] = None
    change_percent: Optional[float] = None

class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = None

# ----- BIST Sembolleri -----
BIST_SYMBOLS = [
    "ASELS.IS", "EREGL.IS", "KCHOL.IS", "SAHOL.IS", "PETKM.IS", "SISE.IS", "THYAO.IS",
    "TUPRS.IS", "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "VAKBNK.IS", "YKBNK.IS",
    "BIMAS.IS", "KOZAL.IS", "KOZAA.IS", "TAVHL.IS", "TOASO.IS", "FROTO.IS", "ARCLK.IS",
    "ENKAI.IS", "SODA.IS", "GUBRF.IS", "TTKOM.IS", "ECILC.IS", "ISGYO.IS", "DOHOL.IS",
    "MGROS.IS", "LOGO.IS", "PGSUS.IS", "AEFES.IS", "OTKAR.IS", "VESTL.IS", "ODAS.IS",
    "KORDS.IS", "TSKB.IS", "AKSEN.IS", "ALARK.IS", "SOKM.IS", "ULKER.IS", "BRYAT.IS",
    "AGHOL.IS", "MAVI.IS", "GESAN.IS", "TTRAK.IS", "AKENR.IS", "KLMSN.IS", "BASGZ.IS",
    "SKBNK.IS", "CVKMD.IS", "ASUZU.IS", "EGEEN.IS", "ENJSA.IS", "MPARK.IS", "CANTE.IS",
    "BERA.IS", "AKSA.IS", "CEMTS.IS", "HEKTS.IS", "IZMDC.IS", "TKFEN.IS", "BUCIM.IS",
    "YUNSA.IS", "ZOREN.IS", "GOZDE.IS", "KONTR.IS", "PRKME.IS", "SNGYO.IS", "GENIL.IS",
    "OYAKC.IS", "BAGFS.IS", "KARSN.IS", "KARTN.IS", "NETAS.IS", "BRISA.IS", "VESBE.IS",
    "KSTUR.IS", "ALGYO.IS", "MIATK.IS", "IPEKE.IS", "GSDHO.IS", "INDES.IS", "SNPAM.IS",
    "ANSGR.IS", "TCELL.IS", "CLEBI.IS", "CRFSA.IS", "ALCTL.IS", "BRKSN.IS", "KLKIM.IS",
    "CCOLA.IS", "CIMSA.IS", "ADEL.IS", "DOAS.IS", "GOODY.IS"
]

# ----- Memory Storage (Sonuçları tutmak için) -----
latest_scan_results: List[StockSignal] = []

# ----- Indicator Calculations (Özel İndikatör Hesaplamaları) -----

def calculate_kama(close_prices, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average (KAMA) hesaplar."""
    change = abs(close_prices.diff(period))
    volatility = close_prices.diff().abs().rolling(window=period).sum()
    er = change / volatility
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = pd.Series(index=close_prices.index, dtype=float)
    kama.iloc[period-1] = close_prices.iloc[period-1]
    for i in range(period, len(close_prices)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_prices.iloc[i] - kama.iloc[i-1])
    return kama

def calculate_supertrend(high, low, close, period=10, multiplier=3.5):
    """SuperTrend indikatörünü hesaplar."""
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=period)
    atr = atr_indicator.average_true_range()
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    direction.iloc[0] = 1
    supertrend.iloc[0] = lowerband.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                 lowerband.iloc[i] = lowerband.iloc[i-1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                 upperband.iloc[i] = upperband.iloc[i-1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lowerband.iloc[i]
        else:
            supertrend.iloc[i] = upperband.iloc[i]
    return supertrend, direction

def calculate_wave_trend(high, low, close, n1=10, n2=21):
    """Wave Trend Osilatörünü hesaplar. (Tekrar eklendi)"""
    ap = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    return wt1, wt2

# ----- Stock Analysis (Hisse Senedi Analizi Fonksiyonu) -----
def analyze_stock(symbol: str) -> Optional[StockSignal]:
    """Tek bir hisse senedini analiz eder ve sinyal üretir. KAMA, ST, WT, Hacim, EMA 8/20, EMA 200 kontrol edilir."""
    try:
        ticker = yf.Ticker(symbol)
        # GÜNLÜK VERİ ÇEKİLİYOR
        df = ticker.history(period="6mo")
        
        if df.empty or len(df) < 50:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # --- İndikatör Hesaplamaları (İstenen Kriterler) ---
        # 1. Trend (EMA)
        ema8 = EMAIndicator(close=close, window=8).ema_indicator()
        ema20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema200 = EMAIndicator(close=close, window=200).ema_indicator()
        
        # 2. Özel Göstergeler (KAMA, SuperTrend, WaveTrend)
        kama = calculate_kama(close)
        supertrend, st_direction = calculate_supertrend(high, low, close)
        wt1, wt2 = calculate_wave_trend(high, low, close) # WaveTrend hesaplaması

        # --- Son Değerleri Alma ---
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        
        c_ema8 = ema8.iloc[-1]
        c_ema20 = ema20.iloc[-1]
        c_ema200 = ema200.iloc[-1] if len(df) > 200 else 0
        
        c_kama = kama.iloc[-1]
        c_st_dir = st_direction.iloc[-1]

        # Hacim Oranı
        avg_vol = volume.rolling(window=20).mean().iloc[-1]
        c_vol = volume.iloc[-1]
        vol_ratio = c_vol / avg_vol if avg_vol > 0 else 0

        # --- Puanlama ve Strateji ---
        conditions_met = []
        buy_score = 0
        sell_score = 0

        # 1. WAVETREND KESİŞİM KONTROLÜ (SON 3 GÜN LOOKBACK - 20 Puan)
        wt_cross_today = (wt1.iloc[-1] > wt2.iloc[-1]) and (wt1.iloc[-2] <= wt2.iloc[-2])
        wt_cross_prev1 = (wt1.iloc[-2] > wt2.iloc[-2]) and (wt1.iloc[-3] <= wt2.iloc[-3])
        wt_cross_prev2 = (wt1.iloc[-3] > wt2.iloc[-3]) and (wt1.iloc[-4] <= wt2.iloc[-4])

        if wt_cross_today or wt_cross_prev1 or wt_cross_prev2:
            buy_score += 20
            if wt_cross_today:
                conditions_met.append("WaveTrend Al Kesişimi (Bugün)")
            elif wt_cross_prev1:
                conditions_met.append("WaveTrend Al Kesişimi (Dün)")
            else:
                conditions_met.append("WaveTrend Al Kesişimi (2 Gün Önce)")

        # 2. EMA 8/20 Kesişimi (20 Puan)
        if c_ema8 > c_ema20:
            buy_score += 20
            conditions_met.append("EMA8 > EMA20 (Kısa Vade Pozitif)")
        
        # 3. Fiyat > EMA 200 (Uzun Vadeli Trend - 15 Puan) -> Bu koşul isteğiniz üzerine eklenmiştir.
        if current_price > c_ema200 and c_ema200 > 0:
            buy_score += 15
            conditions_met.append("Fiyat > EMA 200 Trendi") # Metin daha net olacak şekilde güncellendi

        # 4. SuperTrend (20 Puan)
        if c_st_dir == 1:
            buy_score += 20
            conditions_met.append("SuperTrend Pozitif")

        # 5. KAMA (15 Puan)
        if current_price > c_kama:
            buy_score += 15
            conditions_met.append("Fiyat KAMA Üstünde")
        
        # 6. Hacim Artışı (10 Puan)
        if vol_ratio > 1.2:
            buy_score += 10
            conditions_met.append(f"Hacim Artışı ({vol_ratio:.1f}x)")

        # SAT Koşulları (Karşıt Puanlama)
        if c_ema8 < c_ema20: sell_score += 20
        if c_st_dir == -1: sell_score += 25 
        if current_price < c_kama: sell_score += 15
        if wt1.iloc[-1] < wt2.iloc[-1]: sell_score += 15 # WaveTrend Satış

        # KARAR MEKANİZMASI (Max Buy Skor: 100, Max Sell Skor: 75)
        signal = "TUT"
        final_strength = 0
        
        # AL Eşiği: 75 Puan ve üzeri (Max 100)
        if buy_score >= 75:
            # GÜÇLÜ AL: Yüksek skor (90+) ve önemli hacim artışı (1.5x+)
            signal = "GÜÇLÜ AL" if buy_score >= 90 and vol_ratio > 1.5 else "AL"
            final_strength = min(99.9, buy_score) 
        elif sell_score >= 50:
            signal = "SAT"
            final_strength = min(99.9, sell_score)
            if not conditions_met: conditions_met.append("Teknik Göstergeler Negatif")
        else:
            signal = "TUT"
            final_strength = max(buy_score, sell_score) / 2
            if not conditions_met: conditions_met.append("Nötr Görünüm")

        change_percent = ((current_price - prev_price) / prev_price * 100)

        # JSON uyumluluğu için NaN/Inf temizliği
        def safe_float(val):
            if pd.isna(val) or np.isinf(val):
                return 0.0
            return float(val)

        return StockSignal(
            symbol=symbol.replace(".IS", "").replace(".is", ""),
            price=safe_float(current_price),
            signal=signal,
            signal_strength=safe_float(final_strength),
            conditions_met=conditions_met,
            volume=safe_float(c_vol),
            change_percent=safe_float(change_percent)
        )

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# ----- API Endpoints (FastAPI Yönlendirmeleri) -----

@api_router.post("/scan", response_model=List[StockSignal])
async def scan_stocks(request: ScanRequest):
    """Belirtilen veya tüm BIST sembollerini tarar ve sinyal üretir."""
    symbols = request.symbols if request.symbols else BIST_SYMBOLS
    loop = asyncio.get_event_loop()
    # Paralel analiz görevlerini başlat
    tasks = [loop.run_in_executor(executor, analyze_stock, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    signals = [s for s in results if s is not None]

    global latest_scan_results
    latest_scan_results = signals

    return signals

@api_router.get("/latest-signals", response_model=List[StockSignal])
async def get_latest_signals():
    """Son taranan sinyalleri döndürür."""
    return latest_scan_results

@api_router.get("/")
async def root():
    return {"message": "BIST Stock Scanner API v5 (KAMA, ST, WT, Hacim, EMA 8/20, EMA 200 stratejisi aktif)"}

# ----- Middleware (CORS ayarları) -----
from starlette.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Uygulama kapanırken ThreadPool'u kapat
@app.on_event("shutdown")
async def shutdown():
    executor.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    # Uygulamayı 8000 portunda başlat
    uvicorn.run(app, host="0.0.0.0", port=8000)