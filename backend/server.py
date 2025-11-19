from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# ----- App Setup -----
app = FastAPI()
api_router = APIRouter(prefix="/api")

executor = ThreadPoolExecutor(max_workers=4)

# ----- Logger -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----- Models -----
class StockSignal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    price: float
    signal: str  # "AL", "SAT", "GÜÇLÜ AL", "TUT"
    signal_strength: float  # 0-100
    conditions_met: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    volume: Optional[float] = None
    change_percent: Optional[float] = None

class ScanRequest(BaseModel):
    symbols: Optional[List[str]] = None

# ----- BIST Symbols -----



BIST_SYMBOLS = [
    "ASELS.IS", "EREGL.IS", "KCHOL.IS", "SAHOL.IS", "PETKM.IS", "SISE.IS", "THYAO.IS",
    "TUPRS.IS", "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "VAKBN.IS", "YKBNK.IS",
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


# ----- Memory Storage -----
latest_scan_results: List[StockSignal] = []

# ----- Indicator Calculations -----
def calculate_kama(close_prices, period=10, fast=2, slow=30):
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

def calculate_wave_trend(high, low, close, n1=10, n2=21):
    ap = (high + low + close) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    return wt1, wt2

def calculate_supertrend(high, low, close, period=10, multiplier=3.5):
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=period)
    atr = atr_indicator.average_true_range()
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    for i in range(1, len(close)):
        if close.iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lowerband.iloc[i]
        else:
            supertrend.iloc[i] = upperband.iloc[i]
    return supertrend, direction

def calculate_nadaraya_watson(close, h=8, mult=3.0):
    window = h * 2 + 1
    weights = np.exp(-0.5 * ((np.arange(-h, h+1) / h) ** 2))
    weights = weights / weights.sum()
    nw = close.rolling(window=window, center=True).apply(
        lambda x: np.sum(x * weights) if len(x) == window else np.nan,
        raw=True
    )
    std = close.rolling(window=window).std()
    upper = nw + mult * std
    lower = nw - mult * std
    return nw, upper, lower

# ----- Stock Analysis -----
def analyze_stock(symbol: str) -> Optional[StockSignal]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d")
        if df.empty or len(df) < 30:
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        ema8 = EMAIndicator(close=close, window=8).ema_indicator()
        ema20 = EMAIndicator(close=close, window=20).ema_indicator()
        kama = calculate_kama(close)
        wt1, wt2 = calculate_wave_trend(high, low, close)
        supertrend, st_direction = calculate_supertrend(high, low, close)
        nw, nw_upper, nw_lower = calculate_nadaraya_watson(close)

        current_price = close.iloc[-1]
        current_ema8 = ema8.iloc[-1]
        current_ema20 = ema20.iloc[-1]
        current_kama = kama.iloc[-1] if not pd.isna(kama.iloc[-1]) else 0
        current_wt1 = wt1.iloc[-1] if not pd.isna(wt1.iloc[-1]) else 0
        current_wt2 = wt2.iloc[-1] if not pd.isna(wt2.iloc[-1]) else 0
        prev_wt1 = wt1.iloc[-2] if len(wt1) > 1 and not pd.isna(wt1.iloc[-2]) else 0
        prev_wt2 = wt2.iloc[-2] if len(wt2) > 1 and not pd.isna(wt2.iloc[-2]) else 0
        current_st_direction = st_direction.iloc[-1] if not pd.isna(st_direction.iloc[-1]) else 0

        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0

        conditions_met = []
        buy_score = 0

        if current_ema8 > current_ema20:
            conditions_met.append("EMA(8) > EMA(20) - Yükseliş Trendi")
            buy_score += 20
        if current_price > current_kama:
            conditions_met.append("Fiyat > KAMA - Dinamik Destek Üstünde")
            buy_score += 20
        if current_wt1 > current_wt2 and prev_wt1 <= prev_wt2:
            conditions_met.append("WT1 WT2'yi Yukarı Kesti - Momentum Değişimi")
            buy_score += 20
        if volume_ratio > 1.15:
            conditions_met.append(f"Hacim {volume_ratio:.1f}x - Güçlü Hacim Onayı")
            buy_score += 20
        if current_st_direction == 1:
            conditions_met.append("Supertrend Yükseliş - Trend Onayı")
            buy_score += 20

        # Sell
        sell_conditions = []
        sell_score = 0
        if current_ema8 < current_ema20:
            sell_conditions.append("EMA(8) < EMA(20) - Düşüş Trendi")
            sell_score += 20
        if current_price < current_kama:
            sell_conditions.append("Fiyat < KAMA - Dinamik Destek Altında")
            sell_score += 20
        if current_wt1 < current_wt2 and prev_wt1 >= prev_wt2:
            sell_conditions.append("WT1 WT2'yi Aşağı Kesti - Negatif Momentum")
            sell_score += 20
        if volume_ratio > 1.15 and len(sell_conditions) > 0:
            sell_conditions.append(f"Hacim {volume_ratio:.1f}x - Satış Hacim Onayı")
            sell_score += 20
        if current_st_direction == -1:
            sell_conditions.append("Supertrend Düşüş - Trend Uyarısı")
            sell_score += 20

        signal = "TUT"
        final_conditions = []
        signal_strength = 0
        if buy_score >= 80:
            if volume_ratio > 1.5:
                signal = "GÜÇLÜ AL"
                signal_strength = min(100, buy_score + 15)
            else:
                signal = "AL"
                signal_strength = buy_score
            final_conditions = conditions_met
        elif sell_score >= 60:
            signal = "SAT"
            signal_strength = sell_score
            final_conditions = sell_conditions
        else:
            signal = "TUT"
            signal_strength = max(buy_score, sell_score) / 2
            final_conditions = ["Belirsiz Durum - Net Sinyal Yok"]

        change_percent = ((current_price - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0

        return StockSignal(
            symbol=symbol.replace(".IS", "").replace(".is", ""),
            price=float(current_price),
            signal=signal,
            signal_strength=float(signal_strength),
            conditions_met=final_conditions,
            volume=float(current_volume),
            change_percent=float(change_percent)
        )

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return None

# ----- API Endpoints -----
@api_router.post("/scan", response_model=List[StockSignal])
async def scan_stocks(request: ScanRequest):
    symbols = request.symbols if request.symbols else BIST_SYMBOLS
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, analyze_stock, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    signals = [s for s in results if s is not None]

    global latest_scan_results
    latest_scan_results = signals  # Memory storage

    return signals

@api_router.get("/latest-signals", response_model=List[StockSignal])
async def get_latest_signals():
    return latest_scan_results

@api_router.get("/")
async def root():
    return {"message": "BIST Stock Scanner API (No DB)"}

# ----- Middleware -----
from starlette.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown():
    executor.shutdown(wait=True)
