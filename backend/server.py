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
    #symbols: Optional[List[str]] = None
    group_name: str   # artık sadece grup adı gönderilecek
# ----- BIST Sembolleri -----
BIST_SYMBOLS = [
    "QNBTR.IS", "ASELS.IS", "GARAN.IS", "ENKAI.IS", "KCHOL.IS", "TUPRS.IS", "THYAO.IS", "ISBTR.IS", "ISCTR.IS", "ISKUR.IS",
    "FROTO.IS", "BIMAS.IS", "AKBNK.IS", "YKBNK.IS", "DSTKF.IS", "VAKBN.IS", "KLRHO.IS", "HALKB.IS", "TCELL.IS", "TTKOM.IS",
    "TERA.IS", "PKENT.IS", "EREGL.IS", "SAHOL.IS", "KENT.IS", "CCOLA.IS", "SASA.IS", "KOZAL.IS", "TURSG.IS", "KLNMA.IS",
    "QNBFK.IS", "TOASO.IS", "GUBRF.IS", "SISE.IS", "OYAKC.IS", "ZRGYO.IS", "PGSUS.IS", "ISDMR.IS", "TAVHL.IS", "ENERY.IS",
    "PASEU.IS", "HEDEF.IS", "DOCO.IS", "MGROS.IS", "MAGEN.IS", "ASTOR.IS", "ENJSA.IS", "AEFES.IS", "UFUK.IS", "EKGYO.IS",
    "TRGYO.IS", "ECILC.IS", "RALYH.IS", "AHGAZ.IS", "BRSAN.IS", "PEKGY.IS", "BRYAT.IS", "AGHOL.IS", "AKSEN.IS", "ARCLK.IS",
    "TABGD.IS", "MPARK.IS", "ISMEN.IS", "POLHO.IS", "GLRMK.IS", "RGYAS.IS", "GENIL.IS", "OTKAR.IS", "TBORG.IS", "TTRAK.IS",
    "LIDER.IS", "SMRVA.IS", "EFOR.IS", "AYGAZ.IS", "KLSER.IS", "DOHOL.IS", "ANSGR.IS", "SELEC.IS", "PETKM.IS", "CIMSA.IS",
    "ULKER.IS", "DOAS.IS", "ANHYT.IS", "ECZYT.IS", "ALARK.IS", "RYGYO.IS", "AKSA.IS", "CLEBI.IS", "KOZAA.IS", "AGESA.IS",
    "RAYSG.IS", "GRTHO.IS", "LYDHO.IS", "INVES.IS", "TSKB.IS", "NUHCM.IS", "YGGYO.IS", "POLTK.IS", "GRSEL.IS", "DAPGM.IS",
    "MAVI.IS", "IEYHO.IS", "CMENT.IS", "KTLEV.IS", "BASGZ.IS", "KRDMA.IS", "KRDMB.IS", "KRDMD.IS", "RYSAS.IS", "CWENE.IS",
    "HEKTS.IS", "SOKM.IS", "BRISA.IS", "TKFEN.IS", "BSOKE.IS", "ODINE.IS", "TEHOL.IS", "KONYA.IS", "AKCNS.IS", "IZENR.IS",
    "LYDYE.IS", "EGEEN.IS", "KLYPV.IS", "CVKMD.IS", "GLYHO.IS", "NTHOL.IS", "BTCIM.IS", "OZKGY.IS", "AVPGY.IS", "KCAER.IS",
    "IPEKE.IS", "MOGAN.IS", "GESAN.IS", "AKFYE.IS", "BALSU.IS", "BFREN.IS", "SKBNK.IS", "BINBN.IS", "ARMGD.IS", "EUPWR.IS",
    "KONTR.IS", "ALBRK.IS", "SNGYO.IS", "ENTRA.IS", "OBAMS.IS", "ISGYO.IS", "GSRAY.IS", "MIATK.IS", "BMSTL.IS", "MRSHL.IS",
    "TATEN.IS", "AKSGY.IS", "SUNTK.IS", "KUYAS.IS", "TRHOL.IS", "BANVT.IS", "ZOREN.IS", "PATEK.IS", "ALFAS.IS", "ARASE.IS",
    "CANTE.IS", "LOGO.IS", "ATATP.IS", "SMRTG.IS", "LILAK.IS", "SARKY.IS", "FZLGY.IS", "FENER.IS", "LMKDC.IS", "ALTNY.IS",
    "ESEN.IS", "HLGYO.IS", "HTTBT.IS", "PSGYO.IS", "CRFSA.IS", "KLKIM.IS", "CEMZY.IS", "ISKPL.IS", "KZBGY.IS", "AKFIS.IS",
    "ASUZU.IS", "EGPRO.IS", "BINHO.IS", "ISFIN.IS", "YEOTK.IS", "JANTS.IS", "AYDEM.IS", "ADGYO.IS", "VESBE.IS", "KSTUR.IS",
    "EUREN.IS", "ENSRI.IS", "ULUSE.IS", "DEVA.IS", "KOTON.IS", "OZATD.IS", "KAYSE.IS", "GWIND.IS", "BULGS.IS", "TMSN.IS",
    "GEDIK.IS", "DOFRB.IS", "VSNMD.IS", "BERA.IS", "OYYAT.IS", "TUKAS.IS", "VERUS.IS", "ICBCT.IS", "YYLGD.IS", "MEGMT.IS",
    "ALKLC.IS", "SONME.IS", "AKGRT.IS", "VAKFN.IS", "BIENY.IS", "DGGYO.IS", "AHSGY.IS", "AKFGY.IS", "BIOEN.IS", "VESTL.IS",
    "ESCAR.IS", "AYCES.IS", "SDTTR.IS", "SRVGY.IS", "GARFA.IS", "GLCVY.IS", "QUAGR.IS", "ECOGR.IS", "INVEO.IS", "TRCAS.IS",
    "EGGUB.IS", "INGRM.IS", "ALCAR.IS", "KORDS.IS", "TSPOR.IS", "IZMDC.IS", "VAKKO.IS", "BUCIM.IS", "BASCM.IS", "VKGYO.IS",
    "KLGYO.IS", "HATSN.IS", "ADEL.IS", "EMKEL.IS", "AKENR.IS", "AGROT.IS", "KBORU.IS", "TNZTP.IS", "BOSSA.IS", "TUREX.IS",
    "KARSN.IS", "TCKRC.IS", "OFSYM.IS", "EBEBK.IS", "ADESE.IS", "GIPTA.IS", "SURGY.IS", "MOBTL.IS", "ALGYO.IS", "BESLR.IS",
    "IZFAS.IS", "AKMGY.IS", "GOZDE.IS", "BJKAS.IS", "A1CAP.IS", "PRKAB.IS", "ODAS.IS", "KAREL.IS", "MNDTR.IS", "GENTS.IS",
    "HRKET.IS", "PARSN.IS", "KOPOL.IS", "GOKNR.IS", "BLUME.IS", "REEDR.IS", "YIGIT.IS", "EKOS.IS", "MOPAS.IS", "ALKA.IS",
    "ASGYO.IS", "NTGAZ.IS", "KMPUR.IS", "TARKM.IS", "ATAKP.IS", "GEREL.IS", "AYEN.IS", "BOBET.IS", "KOCMT.IS", "MAALT.IS",
    "PAGYO.IS", "NATEN.IS", "DOKTA.IS", "BARMA.IS", "KAPLM.IS", "ERCB.IS", "YBTAS.IS", "GMTAS.IS", "IHAAS.IS", "ENDAE.IS",
    "BIGCH.IS", "KGYO.IS", "MERIT.IS", "SNPAM.IS", "KARTN.IS", "BORLS.IS", "TEZOL.IS", "GZNMI.IS", "BIGTK.IS", "DESA.IS",
    "SUWEN.IS", "CGCAM.IS", "IHLAS.IS", "GOLTS.IS", "KRVGD.IS", "KONKA.IS", "INDES.IS", "BORSK.IS", "ORGE.IS", "DARDL.IS",
    "ONCSM.IS", "ISGSY.IS", "KUVVA.IS", "INTEM.IS", "PENTA.IS", "SAFKR.IS", "CATES.IS", "PLTUR.IS", "HOROZ.IS", "CRDFA.IS",
    "AFYON.IS", "ARSAN.IS", "ULUUN.IS", "CEMTS.IS", "LINK.IS", "SEGYO.IS", "FORTE.IS", "YATAS.IS", "EGEGY.IS", "TKNSA.IS",
    "KZGYO.IS", "BIGEN.IS", "ALKIM.IS", "OZYSR.IS", "TSGYO.IS", "ARDYZ.IS", "FMIZP.IS", "MHRGY.IS", "BRKVY.IS", "ORMA.IS",
    "IMASM.IS", "GUNDG.IS", "GSDHO.IS", "DMRGD.IS", "YUNSA.IS", "ALCTL.IS", "ANELE.IS", "AZTEK.IS", "TMPOL.IS", "BEGYO.IS",
    "MACKO.IS", "NETAS.IS", "SOKE.IS", "ELITE.IS", "CEMAS.IS", "ALVES.IS", "USAK.IS", "DYOBY.IS", "GOODY.IS", "MNDRS.IS",
    "EGEPO.IS", "FORMT.IS", "LRSHO.IS", "BAGFS.IS", "ONRYT.IS", "BVSAN.IS", "RUZYE.IS", "KUTPO.IS", "CMBTN.IS", "ERBOS.IS",
    "HDFGS.IS", "INFO.IS", "HURGZ.IS", "DCTTR.IS", "KIMMR.IS", "YAPRK.IS", "SERNT.IS", "KATMR.IS", "PINSU.IS", "SAYAS.IS",
    "HUNER.IS", "PNSUT.IS", "OSMEN.IS", "TURGG.IS", "LKMNH.IS", "EKSUN.IS", "EYGYO.IS", "MEKAG.IS", "KRGYO.IS", "PETUN.IS",
    "PAPIL.IS", "MERCN.IS", "OTTO.IS", "TEKTU.IS", "SEGMN.IS", "DITAS.IS", "MEDTR.IS", "ISSEN.IS", "SANKO.IS", "BURCE.IS",
    "DOFER.IS", "KTSKR.IS", "TATGD.IS", "BLCYT.IS", "KNFRT.IS", "DAGI.IS", "BRLSM.IS", "MRGYO.IS", "TRILC.IS", "ISBIR.IS",
    "NUGYO.IS", "LUKSK.IS", "MARBL.IS", "BAHKM.IS", "PNLSN.IS", "ARTMS.IS", "DZGYO.IS", "MSGYO.IS", "DERHL.IS", "IHLGM.IS",
    "BAKAB.IS", "BEYAZ.IS", "ARENA.IS", "FONET.IS", "TGSAS.IS", "MAKTK.IS", "PAMEL.IS", "GLRYH.IS", "PCILT.IS", "SANFM.IS",
    "METRO.IS", "MTRKS.IS", "CELHA.IS", "SNICA.IS", "SKYLP.IS", "LIDFA.IS", "KRONT.IS", "ANGEN.IS", "PRKME.IS", "DUNYH.IS",
    "CONSE.IS", "OZSUB.IS", "DNISI.IS", "VRGYO.IS", "UNLU.IS", "ESCOM.IS", "EDATA.IS", "INTEK.IS", "KLMSN.IS", "EDIP.IS",
    "BURVA.IS", "KLSYN.IS", "EGSER.IS", "AYES.IS", "DOGUB.IS", "DGATE.IS", "DENGE.IS", "KRSTL.IS", "BMSCH.IS", "ULUFA.IS",
    "ATEKS.IS", "TDGYO.IS", "YGYO.IS", "BIZIM.IS", "DMSAS.IS", "YYAPI.IS", "FRIGO.IS", "DGNMO.IS", "BYDNR.IS", "TLMAN.IS",
    "VBTYZ.IS", "DURDO.IS", "SNKRN.IS", "DERIM.IS", "RTALB.IS", "AGYO.IS", "SKYMD.IS", "VERTU.IS", "MAKIM.IS", "VKING.IS",
    "DURKN.IS", "MARTI.IS", "OSTIM.IS", "KFEIN.IS", "OZGYO.IS", "SUMAS.IS", "SODSN.IS", "EUHOL.IS", "TUCLK.IS", "A1YEN.IS",
    "PKART.IS", "OBASE.IS", "IHGZT.IS", "RUBNS.IS", "YESIL.IS", "MERKO.IS", "BNTAS.IS", "CUSAN.IS", "MANAS.IS", "PENGD.IS",
    "ZEDUR.IS", "RNPOL.IS", "HATEK.IS", "AVHOL.IS", "YAYLA.IS", "YKSLN.IS", "GSDDE.IS", "GLBMD.IS", "KRPLS.IS", "BAYRK.IS",
    "KERVN.IS", "MMCAS.IS", "HKTM.IS", "AVGYO.IS", "GEDZA.IS", "MEPET.IS", "PRDGS.IS", "IZINV.IS", "NIBAS.IS", "SEYKM.IS",
    "FADE.IS", "CEOEM.IS", "BRKO.IS", "EMNIS.IS", "AKSUE.IS", "BALAT.IS", "DESPC.IS", "ACSEL.IS", "COSMO.IS", "EPLAS.IS",
    "YONGA.IS", "PSDTC.IS", "OYAYO.IS", "VANGD.IS", "SKTAS.IS", "IHYAY.IS", "MEGAP.IS", "AVOD.IS", "PRZMA.IS", "SILVR.IS",
    "ETILR.IS", "SELVA.IS", "BRMEN.IS", "KRTEK.IS", "MARKA.IS", "OYLUM.IS", "FLAP.IS", "SEKFK.IS", "IHEVA.IS", "ARZUM.IS",
    "SANEL.IS", "OZRDN.IS", "AKYHO.IS", "EKIZ.IS", "HUBVC.IS", "ULAS.IS", "SMART.IS", "AVTUR.IS", "BRKSN.IS", "SAMAT.IS",
    "MZHLD.IS", "ATAGY.IS", "ERSU.IS", "VKFYO.IS", "ATSYH.IS", "RODRG.IS", "SEKUR.IS", "ETYAT.IS", "CASA.IS", "GRNYO.IS",
    "IDGYO.IS", "ATLAS.IS", "MTRYO.IS", "ORCAY.IS", "EUKYO.IS", "DIRIT.IS", "EUYO.IS"
]



# ----- BIST Grupları -----
BIST_GROUPS = {
    "Bankacılık": ["AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "VAKBNK.IS", "YKBNK.IS", "SKBNK.IS"],
    "Havacılık / Otomotiv": ["THYAO.IS", "TUPRS.IS", "TOASO.IS", "FROTO.IS", "TAVHL.IS"],
    "Gıda / Perakende": ["BIMAS.IS", "AEFES.IS", "ULKER.IS", "MAVI.IS", "GOODY.IS"],
    "Enerji / Petrol": ["PETKM.IS", "SISE.IS", "ECILC.IS", "ENKAI.IS"],
    
    
    "BIST 100": [
  "AEFES.IS","AGHOL.IS","AGROT.IS","AHGAZ.IS","AKBNK.IS","AKSA.IS","AKSEN.IS","ALARK.IS","ALFAS.IS","ALTNY.IS",
  "ANHYT.IS","ANSGR.IS","ARCLK.IS","ARDYZ.IS","ASELS.IS","ASTOR.IS","AVPGY.IS","BERA.IS","BIMAS.IS","BRSAN.IS",
  "BRYAT.IS","BSOKE.IS","BTCIM.IS","CANTE.IS","CCOLA.IS","CIMSA.IS","CLEBI.IS","CWENE.IS","DOAS.IS","DOHOL.IS",
  "ECILC.IS","EFOR.IS","EGEEN.IS","EKGYO.IS","ENERY.IS","ENJSA.IS","ENKAI.IS","EREGL.IS","EUPWR.IS","FROTO.IS",
  "GARAN.IS","GESAN.IS","GOLTS.IS","GRTHO.IS","GSRAY.IS","GUBRF.IS","HALKB.IS","HEKTS.IS","IEYHO.IS","ISCTR.IS",
  "ISMEN.IS","KARSN.IS","KCAER.IS","KCHOL.IS","KONTR.IS","KONYA.IS","KOZAA.IS","KOZAL.IS","KRDMD.IS","KTLEV.IS",
  "LMKDC.IS","MAGEN.IS","MAVI.IS","MGROS.IS","MIATK.IS","MPARK.IS","OBAMS.IS","ODAS.IS","OTKAR.IS","OYAKC.IS",
  "PASEU.IS","PETKM.IS","PGSUS.IS","RALYH.IS","REEDR.IS","RYGYO.IS","SAHOL.IS","SASA.IS","SELEC.IS","SISE.IS",
  "SKBNK.IS","SMRTG.IS","SOKM.IS","TABGD.IS","TAVHL.IS","TCELL.IS","THYAO.IS","TKFEN.IS","TOASO.IS","TSKB.IS",
  "TTKOM.IS","TTRAK.IS","TUPRS.IS","TURSG.IS","ULKER.IS","VAKBN.IS","VESTL.IS","YEOTK.IS","YKBNK.IS","ZOREN.IS"
    ],


    "BIST 50": [
    "AEFES.IS", "AKBNK.IS", "ALARK.IS", "ARCLK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "BRSAN.IS",
    "CCOLA.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS",
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "MAVI.IS", "MGROS.IS", "MIATK.IS", "OYAKC.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "SOKM.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
    "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS",
    "YKBNK.IS", "ZOREN.IS"
    ],

    "BIST 30":[
    "AEFES.IS", "AKBNK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "CIMSA.IS",
    "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS", "HEKTS.IS",
    "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KRDMD.IS", "MGROS.IS", "PETKM.IS",
    "PGSUS.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS",
    "THYAO.IS", "TOASO.IS", "TTKOM.IS", "TUPRS.IS", "ULKER.IS", "YKBNK.IS"
    ],




    "Diğer": [s for s in BIST_SYMBOLS if s not in 
              ["AKBNK.IS","GARAN.IS","HALKB.IS","ISCTR.IS","VAKBNK.IS","YKBNK.IS","SKBNK.IS",
               "THYAO.IS","TUPRS.IS","TOASO.IS","FROTO.IS","TAVHL.IS",
               "BIMAS.IS","AEFES.IS","ULKER.IS","MAVI.IS","GOODY.IS",
               "PETKM.IS","SISE.IS","ECILC.IS","ENKAI.IS"]]
}

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
    group_name = request.group_name
    print(f"DEBUG: group_name = '{group_name}'")  # debug için

    symbols = BIST_GROUPS.get(request.group_name, [])
    if not symbols:
        raise HTTPException(status_code=404, detail="Böyle bir grup bulunamadı")


    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, analyze_stock, symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)
    signals = [s for s in results if s is not None]

    global latest_scan_results
    latest_scan_results = signals
    return signals







from pydantic import BaseModel

class GroupScanRequest(BaseModel):
    group_name: str




@api_router.get("/groups")
async def get_groups():
    """Mevcut BIST gruplarını döndürür"""
    return [{"name": k, "symbols": v} for k,v in BIST_GROUPS.items()]


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