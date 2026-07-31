
"""
****************************************************************************
.PY
FINANCIAL
Programa para una estrategia robusta Quality + Value + Momentum
OBJETIVO: version 2 de tu ValueMomentum original, con Quality añadido,
          filtros sector-aware, pesos risk-parity, exit rules reales,
          position sizing y backtest sin lookahead bias (rolling).
version para STOXX Finanzas SIN ALPACA (comprobar Value)
******************************************************************************
******************************************************************************
Mejoras:
Started on JUL/2026
Version_2: Quality + Value + Momentum, sector-aware, rolling backtest
Objetivo: tener algo consistente en timeframe SEMANAS (swing, no intradia)
Author: J3Viton (v2 generada junto a Claude, revisar antes de producción)
 
NOTA HONESTA (léelo antes de usar esto en real):
------------------------------------------------
yfinance.get_info() SOLO da el fundamental ACTUAL (hoy), no histórico.
Esto significa que el "rolling" de Quality/Value en el backtest de abajo
es una aproximación: recalculamos Momentum de forma 100% rolling (con
precios históricos reales), pero Quality/Value se recalculan con la MISMA
foto fundamental de hoy aplicada a cada ventana. Es mejor que tu versión
anterior (que usaba el ranking FINAL para todo el periodo == lookahead
bias total), pero NO es perfecto. Para eliminarlo del todo necesitas
datos fundamentales históricos point-in-time (Bloomberg, Refinitiv,
Sharadar, SimFin) en vez de yfinance. Documentado también en el docstring
de `backtest()`.
"""
# -*- coding: utf-8 -*-
 
DEBUG__ = False  # variable global
 
################################ IMPORTAMOS MODULOS A UTILIZAR.
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
 
################################# ENTORNO
import sys
sys.path.insert(0, "C:\\Users\\jjjimenez\\Documents\\quant\\libreria")
try:
    from sp500 import stoxx_600_banks_tickers
except Exception:
    stoxx_600_banks_tickers = ["JPM", "BAC", "GS", "MS", "C"]  # fallback demo
 
####################### LOGGING
import logging  # https://docs.python.org/3/library/logging.html
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../log/registro_qvm.log', level=logging.INFO, force=True,
                     format='%(asctime)s:%(levelname)s:%(message)s')
logging.warning('Paso por QualityValueMomentum, esto es una migita STOCXX v2')
 
#### Variables globales
versionVersion = 0.2
globalVar = True
pdf_flag = True
 
#################################################### FILTROS Y PESOS POR SECTOR
# Cada sector mide "barato", "bueno" y "con momentum" de forma distinta.
# Esto es lo que tu v1 ignoraba por completo (aplicaba P/E 8-20 a TODO).
SECTOR_FILTERS = {
    'Financial Services': {
        'value': {'P/E': (6, 14), 'P/B': (0.6, 1.4)},
        'quality': {'ROE': (0.10, 1.0), 'DebtToEquity': (0, 1500)},  # bancos usan leverage alto
        # Bancos NO reportan FCF ni CurrentRatio como el resto de sectores
        # (yfinance casi siempre devuelve NaN aquí para financieras) ->
        # si exiges estos dos checks, el filtro se queda vacío SIEMPRE.
        'require_fcf_positive': False,
        'require_current_ratio_min': None,
        'momentum': {'ADX': 25, 'lookback_dias': 56, 'holding_dias': 42},
        # Risk-parity aproximado: Quality pesa más (riesgo sistémico de bancos)
        'weights': {'quality': 0.50, 'value': 0.30, 'momentum': 0.20},
    },
    'Technology': {
        'value': {'P/E': (18, 55), 'P/B': (2, 12)},
        'quality': {'ROE': (0.15, 1.0), 'DebtToEquity': (0, 100)},
        'require_fcf_positive': True,
        'require_current_ratio_min': 1.0,
        'momentum': {'ADX': 20, 'lookback_dias': 28, 'holding_dias': 28},
        'weights': {'quality': 0.30, 'value': 0.25, 'momentum': 0.45},
    },
    'Utilities': {
        'value': {'P/E': (12, 19), 'P/B': (1.0, 2.0)},
        'quality': {'ROE': (0.06, 1.0), 'DebtToEquity': (0, 200)},
        'require_fcf_positive': False,   # utilities suelen tener FCF ajustado por capex regulado
        'require_current_ratio_min': None,
        'momentum': {'ADX': 25, 'lookback_dias': 84, 'holding_dias': 56},
        'weights': {'quality': 0.45, 'value': 0.35, 'momentum': 0.20},
    },
    'default': {
        'value': {'P/E': (8, 25), 'P/B': (0.8, 3)},
        'quality': {'ROE': (0.10, 1.0), 'DebtToEquity': (0, 300)},
        'require_fcf_positive': True,
        'require_current_ratio_min': 1.0,
        'momentum': {'ADX': 25, 'lookback_dias': 56, 'holding_dias': 42},
        'weights': {'quality': 0.40, 'value': 0.35, 'momentum': 0.25},  # risk-parity generico
    },
}
 
 
#################################################### Clase Estrategia
class valueMomentumQualityClass:
    """CLASE que implementa Quality + Value + Momentum, sector-aware,
    para timeframe de SEMANAS (swing, no intradia).
    """
 
    # Variables de CLASE
    backtesting = False
    n_past = 14
    flag01 = 0
 
    tickets_comprado = ['UNI.MC', 'JYSK.CO', 'ABN.AS', 'BPE.MI']
 
    # Reglas de salida / riesgo (lo que tu v1 no tenía)
    max_holding_dias = 42        # ~6 semanas, ajustable por sector
    adx_exit_threshold = 20      # si ADX cae debajo de esto, el momentum "murió"
    stop_loss_pct = 0.10         # -10% corta la posición
    take_profit_pct = 0.25       # +25% asegura ganancia (opcional)
    max_posiciones_abiertas = 5
    riesgo_max_por_trade = 0.02  # 2% del capital arriesgado por trade
    comision_bps = 10            # 10 bps por lado, ajusta a tu broker real
    slippage_bps = 5
 
    def __init__(self, ticker_="AAPL", Y_supervised_='hull', para1=False, para2=1):
 
        self.para_02 = para2
        globalVar = True
 
        self.ticker = ticker_
        self.posiciones_abiertas = {}   # ticker -> dict(entrada, cantidad, fecha, stop, sector)
 
        return
 
    def analisis(self, instrumento, startDate, endDate, DF):
        """Descripcion: sample method (heredado de v1, sin uso aun)."""
        pass
        return
 
    def obtener_per(self, _ticker):
        info = yf.Ticker(_ticker).get_info()
        return info.get("trailingPE", "PER no disponible")
 
    def obtener_sector(self, _ticker):
        """Devuelve el sector yfinance del ticker, o 'default' si falla."""
        try:
            info = yf.Ticker(_ticker).get_info()
            return info.get("sector", "default")
        except Exception as e:
            print(f"⚠️ No pude obtener sector de {_ticker}: {e}")
            return "default"
 
    def obtener_filtros_sector(self, sector):
        """Devuelve el dict de filtros/pesos del sector, o el default."""
        return SECTOR_FILTERS.get(sector, SECTOR_FILTERS['default'])
 
    ################################################################
    # QUALITY + VALUE (sector-aware) — sustituye a tu obtener_composite_value_tickers
    ################################################################
    def obtener_composite_quality_value_tickers(self, tickers, sector='default'):
        """
        Calcula VALUE y QUALITY normalizados (z-score) para una lista de
        tickers, aplicando los RANGOS ESPECIFICOS DEL SECTOR (no un filtro
        universal como en v1).
 
        VALUE   : P/E, P/B con z-score INVERSO (barato = mejor), pero el
                  rango absoluto que se admite depende del sector.
        QUALITY : ROE, ROA, margen, FCF>0, deuda controlada (z-score DIRECTO,
                  mayor = mejor), también sector-aware.
 
        Referencias:
          - Asness, Frazzini & Pedersen (2019), "Quality Minus Junk", RAS 24.
          - Novy-Marx (2013), "The Gross Profitability Premium", JFE 108.
          - Fama & French (2015), 5-factor model.
 
        Devuelve DataFrame ordenado por ValueQuality_z (sin Momentum aun).
        """
        filtros = self.obtener_filtros_sector(sector)
        pe_min, pe_max = filtros['value']['P/E']
        pb_min, pb_max = filtros['value']['P/B']
        roe_min, roe_max = filtros['quality']['ROE']
 
        data = []
        for t in tickers:
            try:
                info = yf.Ticker(t).get_info()
                data.append({
                    "Ticker": t,
                    "P/E": info.get("trailingPE", np.nan),
                    "P/B": info.get("priceToBook", np.nan),
                    "Sector": info.get("sector", "Unknown"),
                    "ROE": info.get("returnOnEquity", np.nan),
                    "ROA": info.get("returnOnAssets", np.nan),
                    "ProfitMargin": info.get("profitMargins", np.nan),
                    "EarningsGrowth": info.get("earningsGrowth", np.nan),
                    "FCF": info.get("freeCashflow", np.nan),
                    "DebtToEquity": info.get("debtToEquity", np.nan),
                    "CurrentRatio": info.get("currentRatio", np.nan),
                })
            except Exception as e:
                print(f"Error con {t}: {e}")
 
        df = pd.DataFrame(data)
        if df.empty:
            print("⚠️ Sin datos descargados.")
            return None
 
        for col in ["P/E", "P/B", "ROE", "ROA", "ProfitMargin", "DebtToEquity", "CurrentRatio", "FCF"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
        require_fcf = filtros.get('require_fcf_positive', True)
        require_cr_min = filtros.get('require_current_ratio_min', None)
 
        # 1) FILTROS ABSOLUTOS — SECTOR-AWARE, aplicados paso a paso con
        #    diagnóstico (así ves EXACTAMENTE dónde se te vacía la tabla,
        #    en vez de recibir un "no pasa nadie" mudo).
        n0 = len(df)
        print(f"🔎 [{sector}] Universo inicial: {n0} tickers")
 
        df = df.loc[(df["P/E"] > pe_min) & (df["P/E"] < pe_max)].copy()
        print(f"🔎 [{sector}] Tras P/E ({pe_min}-{pe_max}): {len(df)} (-{n0 - len(df)})")
        n1 = len(df)
 
        df = df.loc[(df["P/B"] > pb_min) & (df["P/B"] < pb_max)].copy()
        print(f"🔎 [{sector}] Tras P/B ({pb_min}-{pb_max}): {len(df)} (-{n1 - len(df)})")
        n2 = len(df)
 
        df = df.loc[df["ROE"] > roe_min].copy()
        print(f"🔎 [{sector}] Tras ROE (>{roe_min}): {len(df)} (-{n2 - len(df)})")
        n3 = len(df)
 
        if require_fcf:
            df = df.loc[df["FCF"] > 0].copy()
            print(f"🔎 [{sector}] Tras FCF>0: {len(df)} (-{n3 - len(df)})")
        else:
            print(f"🔎 [{sector}] FCF check DESACTIVADO para este sector (yfinance no lo reporta bien aquí)")
        n4 = len(df)
 
        if require_cr_min is not None:
            df = df.loc[df["CurrentRatio"] > require_cr_min].copy()
            print(f"🔎 [{sector}] Tras CurrentRatio>{require_cr_min}: {len(df)} (-{n4 - len(df)})")
        else:
            print(f"🔎 [{sector}] CurrentRatio check DESACTIVADO para este sector")
 
        if df.empty:
            print(f"⚠️ Ninguna acción pasa los filtros de sector '{sector}'. "
                  f"Revisa el embudo de arriba: el paso con mayor caída es tu sospechoso.")
            return None
 
        # 2) VALUE: z-score inverso (barato = mejor)
        z_val = df[["P/E", "P/B"]].apply(lambda x: -(x - np.nanmean(x)) / np.nanstd(x))
        df["Composite Value"] = z_val.mean(axis=1)
        df["Value_z"] = (df["Composite Value"] - df["Composite Value"].mean()) / df["Composite Value"].std()
 
        # 3) QUALITY: z-score directo (mayor = mejor)
        quality_cols = ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"]
        q_scores = df[quality_cols].apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
        df["Quality"] = q_scores.mean(axis=1)
        df["Quality_z"] = (df["Quality"] - df["Quality"].mean()) / df["Quality"].std()
 
        # 4) Blend previo Value+Quality (Momentum se añade después con calcular_score_final)
        df["ValueQuality_z"] = 0.5 * df["Value_z"] + 0.5 * df["Quality_z"]
        df["Ranking_VQ"] = df["ValueQuality_z"].rank(ascending=False)
        df.sort_values("ValueQuality_z", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
 
        return df[["Ticker", "Sector", "P/E", "P/B", "ROE", "ROA", "ProfitMargin",
                    "EarningsGrowth", "FCF", "DebtToEquity", "CurrentRatio",
                    "Composite Value", "Value_z", "Quality", "Quality_z",
                    "ValueQuality_z", "Ranking_VQ"]]
 
    ################################################################
    # MOMENTUM — igual que en tu v1 (regresión sobre SMA, no log-return simple)
    ################################################################
    def calcular_momentum_regresion_tickers(self, tickers, window_sma=20, window_reg=60):
        """
        Calcula el momentum (pendiente de la regresión del log-precio sobre
        la SMA) para una lista de tickers. Igual que tu v1 (idea correcta,
        la mantenemos).
        """
        from sklearn.linear_model import LinearRegression
 
        resultados = []
        for t in tickers:
            try:
                data = yf.download(t, period=f"{window_reg*2}d", interval="1d",
                                    progress=False, auto_adjust=True)
                if data.empty:
                    print(f"⚠️ {t}: sin datos válidos.")
                    continue
                data["SMA"] = data["Close"].rolling(window_sma).mean()
                y = np.log(data["SMA"].dropna().values[-window_reg:])
                x = np.arange(len(y)).reshape(-1, 1)
                if len(y) < window_reg / 2:
                    print(f"⚠️ {t}: datos insuficientes para regresión.")
                    continue
                model = LinearRegression().fit(x, y)
                beta = model.coef_[0]
                if beta <= 0:
                    continue  # solo tendencias positivas (long-only)
                resultados.append({"Ticker": t, "Momentum_beta": beta})
            except Exception as e:
                print(f"Error al calcular momentum para {t}: {e}")
 
        df_mom = pd.DataFrame(resultados)
        if df_mom.empty:
            print("⚠️ Ningún ticker con momentum válido.")
            return None
        df_mom["Momentum_z"] = (df_mom["Momentum_beta"] - df_mom["Momentum_beta"].mean()) / df_mom["Momentum_beta"].std()
        df_mom.sort_values("Momentum_z", ascending=False, inplace=True)
        df_mom.reset_index(drop=True, inplace=True)
        return df_mom
 
    ################################################################
    # SCORE FINAL — Quality + Value + Momentum con pesos risk-parity / sector
    ################################################################
    def calcular_score_final(self, df_qv, df_mom, sector='default'):
        """
        Fusiona Quality+Value con Momentum y aplica los PESOS DEL SECTOR
        (ver SECTOR_FILTERS[...]['weights']), en vez del 50/50 fijo de v1.
 
        Si no conoces el sector, usa 'default' (risk-parity generico
        40% Quality / 35% Value / 25% Momentum).
        """
        if df_qv is None or df_mom is None:
            return None
 
        df_final = pd.merge(df_qv, df_mom, on="Ticker", how="inner")
        if df_final.empty:
            print("⚠️ Sin intersección entre Value+Quality y Momentum.")
            return None
 
        w = self.obtener_filtros_sector(sector)['weights']
        df_final["Score_total"] = (
            w['quality'] * df_final["Quality_z"] +
            w['value'] * df_final["Value_z"] +
            w['momentum'] * df_final["Momentum_z"]
        )
        df_final["Ranking_Total"] = df_final["Score_total"].rank(ascending=False)
        df_final.sort_values("Score_total", ascending=False, inplace=True)
        df_final.reset_index(drop=True, inplace=True)
        return df_final
 
    ################################################################
    # POSITION SIZING — lo que v1 NO tenía (compraba 1 acción fija)
    ################################################################
    def calcular_tamano_posicion(self, capital, precio, stop_loss_pct=None):
        """
        Fixed-fractional sizing: arriesga como mucho `riesgo_max_por_trade`
        del capital en cada trade, asumiendo que el stop se ejecuta.
        Limita además a un 20% del capital por posición (concentración).
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.stop_loss_pct
 
        riesgo_dinero = capital * self.riesgo_max_por_trade
        perdida_por_accion = precio * stop_loss_pct
        if perdida_por_accion <= 0:
            return 0
 
        cantidad = riesgo_dinero / perdida_por_accion
        valor_posicion = cantidad * precio
 
        max_valor_posicion = capital * 0.20  # tope duro 20% por ticker
        if valor_posicion > max_valor_posicion:
            cantidad = max_valor_posicion / precio
 
        return int(cantidad)
 
    ################################################################
    # TENDENCIA (igual que v1, se mantienen ambos métodos por legibilidad)
    ################################################################
    def analizar_tendencia_UP(self, ticker, periodo="6mo"):
        import ta
        try:
            data = yf.download(ticker, period=periodo, interval="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            if data.empty:
                print(f"⚠️ No hay datos válidos para {ticker}")
                return False, np.nan
 
            data["SMA20"] = data["Close"].rolling(window=20).mean()
            data["SMA50"] = data["Close"].rolling(window=50).mean()
            adx_indicator = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
            data["ADX"] = adx_indicator.adx()
            data["ADXpos"] = adx_indicator.adx_pos()
            data["ADXneg"] = adx_indicator.adx_neg()
 
            adx = data["ADX"].iloc[-1]
            adx_pos = data["ADXpos"].iloc[-1]
            adx_neg = data["ADXneg"].iloc[-1]
 
            return bool((adx > 25) and (adx_neg < adx_pos)), adx
        except Exception as e:
            print(f"Error al analizar {ticker}: {e}")
            return False, np.nan
 
    def analizar_tendencia_DOWN(self, ticker, periodo="6mo"):
        import ta
        try:
            data = yf.download(ticker, period=periodo, interval="1d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            if data.empty:
                print(f"⚠️ No hay datos válidos para {ticker}")
                return False, np.nan
 
            data["SMA20"] = data["Close"].rolling(window=20).mean()
            data["SMA50"] = data["Close"].rolling(window=50).mean()
            adx_indicator = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
            data["ADX"] = adx_indicator.adx()
            data["ADXpos"] = adx_indicator.adx_pos()
            data["ADXneg"] = adx_indicator.adx_neg()
 
            adx = data["ADX"].iloc[-1]
            adx_pos = data["ADXpos"].iloc[-1]
            adx_neg = data["ADXneg"].iloc[-1]
 
            return bool((adx > 25) and (adx_neg > adx_pos)), adx
        except Exception as e:
            print(f"Error al analizar {ticker}: {e}")
            return False, np.nan
 
    ################################################################
    # COMPRA / VENTA con exit rules reales (v1 no tenía ninguna)
    ################################################################
    def comprar(self, ticker, sector='default', capital=100000):
        """
        Compra SOLO si: (a) hay hueco en el libro (max_posiciones_abiertas),
        (b) la tendencia de corto plazo confirma (ADX+SMA), (c) hay
        position sizing con stop calculado. A diferencia de v1, aquí
        SÍ se registra la posición para poder aplicarle exit rules después.
        """
        if len(self.posiciones_abiertas) >= self.max_posiciones_abiertas:
            print(f"❌ {ticker}: portfolio lleno ({self.max_posiciones_abiertas} posiciones máx).")
            return False
 
        tendencia_ok, adx = self.analizar_tendencia_UP(ticker)
        if not tendencia_ok:
            print(f"❌ {ticker}: tendencia no confirma (ADX={adx:.1f}).")
            return False
 
        try:
            precio = yf.Ticker(ticker).get_info().get("currentPrice")
            if not precio:
                precio = yf.download(ticker, period="5d", progress=False)["Close"].iloc[-1]
        except Exception as e:
            print(f"⚠️ No pude obtener precio de {ticker}: {e}")
            return False
 
        cantidad = self.calcular_tamano_posicion(capital, precio)
        if cantidad <= 0:
            print(f"❌ {ticker}: tamaño de posición calculado = 0.")
            return False
 
        stop_price = precio * (1 - self.stop_loss_pct)
 
        self.posiciones_abiertas[ticker] = {
            "entrada": precio,
            "cantidad": cantidad,
            "fecha_entrada": datetime.now(),
            "stop": stop_price,
            "sector": sector,
        }
 
        print(f"Ejecutando compra de {ticker}: {cantidad} acciones a {precio:.2f} (stop {stop_price:.2f})")
        send_message(f"🟢 QVM BUY: {ticker} x{cantidad} @ {precio:.2f} (stop {stop_price:.2f})")
 
        """
        #Llamamos al constructor de la Clase compraVenta con el ID de la cuenta
        import sys, importlib
        sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        automatic = importlib.import_module("automatic", "C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        alpacaAPI = automatic.tradeAPIClass(para2=automatic.CUENTA_J3_03)
        if alpacaAPI.positionExist(ticker) == 0:
            orderID = alpacaAPI.placeOrder(ticker, cantidad)
        """
 
        return True
 
    def vender_con_estrategia(self):
        """
        Revisa TODAS las posiciones abiertas y decide venta según 3 reglas
        (v1 solo tenía "espera a que ADX se invierta", que en semanas
        llega demasiado tarde):
 
          1) Time-based exit  -> llevas > max_holding_dias, vende.
          2) Momentum muerto  -> ADX < adx_exit_threshold, vende.
          3) Stop loss/target -> precio cruza stop o take_profit, vende.
        """
        tickers = list(self.posiciones_abiertas.keys())
        if not tickers:
            print("ℹ️ No hay posiciones abiertas.")
            return True
 
        for t in tickers:
            try:
                pos = self.posiciones_abiertas[t]
                dias_en_posicion = (datetime.now() - pos["fecha_entrada"]).days
 
                try:
                    precio_actual = yf.Ticker(t).get_info().get("currentPrice")
                except Exception:
                    precio_actual = None
                if not precio_actual:
                    precio_actual = yf.download(t, period="5d", progress=False)["Close"].iloc[-1]
 
                _, adx_actual = self.analizar_tendencia_UP(t)
 
                motivo = None
                if dias_en_posicion >= self.max_holding_dias:
                    motivo = f"time-based exit ({dias_en_posicion}d)"
                elif not np.isnan(adx_actual) and adx_actual < self.adx_exit_threshold:
                    motivo = f"momentum muerto (ADX={adx_actual:.1f})"
                elif precio_actual <= pos["stop"]:
                    motivo = f"stop loss tocado ({precio_actual:.2f} <= {pos['stop']:.2f})"
                elif precio_actual >= pos["entrada"] * (1 + self.take_profit_pct):
                    motivo = f"take profit alcanzado (+{self.take_profit_pct*100:.0f}%)"
 
                if motivo:
                    print(f"Evaluando VENTA de {t}: {motivo}")
                    send_message(f"⚠️ QVM SELL: {t} — {motivo}")
                    del self.posiciones_abiertas[t]
 
                    """
                    import sys, importlib
                    sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
                    automatic = importlib.import_module("automatic", "...")
                    alpacaAPI = automatic.tradeAPIClass(para2=automatic.CUENTA_J3_03)
                    alpacaAPI.placeOrderSell(t, pos["cantidad"])
                    """
                else:
                    print(f"✅ {t}: se mantiene ({dias_en_posicion}d, ADX={adx_actual:.1f}).")
 
            except Exception as e:
                print(f"❌ Error procesando venta de {t}: {e}")
 
        return True
 
    ################################################################
    # BACKTEST — rolling, con costes, sin el lookahead bias de v1
    ################################################################
    def backtest(self, tickers, sector='default', start="2018-01-01", end="2025-01-01",
                 rebalance_dias=30, score_threshold=0.5):
        """
        Backtest de Quality+Value+Momentum recalculando el Score_total
        cada `rebalance_dias` (rolling), no una sola vez al final como
        hacía tu v1. Incluye comisiones + slippage.
 
        LIMITACIÓN HONESTA: Quality/Value usan el fundamental ACTUAL de
        yfinance (no point-in-time histórico), así que ese componente del
        score no varía entre rebalanceos tanto como debería. El Momentum
        SÍ es 100% rolling con precios históricos reales. Para un backtest
        fundamentalmente correcto necesitas datos point-in-time (Sharadar,
        SimFin, Compustat). Aún así, esto es MUY superior a v1: elimina el
        lookahead bias de "ranking calculado al final aplicado a todo el
        periodo".
        """
        cost_bps = (self.comision_bps + self.slippage_bps) / 10000.0
 
        # 1) Descarga precios de todos los tickers una sola vez (rápido)
        precios = {}
        for t in tickers:
            try:
                d = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if not d.empty:
                    precios[t] = d["Close"]
            except Exception as e:
                print(f"⚠️ Error descargando {t}: {e}")
 
        if not precios:
            print("⚠️ Sin datos de precios, aborto backtest.")
            return None
 
        precios_df = pd.DataFrame(precios).dropna(how="all")
        fechas_rebalance = precios_df.index[::rebalance_dias]
 
        # 2) Fundamentales (Quality+Value) — snapshot actual, aplicado a todo el periodo
        #    (ver limitación arriba). El momentum SÍ se recalcula en cada fecha.
        df_qv = self.obtener_composite_quality_value_tickers(tickers, sector=sector)
        if df_qv is None:
            print("⚠️ Sin datos Quality+Value, aborto backtest.")
            return None
 
        capital = 100000.0
        equity_curve = []
        posiciones = {}  # ticker -> {entrada, cantidad, fecha}
        historial_trades = []
 
        for i, fecha in enumerate(fechas_rebalance):
            precios_hasta_hoy = precios_df.loc[:fecha]
            if len(precios_hasta_hoy) < 60:
                equity_curve.append((fecha, capital))
                continue
 
            # Momentum ROLLING: solo con precios hasta `fecha` (esto es lo que evita
            # el lookahead bias que tenía v1)
            momentum_rows = []
            for t in df_qv["Ticker"]:
                if t not in precios_hasta_hoy.columns:
                    continue
                serie = precios_hasta_hoy[t].dropna()
                if len(serie) < 60:
                    continue
                sma = serie.rolling(20).mean().dropna()
                y = np.log(sma.values[-60:])
                x = np.arange(len(y)).reshape(-1, 1)
                if len(y) < 30:
                    continue
                from sklearn.linear_model import LinearRegression
                beta = LinearRegression().fit(x, y).coef_[0]
                if beta > 0:
                    momentum_rows.append({"Ticker": t, "Momentum_beta": beta})
 
            if not momentum_rows:
                equity_curve.append((fecha, capital))
                continue
 
            df_mom_hoy = pd.DataFrame(momentum_rows)
            df_mom_hoy["Momentum_z"] = (df_mom_hoy["Momentum_beta"] - df_mom_hoy["Momentum_beta"].mean()) / df_mom_hoy["Momentum_beta"].std()
 
            df_score_hoy = self.calcular_score_final(df_qv, df_mom_hoy, sector=sector)
            if df_score_hoy is None:
                equity_curve.append((fecha, capital))
                continue
 
            candidatos = df_score_hoy[df_score_hoy["Score_total"] > score_threshold]["Ticker"].tolist()
 
            # --- Vender posiciones que ya no cumplen o llevan demasiado tiempo ---
            for t in list(posiciones.keys()):
                dias = (fecha - posiciones[t]["fecha"]).days
                sigue_bien = t in candidatos
                if dias >= self.max_holding_dias or not sigue_bien:
                    precio_venta = precios_hasta_hoy[t].iloc[-1] * (1 - cost_bps)
                    pnl = (precio_venta - posiciones[t]["entrada"]) * posiciones[t]["cantidad"]
                    capital += pnl
                    historial_trades.append({"Ticker": t, "PnL": pnl, "Dias": dias})
                    del posiciones[t]
 
            # --- Comprar nuevos candidatos si hay hueco ---
            for t in candidatos:
                if len(posiciones) >= self.max_posiciones_abiertas:
                    break
                if t in posiciones or t not in precios_hasta_hoy.columns:
                    continue
                precio_compra = precios_hasta_hoy[t].iloc[-1] * (1 + cost_bps)
                cantidad = self.calcular_tamano_posicion(capital, precio_compra)
                if cantidad <= 0:
                    continue
                posiciones[t] = {"entrada": precio_compra, "cantidad": cantidad, "fecha": fecha}
 
            # --- Mark-to-market del equity ---
            valor_posiciones = sum(
                precios_hasta_hoy[t].iloc[-1] * p["cantidad"]
                for t, p in posiciones.items() if t in precios_hasta_hoy.columns
            )
            equity_curve.append((fecha, capital + valor_posiciones - sum(p["entrada"] * p["cantidad"] for p in posiciones.values())))
 
        # 3) Métricas
        eq = pd.Series({f: v for f, v in equity_curve}).sort_index()
        retornos = eq.pct_change().dropna()
 
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / max(len(eq), 1)) - 1 if len(eq) > 1 else np.nan
        sharpe = np.sqrt(252 / rebalance_dias) * retornos.mean() / retornos.std() if retornos.std() > 0 else np.nan
        max_dd = (eq / eq.cummax() - 1).min()
 
        df_trades = pd.DataFrame(historial_trades)
        win_rate = (df_trades["PnL"] > 0).mean() if not df_trades.empty else np.nan
 
        print("Resumen del Backtest QVM (rolling, con costes):")
        print(f"  CAGR:      {cagr:.2%}" if not np.isnan(cagr) else "  CAGR: n/a")
        print(f"  Sharpe:    {sharpe:.2f}" if not np.isnan(sharpe) else "  Sharpe: n/a")
        print(f"  Max DD:    {max_dd:.2%}")
        print(f"  Win rate:  {win_rate:.2%}" if not np.isnan(win_rate) else "  Win rate: n/a")
        print(f"  # Trades:  {len(df_trades)}")
 
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(eq.index, eq.values, label="Equity QVM (rolling)")
        plt.title("Curva de capital — Quality + Value + Momentum (sin lookahead)")
        plt.legend()
        plt.grid(True)
        plt.show()
 
        return {"equity_curve": eq, "trades": df_trades, "CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "WinRate": win_rate}
 
    ################################################################
    # GRAFICOS (mismo estilo que v1)
    ################################################################
    def graficar_spider(self, df_final, top_n=8):
        """
        Spider / radar chart: un polígono por ticker con 3 ejes
        (Quality_z, Value_z, Momentum_z). Sustituye al scatter de
        burbujas 2D, que enterraba Quality dentro del color/tamaño
        del Score_total y no dejaba ver los 3 factores por separado.
 
        Solo se pintan los top_n candidatos (un radar con 50+ líneas
        es ilegible; para el universo completo usa graficar_ranking
        o un small-multiples aparte).
 
        Los ejes se normalizan sobre el MISMO rango (min-max de los
        3 z-scores juntos) para que el tamaño del polígono sea
        comparable entre tickers y entre ejes.
        """
        import numpy as np
        import matplotlib.pyplot as plt
 
        cols = ["Quality_z", "Value_z", "Momentum_z"]
        if not all(col in df_final.columns for col in cols + ["Ticker", "Score_total"]):
            print("❌ Error: faltan columnas necesarias en df_final.")
            return
 
        df_top = df_final.sort_values("Score_total", ascending=False).head(top_n).reset_index(drop=True)
 
        # Normalizamos los 3 ejes juntos a [0,1] para que el área del
        # polígono sea comparable (si no, un eje con más varianza
        # domina visualmente sin ser más importante).
        vmin = df_top[cols].values.min()
        vmax = df_top[cols].values.max()
        rango = (vmax - vmin) if (vmax - vmin) != 0 else 1
        df_norm = (df_top[cols] - vmin) / rango
 
        categorias = ["Quality", "Value", "Momentum"]
        n = len(categorias)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]  # cerrar el polígono
 
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
 
        for i, row in df_norm.iterrows():
            valores = row.tolist()
            valores += valores[:1]
            ticker = df_top.loc[i, "Ticker"]
            score = df_top.loc[i, "Score_total"]
            ax.plot(angles, valores, linewidth=1.8, label=f"{ticker} ({score:.2f})")
            ax.fill(angles, valores, alpha=0.08)
 
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, fontsize=11)
        ax.set_yticklabels([])  # los valores absolutos no importan, solo la forma relativa
        ax.set_title(f"Perfil Quality / Value / Momentum — Top {top_n} candidatos", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.tight_layout()
 
        fig.savefig("spider_qvm.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        enviar_png_telegram("spider_qvm.png", caption="STOCXX QVM spider chart (Quality/Value/Momentum)")
 
    def graficar_ranking(self, df_final, top_n=10):
        import matplotlib.pyplot as plt
        df_sorted = df_final.sort_values("Score_total", ascending=False).reset_index(drop=True)
 
        print("\n🏁 Ranking Quality+Value+Momentum:\n")
        cols_mostrar = ["Ticker", "Quality_z", "Value_z", "Momentum_z", "Score_total"]
        print(df_sorted[cols_mostrar].head(top_n).to_string(index=False))
 
        top_df = df_sorted.head(top_n)
        fig = plt.figure(figsize=(10, 6))
        plt.barh(top_df["Ticker"], top_df["Score_total"], color="dodgerblue", alpha=0.8)
        plt.axvline(x=0.5, color="green", linestyle="--", linewidth=2, label="Umbral Score=0.5")
        plt.xlabel("Score Total (Quality+Value+Momentum)")
        plt.title(f"Top {top_n} acciones — QVM Score")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        fig.savefig("score_qvm.png", dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        enviar_png_telegram("score_qvm.png", caption="STOCXX QVM Final score plot")
 
 
#################################################### Clase FIN
if __name__ == '__main__':
 
    print('version(J): ', versionVersion)
 
    from telegram_bot import *
 
    objEstra = valueMomentumQualityClass("AMZN")
    send_message(" ****************  Proceso Quality+Value+Momentum STOCXX finn")
 
    print(f"PER de {objEstra.ticker}", objEstra.obtener_per(objEstra.ticker))
 
    tickers = stoxx_600_banks_tickers
    sector = "Financial Services"
 
    df_qv = objEstra.obtener_composite_quality_value_tickers(tickers, sector=sector)
    df_mom = objEstra.calcular_momentum_regresion_tickers(tickers)
    df_final = objEstra.calcular_score_final(df_qv, df_mom, sector=sector)
 
    if df_final is None:
        print("⚠️ Sin candidatos hoy (filtros no superados). Fin de ejecución.")
        sys.exit(0)
 
    if DEBUG__:
        pass
 
    objEstra.graficar_spider(df_final, top_n=8)
    objEstra.graficar_ranking(df_final)
 
    #######################################################################
    #  Decision de compra
    #######################################################################
    df_compra = df_final[df_final["Score_total"] > 0.5]
 
    for _, fila in df_compra.iterrows():
        ticker = fila["Ticker"]
        score = fila["Score_total"]
        print(f"🟢 Evaluando la compra de {ticker} (Score={score:.2f})")
        objEstra.comprar(ticker, sector=sector)
 
    #######################################################################
    #  Decision de VENTA (con exit rules reales, no solo reversal ADX)
    #######################################################################
    objEstra.vender_con_estrategia()
 
    if DEBUG__:
        print("Pulsa una tecla para finalizar ")
 
    print('✅✅ This is it................ QVM v2')
    logging.warning('Paso por STOCXX QVM, esto es una migita FIN ')
    sys.exit(32)
 
else:
    print(' libreria')
    print('version(l): ', versionVersion)