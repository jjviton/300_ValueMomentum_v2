# -*- coding: utf-8 -*-


"""
*****************************************************************************
.PY

Programa para un estrategia robusta y basica
OBEJTIVO: tener algo consistente que me anime en otras estrategias mas arriesgadas

verion para Defensa
******************************************************************************
******************************************************************************


Mejoras:    

Started on DIC/2022
Version_1: 

Objetivo: 

Author: J3Viton

"""





# -*- coding: utf-8 -*-


"""
******************************************************************************
Clase que implementa una 
 
******************************************************************************
******************************************************************************

Mejoras:    

Started on DIC/2022
Version_1: 

Objetivo: 

Author: J3Viton

"""

DEBUG__ = False  #variable global (global J3_DEBUG__ )


################################ IMPORTAMOS MODULOS A UTILIZAR.
import pandas as pd
import numpy as np

import yfinance as yf


################################# ENTORNO
import sys
sys.path.insert(0,"C:\\Users\\jjjimenez\\Documents\\quant\\libreria")
from sp500 import tickers_financials, defense_tickers


####################### LOGGING
import logging    #https://docs.python.org/3/library/logging.html
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../log/registro.log', level=logging.INFO ,force=True,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.warning('Paso por ValueMomentum, esto es una migita')

#### Variables globales  (refereniarlas con 'global' desde el codigo
versionVersion = 0.1
globalVar  = True
pdf_flag = True

#################################################### Clase Estrategia 



class valueMomentumClass:

    """CLASE xxx

       
    """  
    
    #Variable de CLASE
    backtesting = False  #variable de la clase, se accede con el nombre
    n_past = 14  # Number of past days we want to use to predict the future.  FILAS
    flag01 =0
   
    def __init__(self, ticker_= "AAPL", Y_supervised_ = 'hull', para1=False, para2=1):
        
        #Variable de INSTANCIA
        self.para_02 = para2   #variable de la isntancia
        
        globalVar = True
        #intance.flag01 =True
        
        
        self.ticker = ticker_ 
        
        return

        
    def analisis(self, instrumento, startDate, endDate, DF):
        """
        Descripcion: sample method
        
        Parameters
        ----------
        beneficio : TYPE
            DESCRIPTION.

        Returns
        -------


        """
        pass
   
        return
    
    
        
    def obtener_per(self, _ticker):
        
        info = yf.Ticker(_ticker).get_info()                
      
        return info.get("trailingPE", "PER no disponible")
    

      
    
    def obtener_composite_value_tickers(self, tickers, sector='defensa'):
        """
        Calcula un Composite Value adaptado a empresas de defensa (Aerospace & Defense):
          - EV/EBITDA (LTM)
          - FCF Yield (LTM)
          - Forward P/E
          - P/B
          - Dividend Yield (opcional)
          - Net Debt/EBITDA (opcional si hay datos)
        Devuelve un DataFrame ordenado (Composite Value alto = más "barata"/atractiva en value).
        """
    
        import math
        import numpy as np
        import pandas as pd
        import yfinance as yf
    
        # ----------------------------
        # Helpers
        # ----------------------------
        def to_float(x):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                return float(x)
            except Exception:
                return np.nan
    
        def safe_div(a, b):
            try:
                if b is None or pd.isna(b) or b == 0:
                    return np.nan
                return a / b
            except Exception:
                return np.nan
    
        def pick_first_key(df_index, candidates):
            """Devuelve la primera clave encontrada en el índice de un DataFrame."""
            for k in candidates:
                if k in df_index:
                    return k
            return None
    
        def winsorize_series(s, low=0.05, high=0.95):
            s = s.astype(float)
            q_low, q_high = s.quantile([low, high])
            return s.clip(q_low, q_high)
    
        # ----------------------------
        # Cálculos financieros (LTM)
        # ----------------------------
        def get_basic_info(t: yf.Ticker) -> dict:
            out = {}
            # fast_info: market cap/price si está disponible
            try:
                fi = t.fast_info
                out["market_cap"] = to_float(getattr(fi, "market_cap", np.nan))
                out["last_price"] = to_float(getattr(fi, "last_price", np.nan))
                out["shares"] = to_float(getattr(fi, "shares", np.nan))
            except Exception:
                pass
            # info: EV, P/E, P/B, dividend yield, sector/industry
            try:
                info = t.get_info()
                out["trailing_pe"] = to_float(info.get("trailingPE"))
                out["forward_pe"]  = to_float(info.get("forwardPE"))
                out["price_to_book"] = to_float(info.get("priceToBook"))
                out["dividend_yield"] = to_float(info.get("dividendYield"))  # fracción (0.02 = 2%)
                out["enterprise_value"] = to_float(info.get("enterpriseValue"))
                out["sector"] = info.get("sector", None)
                out["industry"] = info.get("industry", None)
                if pd.isna(out.get("market_cap")):
                    out["market_cap"] = to_float(info.get("marketCap"))
            except Exception:
                pass
            return out
    
        def get_ebitda_ltm(t: yf.Ticker) -> float:
            """EBITDA LTM sumando últimos 4 trimestres; si falta, usa último anual."""
            ebitda = np.nan
            try:
                inc_q = t.get_income_stmt(freq="quarterly")
                if isinstance(inc_q, pd.DataFrame) and not inc_q.empty:
                    key = pick_first_key(inc_q.index, {"EBITDA", "Ebitda"})
                    if key:
                        ebitda = to_float(inc_q.loc[key].iloc[:4].sum())
            except Exception:
                pass
            if pd.isna(ebitda):
                try:
                    inc_a = t.get_income_stmt(freq="annual")
                    if isinstance(inc_a, pd.DataFrame) and not inc_a.empty:
                        key = pick_first_key(inc_a.index, {"EBITDA", "Ebitda"})
                        if key:
                            ebitda = to_float(inc_a.loc[key].iloc[0])
                except Exception:
                    pass
            return ebitda
    
        def get_fcf_ltm(t: yf.Ticker) -> float:
            """FCF LTM = Operating Cash Flow - CapEx (4 últimos trimestres o último anual)."""
            ocf_keys = {"Operating Cash Flow", "Total Cash From Operating Activities", "OperatingCashFlow"}
            capex_keys = {"Capital Expenditures", "CapitalExpenditures","CapitalExpenditure"}
    
            ocf, capex = np.nan, np.nan
            # Trimestral
            try:
                cf_q = t.get_cashflow(freq="quarterly")
                if isinstance(cf_q, pd.DataFrame) and not cf_q.empty:
                    ocf_key = pick_first_key(cf_q.index, ocf_keys)
                    capex_key = pick_first_key(cf_q.index, capex_keys)
                    if ocf_key and capex_key:
                        ocf = to_float(cf_q.loc[ocf_key].iloc[:4].sum())
                        capex = to_float(cf_q.loc[capex_key].iloc[:4].sum())
            except Exception:
                pass
            # Anual si falta
            if pd.isna(ocf) or pd.isna(capex):
                try:
                    cf_a = t.get_cashflow(freq="annual")
                    if isinstance(cf_a, pd.DataFrame) and not cf_a.empty:
                        ocf_key = pick_first_key(cf_a.index, ocf_keys)
                        capex_key = pick_first_key(cf_a.index, capex_keys)
                        if ocf_key and capex_key:
                            ocf = to_float(cf_a.loc[ocf_key].iloc[0])
                            capex = to_float(cf_a.loc[capex_key].iloc[0])
                except Exception:
                    pass
    
            if pd.isna(ocf) or pd.isna(capex):
                return np.nan
            # CapEx suele estar negativo; restamos su valor absoluto para obtener gasto
            return ocf - abs(capex)
    
        def get_net_debt_ebitda(t: yf.Ticker, ebitda_ltm: float) -> float:
            """Net Debt / EBITDA (opcional). Si faltan datos, devuelve NaN."""
            if pd.isna(ebitda_ltm) or ebitda_ltm == 0:
                return np.nan
            try:
                bs_q = t.get_balance_sheet(freq="quarterly")
                if isinstance(bs_q, pd.DataFrame) and not bs_q.empty:
                    # Intento robusto: Total Debt o suma de Short+Long Term Debt
                    debt_key = pick_first_key(bs_q.index, {
                        "Total Debt", "TotalDebt", "Short Long Term Debt", "Long Term Debt",
                        "ShortLongTermDebt", "LongTermDebt"
                    })
                    cash_key = pick_first_key(bs_q.index, {
                        "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents"
                    })
                    total_debt = np.nan
                    if debt_key:
                        total_debt = to_float(bs_q.loc[debt_key].iloc[0])
                    else:
                        # Suma de corto+largo si existen
                        short_k = pick_first_key(bs_q.index, {"Short Long Term Debt", "ShortLongTermDebt"})
                        long_k  = pick_first_key(bs_q.index, {"Long Term Debt", "LongTermDebt"})
                        total_debt = 0.0
                        if short_k:
                            total_debt += to_float(bs_q.loc[short_k].iloc[0])
                        if long_k:
                            total_debt += to_float(bs_q.loc[long_k].iloc[0])
                        if total_debt == 0.0:
                            total_debt = np.nan
                    cash = to_float(bs_q.loc[cash_key].iloc[0]) if cash_key else np.nan
                    if pd.isna(total_debt) or pd.isna(cash):
                        return np.nan
                    net_debt = total_debt - cash
                    return safe_div(net_debt, ebitda_ltm)
            except Exception:
                return np.nan
            return np.nan
    
        # ----------------------------
        # Extracción por ticker
        # ----------------------------
        rows = []
        for tkr in tickers:
            try:
                t = yf.Ticker(tkr)
                bi = get_basic_info(t)
                ebitda_ltm = get_ebitda_ltm(t)
                fcf_ltm    = get_fcf_ltm(t)
    
                ev = bi.get("enterprise_value")
                mc = bi.get("market_cap")
    
                ev_ebitda = safe_div(ev, ebitda_ltm)
                fcf_yield = safe_div(fcf_ltm, mc)
    
                nd_ebitda = get_net_debt_ebitda(t, ebitda_ltm)
    
                rows.append({
                    "Ticker": tkr,
                    "Sector": bi.get("sector") or "Unknown",
                    "Industry": bi.get("industry") or "Unknown",
                    "EV": ev,
                    "MarketCap": mc,
                    "EBITDA_LTM": ebitda_ltm,
                    "FCF_LTM": fcf_ltm,
                    "EV/EBITDA": ev_ebitda,
                    "FCF_Yield": fcf_yield,             # fracción: 0.05 = 5%
                    "Forward_PE": bi.get("forward_pe"),
                    "P/B": bi.get("price_to_book"),
                    "Dividend_Yield": bi.get("dividend_yield"),  # fracción
                    "NetDebt_EBITDA": nd_ebitda
                })
            except Exception as e:
                print(f"Error con {tkr}: {e}")
    
        df = pd.DataFrame(rows)
    
        # ----------------------------
        # Filtro opcional por sector defensa
        # ----------------------------
        if sector and sector.lower().startswith("def"):
            # Nos quedamos con 'Aerospace & Defense' si aparece; si no, mantenemos todos.
            mask_def = df["Industry"].fillna("").str.contains("Aerospace", case=False) & \
                       df["Industry"].fillna("").str.contains("Defense", case=False)
            if mask_def.any():
                df = df.loc[mask_def].copy()
    
        # ----------------------------
        # Conversión a numérico y limpieza
        # ----------------------------
        num_cols = ["EV/EBITDA", "FCF_Yield", "Forward_PE", "P/B", "Dividend_Yield", "NetDebt_EBITDA"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
        # Requisitos mínimos: EV/EBITDA y FCF_Yield disponibles
        df = df.dropna(subset=["EV/EBITDA", "FCF_Yield"]).copy()
        if df.empty:
            print("⚠️ Ninguna acción con datos suficientes (EV/EBITDA y FCF Yield).")
            return None
    
        # ----------------------------
        # Winsorización (5–95%) para reducir outliers
        # ----------------------------
        for col in ["EV/EBITDA", "FCF_Yield", "Forward_PE", "P/B", "Dividend_Yield", "NetDebt_EBITDA"]:
            if col in df.columns:
                df[col] = winsorize_series(df[col], low=0.05, high=0.95)
    
        # ----------------------------
        # Z-scores con signos correctos
        #   - Ratios "más bajo es mejor": EV/EBITDA, Forward_PE, P/B, NetDebt/EBITDA → invertidos (-z)
        #   - Yields "más alto es mejor": FCF_Yield, Dividend_Yield → z directo
        # ----------------------------
        def zscore(series):
            mu, sigma = series.mean(), series.std(ddof=1)
            if sigma is None or sigma == 0 or pd.isna(sigma):
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - mu) / sigma
    
        z_ev_ebitda   = -zscore(df["EV/EBITDA"])
        z_forward_pe  = -zscore(df["Forward_PE"])      if df["Forward_PE"].notna().any() else 0.0
        z_pb          = -zscore(df["P/B"])             if df["P/B"].notna().any()        else 0.0
        z_nd_ebitda   = -zscore(df["NetDebt_EBITDA"])  if df["NetDebt_EBITDA"].notna().any() else 0.0
        z_fcf_yield   =  zscore(df["FCF_Yield"])
        z_div_yield   =  zscore(df["Dividend_Yield"])  if df["Dividend_Yield"].notna().any() else 0.0
    
        # Puedes ponderar; aquí ponderamos más EV/EBITDA y FCF Yield
        df["Composite Value"] = (
            0.35 * z_ev_ebitda +
            0.35 * z_fcf_yield +
            0.10 * z_forward_pe +
            0.10 * z_pb +
            0.05 * z_div_yield +
            0.05 * z_nd_ebitda
        )
    
        # Z-score del composite (para interpretar en desviaciones estándar)
        comp_mu, comp_sigma = df["Composite Value"].mean(), df["Composite Value"].std(ddof=1)
        df["Composite_z"] = (df["Composite Value"] - comp_mu) / (comp_sigma if comp_sigma else 1.0)
    
        # Ranking final (mayor composite = mejor value)
        df["Ranking"] = df["Composite Value"].rank(ascending=False, method="dense")
        df = df.sort_values("Composite Value", ascending=False).reset_index(drop=True)
    
        # Selección de columnas finales
        cols_out = [
            "Ticker", "Sector", "Industry",
            "EV", "MarketCap", "EBITDA_LTM", "FCF_LTM",
            "EV/EBITDA", "FCF_Yield", "Forward_PE", "P/B", "Dividend_Yield", "NetDebt_EBITDA",
            "Composite Value", "Composite_z", "Ranking"
        ]
        for c in cols_out:
            if c not in df.columns:
                df[c] = np.nan
    
        return df[cols_out]
    
    
    
    



    def obtener_composite_value_tickerskk(self, tickers, sector='fin'):
        """
        Calcula el Composite Value (promedio normalizado de ratios de valoración) 
        para una lista de tickers. 
        Selecciona los ratios adecuados según el sector (fin, tech, energy, ind, health...).
        """
    
        import pandas as pd
        import numpy as np
        import yfinance as yf
    
        # 1️⃣ Diccionario de ratios por sector
        sector_ratios = {
            'fin': ["P/E", "P/B"],  # Bancos, financieras
            'energy': ["P/E", "EV/EBITDA", "P/S"],
            'tech': ["P/E", "P/S", "P/B"],
            'ind': ["P/E", "EV/EBITDA", "P/B"],
            'health': ["P/E", "P/B", "P/S"],
            'default': ["P/E", "P/B", "EV/EBITDA", "P/S"]
        }
    
        # 2️⃣ Escoger ratios según el sector
        ratios = sector_ratios.get(sector.lower(), sector_ratios['default'])
        print(f"🔎 Sector: {sector} → usando ratios: {ratios}")
    
        data = []
    
        # 3️⃣ Descargar datos de Yahoo Finance
        for t in tickers:
            try:
                info = yf.Ticker(t).get_info()
                data.append({
                    "Ticker": t,
                    "P/E": info.get("trailingPE", np.nan),
                    "P/B": info.get("priceToBook", np.nan),
                    "EV/EBITDA": info.get("enterpriseToEbitda", np.nan),
                    "P/S": info.get("priceToSalesTrailing12Months", np.nan),
                    "Sector": info.get("sector", "Unknown")
                })
            except Exception as e:
                print(f"Error con {t}: {e}")
    
        df = pd.DataFrame(data)
    
        if df.empty:
            print("⚠️ No se pudieron descargar datos válidos.")
            return None
    
        # 4️⃣ Limpieza de datos (solo las columnas elegidas)
        for col in ratios:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[(df[col] <= 0) | (df[col] > 200), col] = np.nan  # filtrar outliers absurdos
    
        # 5️⃣ Filtros absolutos básicos (según sector)
        if sector == 'fin':
            df = df.loc[(df["P/E"] > 5) & (df["P/E"] < 20) &
                        (df["P/B"] > 0.4) & (df["P/B"] < 2.0)].copy()
        elif sector == 'energy':
            df = df.loc[(df["EV/EBITDA"] > 2) & (df["EV/EBITDA"] < 15)].copy()
        elif sector == 'tech':
            df = df.loc[(df["P/S"] > 1) & (df["P/S"] < 10)].copy()
    
        if df.empty:
            print("⚠️ Ninguna acción pasa los filtros absolutos.")
            return None
    
        # 6️⃣ Normalización (z-score inverso)
        #   Convierte a misma escala por COLUMNA, nos dice lo lejos que está de la media. Z=0 valor en la media 
        z_scores = df[ratios].apply(lambda x: -(x - np.nanmean(x)) / np.nanstd(x))
    
        # 7️⃣ Calcular Composite Value y z-score final
        #   Normaliza los inidcadores para cada valor por FILA
        df["Composite Value"] = z_scores.mean(axis=1)
        #   Normaliza por columnas los valores de todos los marcadores
        df["Composite_z"] = (df["Composite Value"] - df["Composite Value"].mean()) / df["Composite Value"].std()
    
        # 8️⃣ Ranking final. despues de tomar dos indicadores, normalizar y ordenar.
        df["Ranking"] = df["Composite Value"].rank(ascending=False)
        df.sort_values("Composite Value", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
        return df[["Ticker", "Sector"] + ratios + ["Composite Value", "Composite_z", "Ranking"]]
      
    
    def obtener_momentum_log(self, tickers, period=252, start="2020-01-01", end=None):
        """
        Calcula el momentum logarítmico para una lista de tickers en el periodo indicado. Por defecto un año 252
        Devuelve un DataFrame con el momentum y su ranking (1 = más momentum).
        Problema: dehecahmos esta metrica porque calcula el momento entre dos puntos... no es muy matematico.
        Mejoramos haciendo regresion lineal en la funcion calcular_momentum_regresion_tickers
        
        Parámetros
        ----------
        tickers : list
            Lista de símbolos (ej: ["AAPL", "MSFT", "AMZN"])
        period : int
            Periodos (en días) para calcular el momentum (por defecto 252 ≈ 1 año)
        start : str
            Fecha inicial para descargar datos (YYYY-MM-DD)
        end : str
            Fecha final (por defecto hoy)
        """
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from datetime import datetime

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        resultados = []

        for t in tickers:
            try:
                # Descargar precios ajustados
                data = yf.download(t, start=start, end=end, progress=False,auto_adjust=True)
                if len(data) < period:
                    continue

                # Calcular momentum logarítmico
                data["Momentum_log"] = np.log(data["Close"] / data["Close"].shift(period))
                momentum_actual = data["Momentum_log"].iloc[-1]

                resultados.append({"Ticker": t, "Momentum_log": momentum_actual})

            except Exception as e:
                print(f"Error con {t}: {e}")

        df_mom = pd.DataFrame(resultados).dropna()

        # Ranking (1 = más fuerte)
        df_mom["Ranking"] = df_mom["Momentum_log"].rank(ascending=False)
        df_mom.sort_values("Momentum_log", ascending=False, inplace=True)
        df_mom.reset_index(drop=True, inplace=True)
        
        df_mom["Momentum_z"] = (df_mom["Momentum_log"] - df_mom["Momentum_log"].mean()) / df_mom["Momentum_log"].std()


        return df_mom
    
    
    def calcular_momentum_regresion_tickers(self, tickers, window_sma=20, window_reg=60):
        """
        Calcula el momentum (pendiente de la regresión del log-precio sobre la SMA)
        para una lista de tickers. Devuelve un DataFrame con las pendientes normalizadas (z-score).
        Parámetros:
        -----------
        tickers : list[str]
            Lista de símbolos (por ejemplo ['AAPL', 'MSFT', 'JPM'])
        window_sma : int
            Periodo de la media móvil (por defecto 20)
        window_reg : int
            Ventana usada en la regresión (por defecto 60 días)

        Retorna:
        --------
        DataFrame con columnas:
            ['Ticker', 'Momentum_beta', 'Momentum_z']
        """
        
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import yfinance as yf
        
        resultados = []

        for t in tickers:
            try:
                data = yf.download(t, period=f"{window_reg*2}d", interval="1d", progress=False,auto_adjust=True)
                if data.empty:
                    print(f"⚠️ {t}: sin datos válidos.")
                    continue

                # 1️⃣ Calcular SMA
                data["SMA"] = data["Close"].rolling(window_sma).mean()
                y = np.log(data["SMA"].dropna().values[-window_reg:])
                x = np.arange(len(y)).reshape(-1, 1)

                if len(y) < window_reg / 2:
                    print(f"⚠️ {t}: datos insuficientes para regresión.")
                    continue

                # 2️⃣ Ajustar regresión lineal
                model = LinearRegression().fit(x, y)
                beta = model.coef_[0]  # pendiente

                resultados.append({"Ticker": t, "Momentum_beta": beta})

            except Exception as e:
                print(f"Error al calcular momentum para {t}: {e}")

        # 3️⃣ Convertir a DataFrame
        df_mom = pd.DataFrame(resultados)

        if df_mom.empty:
            print("⚠️ Ningún ticker con momentum válido.")
            return None

        # 4️⃣ Normalizar pendientes (z-score) por columna
        df_mom["Momentum_z"] = (df_mom["Momentum_beta"] - df_mom["Momentum_beta"].mean()) / df_mom["Momentum_beta"].std()

        # 5️⃣ Ordenar por momentum
        df_mom.sort_values("Momentum_z", ascending=False, inplace=True)
        df_mom.reset_index(drop=True, inplace=True)

        return df_mom

    def comprar(self, ticker):
        # Aquí pondrías tu lógica real de compra, API o simulación
        print(f"Ejecutando compra de {ticker}")
        
        # Compruebo la tendencia  ADX >25  SMA20>SMA50
        if (not self.analizar_tendencia_UP(ticker)):
            return False   
        
        
        #Llamamos al constructor de la Clase compraVenta con el ID de la cuenta
        import sys
        import importlib
        sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        automatic = importlib.import_module("automatic", "C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        

        alpacaAPI= automatic.tradeAPIClass(para2=automatic.CUENTA_J3_02) 
        
        cantidad = alpacaAPI.positionExist(ticker)
        
        #Compruebo que no estoy 'comprado'
        if (cantidad == 0):
            orderID= alpacaAPI.placeOrder(ticker, 1)
        else:
            print(f"⚠️🔴 {ticker}: Ya estamos comprados .")
            
        return "ok"
    
    def vender(self, ticker):
        # Aquí pondrías tu lógica real de compra, API o simulación
        print(f"Evaluanado VENTA de {ticker}")
        
        # Compruebo la tendencia  BAJISTA   ADX >25  SMA20>SMA50
        if (not self.analizar_tendencia_DOWN(ticker)):
            return False           
        
        #Llamamos al constructor de la Clase compraVenta con el ID de la cuenta
        import sys
        import importlib
        sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        automatic = importlib.import_module("automatic", "C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")

        alpacaAPI= automatic.tradeAPIClass(para2=automatic.CUENTA_J3_02) 
        
        #ticker= 'LNC'
        alpacaAPI.placeOrderSell(ticker,1)        
        
        
        
        return "ok"    

    
    def analizar_tendencia_DOWN(self, ticker, periodo="6mo"):
        """
        Descarga los datos históricos de un ticker y calcula los indicadores:
        - ADX (fuerza de tendencia)
        - SMA20 y SMA50 (medias móviles simples)
        
        Devuelve True si:
          ADX > 25  y  SMA20 < SMA50   → tendencia bajista fuerte
        """
        
        import yfinance as yf
        import ta
        import pandas as pd
        
        try:
            # 1️⃣ Descargar datos históricos
            data = yf.download(ticker, period=periodo, interval="1d", progress=False, auto_adjust=True)
            data.columns = data.columns.droplevel(1)
            if data.empty:
                print(f"⚠️ No hay datos válidos para {ticker}")
                return False
    
            # 2️⃣ Calcular medias móviles
            data["SMA20"] = data["Close"].rolling(window=20).mean()
            data["SMA50"] = data["Close"].rolling(window=50).mean()
    
            # 3️⃣ Calcular ADX
            adx_indicator = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
            data["ADX"] = adx_indicator.adx()
    
            # 4️⃣ Tomar los últimos valores

            sma20 = data["SMA20"].iloc[-1]
            sma50 = data["SMA50"].iloc[-1]
            adx = data["ADX"].iloc[-1]
    
            # 5️⃣ Evaluar condición
            if ( (adx > 25) and (sma20 < sma50)):
                return True
            return False
            #print(f"{ticker}: ADX(25)={adx:.2f}, SMA20={sma20:.2f}, SMA50={sma50:.2f} → {'✅ True' if condicion else '❌ False'}")

    
        except Exception as e:
            print(f"Error al analizar {ticker}: {e}")
            return False
        return False
    
    def analizar_tendencia_UP(self, ticker, periodo="6mo"):
        """
        Descarga los datos históricos de un ticker y calcula los indicadores:
        - ADX (fuerza de tendencia)
        - SMA20 y SMA50 (medias móviles simples)
        
        Devuelve True si:
          ADX > 25  y  SMA20 > SMA50   → tendencia Alcista fuerte
        """
        
        import yfinance as yf
        import ta
        import pandas as pd
        
        try:
            # 1️⃣ Descargar datos históricos
            data = yf.download(ticker, period=periodo, interval="1d", progress=False, auto_adjust=True)
            data.columns = data.columns.droplevel(1)
            if data.empty:
                print(f"⚠️ No hay datos válidos para {ticker}")
                return False
    
            # 2️⃣ Calcular medias móviles
            data["SMA20"] = data["Close"].rolling(window=20).mean()
            data["SMA50"] = data["Close"].rolling(window=50).mean()
    
            # 3️⃣ Calcular ADX
            adx_indicator = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
            data["ADX"] = adx_indicator.adx()
    
            # 4️⃣ Tomar los últimos valores
            sma20 = data["SMA20"].iloc[-1]
            sma50 = data["SMA50"].iloc[-1]
            adx = data["ADX"].iloc[-1]
    
            # 5️⃣ Evaluar condición
            if ( (adx > 25) and (sma20 > sma50)):  #sube fuerts
                return True
    
            #print(f"{ticker}: ADX(25)={adx:.2f}, SMA20={sma20:.2f}, SMA50={sma50:.2f} → {'✅ True' if condicion else '❌ False'}")
            return False
    
        except Exception as e:
            print(f"Error al analizar {ticker}: {e}")
            return False
        return False
    
    
    def vender_con_estrategia (self):
        """
        De los valores invertidos, ejecuta la estrategia de salida
        
        Parámetros:
        -----------

        Retorna:
        --------

        """
        
        
        #Llamamos al constructor de la Clase compraVenta con el ID de la cuenta
        import sys
        import importlib
        import ta
        sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        automatic = importlib.import_module("automatic", "C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")


        alpacaAPI= automatic.tradeAPIClass(para2=automatic.CUENTA_J3_02) 
        
        # 1️⃣ Obtener las posiciones abiertas desde Alpaca
        posiciones = alpacaAPI.get_positions()
    
        if isinstance(posiciones, pd.DataFrame):
            df_valores = posiciones.copy()
        else:
            # Convertir lista de dicts en DataFrame si es necesario
            df_valores = pd.DataFrame([p._raw for p in posiciones])
    
        if df_valores.empty:
            print("⚠️ No hay posiciones abiertas en la cuenta Alpaca.")
            return pd.DataFrame()
    
        # 2️⃣ Normalizar columnas (Alpaca usa snake_case y strings)
        df_valores.rename(columns={
            "symbol": "symbol",
            "qty": "qty",
            "avg_entry_price": "avg_entry_price",
            "current_price": "current_price",
            "unrealized_pl": "unrealized_pl"
        }, inplace=True, errors="ignore")
    
        # Convertir tipos numéricos
        cols_num = ["qty", "avg_entry_price", "current_price", "unrealized_pl"]
        for c in cols_num:
            df_valores[c] = pd.to_numeric(df_valores[c], errors="coerce")
    
        tickers = df_valores["symbol"].unique().tolist()
    

    
        # 4️⃣ Analizar cada ticker
        for t in tickers:
            try:
                self.vender(t)
    
            except Exception as e:
                print(f"❌ Error procesando {t}: {e}")
    

    
        return True
    
        
    def analizar(self, df_tickers, window=60):
        """
        Analiza una lista de tickers según una estrategia técnica usando 'ta'.
        Detecta automáticamente si el DataFrame tiene MultiIndex y lo aplana.
        Devuelve un DataFrame con los tickers que cumplen las condiciones.
        """
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import ta
    
        resultados = []
    
        # Convertir la lista de tickers en lista pura
        tickers = df_tickers["Ticker"].tolist()
    
        # Descargar todos los tickers juntos para mejorar rendimiento
        data_all = yf.download(tickers, period=f"{window*2}d", interval="1d", progress=False,auto_adjust=True)
    
        # Si los datos tienen MultiIndex (varios tickers)
        if isinstance(data_all.columns, pd.MultiIndex):
            # Aplanar o iterar por ticker
            for t in tickers:
                try:
                    # Extraer los datos de un ticker específico
                    data = data_all.xs(t, level=1, axis=1).dropna()
    
                    if data.empty:
                        print(f"⚠️ {t}: sin datos válidos.")
                        continue
    
                    # Calcular indicadores técnicos
                    high = data["High"].squeeze().astype(float)
                    low = data["Low"].squeeze().astype(float)
                    close = data["Close"].squeeze().astype(float)
    
                    data["SMA50"] = close.rolling(50).mean()
                    data["ADX"] = ta.trend.adx(high, low, close, window=14)
                    data["MACD"] = ta.trend.macd(close)
                    data["MACD_signal"] = ta.trend.macd_signal(close)
    
                    # Condiciones de compra
                    adx_ok = data["ADX"].iloc[-1] > 25
                    sma_ok = close.iloc[-1] > data["SMA50"].iloc[-1]
                    macd_ok = data["MACD"].iloc[-1] > data["MACD_signal"].iloc[-1]
    
                    if adx_ok and sma_ok and macd_ok:
                        resultados.append({
                            "Ticker": t,
                            "ADX": round(data["ADX"].iloc[-1], 2),
                            "Close": round(close.iloc[-1], 2),
                            "SMA50": round(data["SMA50"].iloc[-1], 2),
                            "MACD": round(data["MACD"].iloc[-1], 3),
                            "MACD_signal": round(data["MACD_signal"].iloc[-1], 3),
                            "Buy_Signal": True
                        })
    
                except Exception as e:
                    print(f"❌ Error con {t}: {e}")
    
        else:
            # Caso: datos de un solo ticker (sin MultiIndex)
            try:
                t = tickers[0]
                data = data_all.dropna()
    
                high = data["High"].squeeze().astype(float)
                low = data["Low"].squeeze().astype(float)
                close = data["Close"].squeeze().astype(float)
    
                data["SMA50"] = close.rolling(50).mean()
                data["ADX"] = ta.trend.adx(high, low, close, window=14)
                data["MACD"] = ta.trend.macd(close)
                data["MACD_signal"] = ta.trend.macd_signal(close)
    
                adx_ok = data["ADX"].iloc[-1] > 25
                sma_ok = close.iloc[-1] > data["SMA50"].iloc[-1]
                macd_ok = data["MACD"].iloc[-1] > data["MACD_signal"].iloc[-1]
    
                if adx_ok and sma_ok and macd_ok:
                    resultados.append({
                        "Ticker": t,
                        "ADX": round(data["ADX"].iloc[-1], 2),
                        "Close": round(close.iloc[-1], 2),
                        "SMA50": round(data["SMA50"].iloc[-1], 2),
                        "MACD": round(data["MACD"].iloc[-1], 3),
                        "MACD_signal": round(data["MACD_signal"].iloc[-1], 3),
                        "Buy_Signal": True
                    })
    
            except Exception as e:
                print(f"❌ Error con {t}: {e}")
    
        # Convertir resultados a DataFrame
        df_result = pd.DataFrame(resultados)
    
        if df_result.empty:
            print("⚠️ Ningún ticker cumple la estrategia técnica.")
        else:
            print(f"✅ {len(df_result)} tickers cumplen las condiciones.")
    
        return df_result
    
        
    def backtest(self, tickers, start="2015-01-01", end="2025-01-01", score_threshold=1.0, atr_mult=2):
        """
        Backtest simplificado de la estrategia Value + Momentum + Stop ATR.
        Simula compras cuando el Score_total es alto y salidas cuando salta el stop dinámico.
        """
    
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from datetime import timedelta
    
        results = []
        equity_curve = pd.Series(dtype=float)
    
        for t in tickers:
            try:
                # 1️⃣ Descargar precios
                data = yf.download(t, start=start, end=end, progress=False,auto_adjust=True)
                if data.empty:
                    continue
    
                # 2️⃣ Calcular indicadores técnicos
                data["H-L"] = data["High"] - data["Low"]
                data["H-PC"] = abs(data["High"] - data["Close"].shift(1))
                data["L-PC"] = abs(data["Low"] - data["Close"].shift(1))
                data["TR"] = data[["H-L", "H-PC", "L-PC"]].max(axis=1)
                data["ATR"] = data["TR"].rolling(14).mean()
    
                data["max20"] = data["High"].rolling(20).max()
                data["stop_atr"] = data["max20"] - atr_mult * data["ATR"]
    
                # 3️⃣ Simular señales
                # Entrada: Score_total alto (aquí simulamos un score aleatorio por ejemplo)
                np.random.seed(0)
                data["Score_total"] = np.random.normal(0.8, 0.4, len(data))  # TODO: reemplazar con tu score real
                data["signal"] = 0
                data.loc[data["Score_total"] > score_threshold, "signal"] = 1
                
                ###########
                df_val = self.obtener_composite_value_tickers(tickers, sector='fin')
                
                #df_mom = objEstra.obtener_momentum_log(tickers, period=252)  # semestral
                df_mom =self.calcular_momentum_regresion_tickers(tickers)
                
                 # Fusionar ambos DataFrames
                df_final = pd.merge(df_val, df_mom, on="Ticker", how="inner")
                
                # Calcular score total combinado (igual peso)
                df_final["Score_total"] = 0.5 * df_final["Composite_z"] + 0.5 * df_final["Momentum_z"]
                
                # Ranking general
                df_final["Ranking_Total"] = df_final["Score_total"].rank(ascending=False)
                df_final.sort_values("Score_total", ascending=False, inplace=True)
                df_final.reset_index(drop=True, inplace=True)
                
                data["Score_total"] = df_final["Ranking_Total"]
                data["signal"] = 0
                data.loc[data["Score_total"] > score_threshold, "signal"] = 1
                
                #############
                
                
                
    
                # Salida: Stop ATR
                data.loc[data["Close"] < data["stop_atr"], "signal"] = 0
    
                # 4️⃣ Determinar posición
                data["position"] = data["signal"].shift(1).ffill()
    
                # 5️⃣ Calcular rendimiento diario
                data["returns"] = data["Close"].pct_change()
                data["strategy"] = data["position"] * data["returns"]
    
                # 6️⃣ Acumular resultado
                cumulative = (1 + data["strategy"].fillna(0)).cumprod()
                results.append({
                    "Ticker": t,
                    "Total_Return": cumulative.iloc[-1] - 1,
                    "CAGR": ((1 + cumulative.iloc[-1]) ** (252 / len(data)) - 1),
                    "Sharpe": np.sqrt(252) * data["strategy"].mean() / data["strategy"].std(),
                    "Max_Drawdown": (cumulative / cumulative.cummax() - 1).min()
                })
    
                equity_curve = equity_curve.add(cumulative, fill_value=0)
    
            except Exception as e:
                print(f"⚠️ Error en {t}: {e}")
    
        # 7️⃣ Consolidar resultados
        df_results = pd.DataFrame(results)
        df_results.sort_values("CAGR", ascending=False, inplace=True)
        equity_curve /= len(results)  # promedio
    
        # 8️⃣ Mostrar resultados
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve.index, equity_curve.values, label="Estrategia")
        plt.title("Curva de capital promedio - Value + Momentum + Stop ATR")
        plt.legend()
        plt.grid(True)
        plt.show()
    
        print("Resumen del Backtest:\n", df_results.round(3))
        print("\nRentabilidad media:", df_results["CAGR"].mean())
        print("Sharpe promedio:", df_results["Sharpe"].mean())
    
        return df_results
    
    
    
        
        def graficar_dispersion(self, df_final):
           """
           Genera un gráfico de dispersión (scatter) para visualizar la relación 
           entre Value (Composite_z) y Momentum (Momentum_z).
           El color de los puntos representa el Score_total.
           """
           import matplotlib.pyplot as plt
    
           if not all(col in df_final.columns for col in ["Composite_z", "Momentum_z", "Score_total", "Ticker"]):
               print("❌ Error: faltan columnas necesarias en df_final.")
               return
    
           plt.figure(figsize=(8,6))
           sc = plt.scatter(
               df_final["Composite_z"], 
               df_final["Momentum_z"],
               c=df_final["Score_total"],
               cmap="viridis",
               s=120,
               edgecolors="k"
           )
    
           # Etiquetas con nombres de los tickers
           for i, txt in enumerate(df_final["Ticker"]):
               plt.annotate(txt, (df_final["Composite_z"][i], df_final["Momentum_z"][i]), fontsize=9)
    
           plt.xlabel("Value (Composite_z)")
           plt.ylabel("Momentum (Momentum_z)")
           plt.title("Mapa Value vs Momentum (color = Score total)")
           plt.colorbar(sc, label="Score total")
           plt.grid(True, linestyle="--", alpha=0.6)
           plt.show()   
    
    def graficar_burbujas(self, df_final):
        """
        Genera un gráfico de burbujas donde:
        - Eje X = Value (Composite_z)
        - Eje Y = Momentum (Momentum_z)
        - Tamaño y color = Score_total
        """
        import matplotlib.pyplot as plt

        if not all(col in df_final.columns for col in ["Composite_z", "Momentum_z", "Score_total", "Ticker"]):
            print("❌ Error: faltan columnas necesarias en df_final.")
            return

        plt.figure(figsize=(9,6))
        sc = plt.scatter(
            df_final["Composite_z"],
            df_final["Momentum_z"],
            s=(df_final["Score_total"] + 3) * 80,  # tamaño proporcional
            c=df_final["Score_total"],
            cmap="coolwarm",
            alpha=0.7,
            edgecolors="k"
        )

        # Etiquetas con los tickers
        for i, txt in enumerate(df_final["Ticker"]):
            plt.annotate(txt, (df_final["Composite_z"][i], df_final["Momentum_z"][i]), fontsize=9)

        plt.xlabel("Value (Composite_z)")
        plt.ylabel("Momentum (Momentum_z)")
        plt.title("Relación Value vs Momentum (tamaño y color = Score total)")
        plt.colorbar(sc, label="Score total")
        plt.grid(alpha=0.3)
        plt.show()

    def graficar_ranking(self, df_final):
        
        import matplotlib.pyplot as plt
             
        # 3️⃣ Ordenar de mayor a menor atractivo
        df_sorted = df_final.sort_values("Score_total", ascending=False).reset_index(drop=True)
        
        # 4️⃣ Mostrar resultados en tabla
        top_n=10
        print("\n🏁 Ranking de acciones por Score_total (Value + Momentum):\n")
        print(df_sorted[["Ticker", "Composite_z", "Momentum_z", "Score_total"]].head(top_n).to_string(index=False))
        
        # 5️⃣ Visualización (opcional)
        #if grafico:

        top_df = df_sorted.head(top_n)
        plt.figure(figsize=(10,6))
        plt.barh(top_df["Ticker"], top_df["Score_total"], color="dodgerblue", alpha=0.8)
        # Línea vertical roja en x=1
        plt.axvline(x=0.6, color="green", linestyle="--", linewidth=2, label="Umbral Score=1")
        
        plt.xlabel("Score Total (Value + Momentum)")
        plt.title(f"Top {top_n} acciones según Score_total")
        plt.gca().invert_yaxis()  # Mostrar el top arriba
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.show()       
             
 
    
#################################################### Clase FIN






#/******************************** FUNCION PRINCIPAL main() *********/
#     def main():   
if __name__ == '__main__':    
        
    """Esta parte del codigo se ejecuta cuando llamo tipo EXE
    Abajo tenemos el else: librería que es cuando se ejecuta como libreria.
        
    Parámetros:
    a -- 
    b -- 
    c -- 

    
    """   

    print ('version(J): ',versionVersion) 
    print (' DEFENSE flavour') 

    """
    print(sys.argv[1])   #se configura en 'run' 'configuration per file'

    if (True or sys.argv[1]== 'prod' ):
        print('Produccion')
        sys.exit()
    """
    
    from telegram_bot import *

    
    objEstra =valueMomentumClass("AMZN")
    send_message("Procesosna ValueMomentum")

    
    print(f"PER de {objEstra.ticker}", objEstra.obtener_per(objEstra.ticker))


    #tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN", "META", "NVDA", "KO", "PFE", "INTC"]
    #objEstra.backtest(tickers, start="2015-01-01", end="2025-01-01", score_threshold=1.0, atr_mult=2)
    
    
    #tickers=  tickers_financials
    tickers= defense_tickers
    df_val = objEstra.obtener_composite_value_tickers(tickers, sector='fin')
    
    #df_mom = objEstra.obtener_momentum_log(tickers, period=252)  # semestral
    df_mom = objEstra.calcular_momentum_regresion_tickers(tickers)
    
     # Fusionar ambos DataFrames
    df_final = pd.merge(df_val, df_mom, on="Ticker", how="inner")
    
    # Calcular score total combinado (igual peso)
    df_final["Score_total"] = 0.5 * df_final["Composite_z"] + 0.5 * df_final["Momentum_z"]
    
    # Ranking general
    df_final["Ranking_Total"] = df_final["Score_total"].rank(ascending=False)
    df_final.sort_values("Score_total", ascending=False, inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    #graficamos    
    
    if DEBUG__:
            
        #objEstra.graficar_dispersion(df_final)
        objEstra.graficar_burbujas(df_final)
        objEstra.graficar_ranking(df_final)

    #######################################################################
    #  Decision de compra
    #######################################################################
    
    # 1.- Total score por encima de 1 --> BUY 
    df_compra = df_final[df_final["Score_total"] > 0.5]
    
    for _, fila in df_compra.iterrows():
        ticker = fila["Ticker"]
        score = fila["Score_total"]
        print(f"🟢 Evaluando la compra de {ticker} (Score={score:.2f})")
        objEstra.comprar(ticker)



    #######################################################################
    #  Decision de VENTA
    #######################################################################

    objEstra.vender_con_estrategia()


    if DEBUG__:
        import keyboard
        print("Pulsa una tecla para finalizar ")

        tecla = keyboard.read_key()
    
    

    print('✅✅ This is it................ 1')
    logging.warning('Paso por ValueMomentum, esto es una migita FIN ')
    sys.exit(32)

    
    """
    Entrada por la librería, no por el MAIN.
    """
else:
    """
    Esta parte del codigo se ejecuta si uso como libreria/paquete""    
    """    
    print (' libreria')
    print ('version(l): ',versionVersion)    
    







