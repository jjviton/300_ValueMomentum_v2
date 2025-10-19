# -*- coding: utf-8 -*-


"""
*****************************************************************************
.PY

Programa para un estrategia robusta y basica
OBEJTIVO: tener algo consistente que me anime en otras estrategias mas arriesgadas


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

# J3_DEBUG__ = False  #variable global (global J3_DEBUG__ )


################################ IMPORTAMOS MODULOS A UTILIZAR.
import pandas as pd
import numpy as np

import yfinance as yf


################################# ENTORNO
import sys
sys.path.insert(0,"C:\\Users\\jjjimenez\\Documents\\quant\\libreria")
from sp500 import tickers_financials


####################### LOGGING
import logging    #https://docs.python.org/3/library/logging.html
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../log/registro.log', level=logging.INFO ,force=True,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.warning('esto es una kkk')

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
    

    def obtener_composite_value_tickers(self, tickers, sector ='fin'):
       """
       Calcula el Composite Value (promedio normalizado de ratios de valoración) 
       para una lista de tickers.
       Aplica filtros absolutos para evitar que acciones 'caras' distorsionen el ranking.
       Devuelve un DataFrame ordenado (menor valor = más barata).
       """
       import pandas as pd
       import numpy as np
       import yfinance as yf

       data = []

       # 1️⃣ Descargar datos de Yahoo Finance
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

       # 2️⃣ Limpieza de datos
       for col in ["P/E", "P/B"]:  #, "EV/EBITDA", "P/S"]:
           df[col] = pd.to_numeric(df[col], errors="coerce")

       # 3️⃣ Filtros absolutos (descarta extremos)
       df = df.loc[
           (df["P/E"] > 8) & (df["P/E"] < 20) &
           (df["P/B"] > 0.8) & (df["P/B"] < 2) 
       ].copy()
       
       """&
       (df["EV/EBITDA"] > -99999) & (df["EV/EBITDA"] < 999999930) &
       (df["P/S"] > -9999990) & (df["P/S"] < 99999910)"""



       if df.empty:
           print("⚠️ Ninguna acción pasa los filtros absolutos.")
           return None

       # 4️⃣ Normalización (z-score inverso) por columnas
       z_scores = df[["P/E", "P/B" ]].apply(       
           lambda x: -(x - np.nanmean(x)) / np.nanstd(x)
       )
       
       """, "EV/EBITDA", "P/S" """

       # 5️⃣ Calcular Composite Value (media de z-scores)  por fila
       df["Composite Value"] = z_scores.mean(axis=1)
       # 6️⃣ Calcular Composite_z (z-score del Composite Value)
       df["Composite_z"] = (df["Composite Value"] - df["Composite Value"].mean()) / df["Composite Value"].std()
       # J· repasar este calculo de arriba, no me cuadra

       # 6️⃣ Ranking final
       df["Ranking"] = df["Composite Value"].rank(ascending=False)
       df.sort_values("Composite Value", ascending=False, inplace=True)
       df.reset_index(drop=True, inplace=True)
       # 7️⃣ Ranking final
       
       #df["Ranking"] = df["Composite Value"].rank(ascending=False)
       #df.sort_values("Composite Value", ascending=False, inplace=True)
       #df.reset_index(drop=True, inplace=True)
        
       return df[["Ticker", "Sector", "P/E", "P/B", "EV/EBITDA", "P/S",
                   "Composite Value", "Composite_z", "Ranking"]]
    
    def obtener_composite_value_tickers(self, tickers, sector='fin'):
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
        z_scores = df[ratios].apply(lambda x: -(x - np.nanmean(x)) / np.nanstd(x))
    
        # 7️⃣ Calcular Composite Value y z-score final
        df["Composite Value"] = z_scores.mean(axis=1)
        df["Composite_z"] = (df["Composite Value"] - df["Composite Value"].mean()) / df["Composite Value"].std()
    
        # 8️⃣ Ranking final
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
                data = yf.download(t, start=start, end=end, progress=False)
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
                data = yf.download(t, period=f"{window_reg*2}d", interval="1d", progress=False)
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

        # 4️⃣ Normalizar pendientes (z-score)
        df_mom["Momentum_z"] = (df_mom["Momentum_beta"] - df_mom["Momentum_beta"].mean()) / df_mom["Momentum_beta"].std()

        # 5️⃣ Ordenar por momentum
        df_mom.sort_values("Momentum_z", ascending=False, inplace=True)
        df_mom.reset_index(drop=True, inplace=True)

        return df_mom

    def comprar(self, ticker):
        # Aquí pondrías tu lógica real de compra, API o simulación
        print(f"Ejecutando compra de {ticker}")
        
        #Llamamos al constructor de la Clase compraVenta con el ID de la cuenta
        import sys
        import importlib
        sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
        automatic = importlib.import_module("automatic", "C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")


        alpacaAPI= automatic.tradeAPIClass(para2=automatic.CUENTA_J3_01) 
        
        
        orderID= alpacaAPI.placeOrder(ticker, 1)
        
        
        
        return "ok"

        
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
        data_all = yf.download(tickers, period=f"{window*2}d", interval="1d", progress=False)
    
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
        plt.axvline(x=1, color="red", linestyle="--", linewidth=2, label="Umbral Score=1")
        
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

    """
    print(sys.argv[1])   #se configura en 'run' 'configuration per file'

    if (True or sys.argv[1]== 'prod' ):
        print('Produccion')
        sys.exit()
    """
    
    objEstra =valueMomentumClass("AMZN")
    
    
    #objEstra.comprar("LNC")
    
    objEstra.obtener_per(objEstra.ticker)
    
    print(f"PER de {objEstra.ticker}", objEstra.obtener_per(objEstra.ticker))


    #tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN", "META", "NVDA", "KO", "PFE", "INTC"]
    
    tickers=  tickers_financials
    df_val = objEstra.obtener_composite_value_tickers(tickers, sector='fin')
    
    #df_mom = objEstra.obtener_momentum_log(tickers, period=252)  # semestral
    

    df_mom=objEstra.calcular_momentum_regresion_tickers(tickers)
    
     # Fusionar ambos DataFrames
    df_final = pd.merge(df_val, df_mom, on="Ticker", how="inner")
    
    # Calcular score total combinado (igual peso)
    df_final["Score_total"] = 0.5 * df_final["Composite_z"] + 0.5 * df_final["Momentum_z"]
    
    # Ranking general
    df_final["Ranking_Total"] = df_final["Score_total"].rank(ascending=False)
    df_final.sort_values("Score_total", ascending=False, inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    #graficamos    
    #objEstra.graficar_dispersion(df_final)
    objEstra.graficar_burbujas(df_final)
    objEstra.graficar_ranking(df_final)

    #######################################################################
    #  Decision de compra
    
    
    # 1.- Total score por encima de 1 --> BUY 


    df_compra = df_final[df_final["Score_total"] > 1]
    
    for _, fila in df_compra.iterrows():
        ticker = fila["Ticker"]
        score = fila["Score_total"]
        print(f"🟢 Comprando {ticker} (Score={score:.2f})")
        objEstra.comprar(ticker)

    
    # 2.- TotalScore entre 0,5 y 1 confirmacion tecnica
    df_analizar = df_final[(df_final["Score_total"] > 0.5) & (df_final["Score_total"] < 1)]
    
    df_compra = objEstra.analizar(df_analizar)

    if df_compra.empty:
        print(f"⚠️Sin buenas inversiones tras analisis 0.5 to 1.")
    else:
        for _, fila in df_compra.iterrows():
            ticker = fila["Ticker"]
            score = fila["Score_total"]
            print(f"🟢 Comprando {ticker} (Score={score:.2f})")
            objEstra.comprar(ticker)

 
    
    print('✅✅ This is it................ 1')
    



    
    """
    Entrada por la librería, no por el MAIN.
    """
else:
    """
    Esta parte del codigo se ejecuta si uso como libreria/paquete""    
    """    
    print (' libreria')
    print ('version(l): ',versionVersion)    
    







