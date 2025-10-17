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
    

    def obtener_composite_value_tickers(self, tickers):
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
       for col in ["P/E", "P/B", "EV/EBITDA", "P/S"]:
           df[col] = pd.to_numeric(df[col], errors="coerce")

       # 3️⃣ Filtros absolutos (descarta extremos)
       df = df.loc[
           (df["P/E"] > 2) & (df["P/E"] < 25) &
           (df["P/B"] > 0.2) & (df["P/B"] < 3) &
           (df["EV/EBITDA"] > -99999) & (df["EV/EBITDA"] < 999999930) &
           (df["P/S"] > -9999990) & (df["P/S"] < 99999910)
       ].copy()

       if df.empty:
           print("⚠️ Ninguna acción pasa los filtros absolutos.")
           return None

       # 4️⃣ Normalización (z-score inverso)
       z_scores = df[["P/E", "P/B", "EV/EBITDA", "P/S"]].apply(
           lambda x: -(x - np.nanmean(x)) / np.nanstd(x)
       )

       # 5️⃣ Calcular Composite Value (media de z-scores)
       df["Composite Value"] = z_scores.mean(axis=1)
       # 6️⃣ Calcular Composite_z (z-score del Composite Value)
       df["Composite_z"] = (df["Composite Value"] - df["Composite Value"].mean()) / df["Composite Value"].std()


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
    
    def obtener_momentum_log(self, tickers, period=252, start="2020-01-01", end=None):
        """
        Calcula el momentum logarítmico para una lista de tickers en el periodo indicado. Por defecto un año 252
        Devuelve un DataFrame con el momentum y su ranking (1 = más momentum).
        
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
    
    objEstra.obtener_per(objEstra.ticker)
    
    print(f"PER de {objEstra.ticker}", objEstra.obtener_per(objEstra.ticker))


    #tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN", "META", "NVDA", "KO", "PFE", "INTC"]
    
    tickers=  tickers_financials
    df_val = objEstra.obtener_composite_value_tickers(tickers)
    
    df_mom = objEstra.obtener_momentum_log(tickers, period=252)  # semestral
    
     # Fusionar ambos DataFrames
    df_final = pd.merge(df_val, df_mom, on="Ticker", how="inner")
    
    # Calcular score total combinado (igual peso)
    df_final["Score_total"] = 0.5 * df_final["Composite_z"] + 0.5 * df_final["Momentum_z"]
    
    # Ranking general
    df_final["Ranking_Total"] = df_final["Score_total"].rank(ascending=False)
    df_final.sort_values("Score_total", ascending=False, inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    
    
    objEstra.graficar_dispersion(df_final)
    objEstra.graficar_burbujas(df_final)

    
    print('This is it................ 1')
    



    
    """
    Entrada por la librería, no por el MAIN.
    """
else:
    """
    Esta parte del codigo se ejecuta si uso como libreria/paquete""    
    """    
    print (' libreria')
    print ('version(l): ',versionVersion)    
    







