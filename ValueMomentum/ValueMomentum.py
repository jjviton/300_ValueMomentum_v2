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
        Calcula el Composite Value (promedio normalizado de ratios de valoración) para una lista de tickers.
        Devuelve un DataFrame ordenado (menor valor = más barata).
        """
        data = []
    
        for t in tickers:
            try:
                info = yf.Ticker(t).get_info()
                pe = info.get("trailingPE", np.nan)
                pb = info.get("priceToBook", np.nan)
                ev_ebitda = info.get("enterpriseToEbitda", np.nan)
                ps = info.get("priceToSalesTrailing12Months", np.nan)
    
                data.append({
                    "Ticker": t,
                    "P/E": pe,
                    "P/B": pb,
                    "EV/EBITDA": ev_ebitda,
                    "P/S": ps
                })
    
            except Exception as e:
                print(f"Error con {t}: {e}")
    
        df = pd.DataFrame(data)
    
        # Limpiar datos absurdos o negativos
        for col in ["P/E", "P/B", "EV/EBITDA", "P/S"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[(df[col] <= 0) | (df[col] > 200), col] = np.nan  # filtrar outliers
    
        # Normalización por columna (z-score inverso: menor ratio = más value)
        z_scores = df[["P/E", "P/B", "EV/EBITDA", "P/S"]].apply(
            lambda x: -(x - np.nanmean(x)) / np.nanstd(x)
        )
    
        # Calcular promedio de z-scores disponibles (Composite Value)
        df["Composite Value"] = z_scores.mean(axis=1)
    
        # Ranking (1 = más barata, N = más cara)
        df["Ranking"] = df["Composite Value"].rank(ascending=False)
    
        # Ordenar por valor compuesto
        df.sort_values("Composite Value", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    
        return df[["Ticker", "P/E", "P/B", "EV/EBITDA", "P/S", "Composite Value", "Ranking"]]

    
    
    
    

    
 
    
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


    tickers = ["AAPL", "MSFT", "JPM", "XOM", "AMZN", "META", "NVDA", "KO", "PFE", "INTC"]
    resultado = objEstra.obtener_composite_value_tickers(tickers)
    

    
    print('This is it................ 6')
    



    
    """
    Entrada por la librería, no por el MAIN.
    """
else:
    """
    Esta parte del codigo se ejecuta si uso como libreria/paquete""    
    """    
    print (' libreria')
    print ('version(l): ',versionVersion)    
    







