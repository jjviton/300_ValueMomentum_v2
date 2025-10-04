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


import configKEY
from openai import OpenAI
 
#print (configKEY.J3_OIA_KEY)

ticker = "AMZN"
 
client = OpenAI(api_key=configKEY.J3_OIA_KEY)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [
    {
        "role": "system",
        "content": (
            "Eres un analista financiero experto en mercados bursátiles. "
            "Respondes con detalle y claridad sobre los datos fundamentales de las empresas "
            "(ingresos, beneficios, márgenes, deuda, PER, flujo de caja, etc.), "
            "y además contextualizas con análisis del sector cuando sea relevante. "
            "tu fuente de informacion no pueden ser los datos publicados por la propia empresa, analiza fuentes independientes"
            "Tu estilo es profesional, riguroso y claro, como si escribieras un informe para un inversor institucional."
            "El informe es de 10 lineas maximo"
            "Indicmane la fuente de estos datos con los que has trabajado"
            
        )
    },
    {
        "role": "user",
        "content": f"Dame un análisis fundamental de {ticker}."
    }
]

)

print(response.choices[0].message.content)

del client



import yfinance as yf

def obtener_per(_ticker):
    accion = yf.Ticker(_ticker)
    info = accion.info
    return info.get("trailingPE", "PER no disponible")

print(f"PER de {ticker}", obtener_per(ticker))







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
versionVersion = 1.1
globalVar  = True
pdf_flag =True

#################################################### Clase Estrategia 



class valueMomentumClass:

    """CLASE xxx

       
    """  
    
    #Variable de CLASE
    backtesting = False  #variable de la clase, se accede con el nombre
    n_past = 14  # Number of past days we want to use to predict the future.  FILAS
    flag01 =0
   
    def __init__(self, previson_a_x_days=4, Y_supervised_ = 'hull', para1=False, para2=1):
        
        #Variable de INSTANCIA
        self.para_02 = para2   #variable de la isntancia
        
        globalVar = True
        #intance.flag01 =True
        
        return
    
    """
    Getter y setter para el acceso a atributo/propiedades
    """    
    def __getattribute__(self, attr):
        if attr == 'loss':
            return self._loss
        elif attr == 'xxx':
            return self._edad
        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, valor):
        if attr == 'loss':
            self._loss = valor
        elif attr == 'xxx':
            self._edad = valor
        else:
            object.__setattr__(self, attr, valor)    
    
        
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

    #print(sys.argv[1])   #se configura en 'run' 'configuration per file'

    if (True or sys.argv[1]== 'prod' ):
        print('Produccion')
        sys.exit()

    
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
    







