# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 20:53:43 2025

@author: jjjimenez
"""


import configKEY
from openai import OpenAI
 
#print (configKEY.J3_OIA_KEY)

ticker = "AMZN"
 
#client = OpenAI(api_key=configKEY.J3_OIA_KEY)

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

