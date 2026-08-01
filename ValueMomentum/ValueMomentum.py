
"""
****************************************************************************
.PY
FINANCIAL
Programa para una estrategia robusta Quality + Value + Momentum
OBJETIVO: version 2 de tu ValueMomentum original, con Quality añadido,
          filtros sector-aware, pesos risk-parity, exit rules reales,
          stop por ATR y position sizing.
Universos: bancos STOXX 600 + defensa europea. SIN ALPACA (solo señales).
******************************************************************************
******************************************************************************
Mejoras:
Started on JUL/2026
Version_2: Quality + Value + Momentum, sector-aware
Objetivo: tener algo consistente en timeframe SEMANAS (swing, no intradia)
Author: J3Viton (v2 generada junto a Claude, revisar antes de producción)
 
NOTA IMPORTANTE — ESTE SCRIPT NO TIENE BACKTEST:
------------------------------------------------
El método backtest() se ha eliminado deliberadamente. Consecuencia directa:
NO hay forma dentro de este script de estimar si la estrategia funciona
antes de arriesgar dinero. Los parámetros que lleva (pesos por sector,
umbrales de ADX, 2xATR de stop, 42 días de holding, 10 posiciones) NO
están validados empíricamente aquí: son valores razonables por criterio,
no óptimos demostrados.
 
Antes de operar en real, valida la estrategia por otra vía (backtest
externo, paper trading varias semanas registrando resultados, o
reimplantando un backtest). Operar esto en real sin validación previa
es operar a ciegas.
 
Limitación adicional de la fuente de datos: yfinance.get_info() solo da
el fundamental ACTUAL (hoy), no histórico point-in-time. Para cualquier
validación histórica seria de Quality/Value necesitarías Sharadar, SimFin,
Compustat o Refinitiv.
"""
# -*- coding: utf-8 -*-
 
DEBUG__ = False  # variable global
 
################################ IMPORTAMOS MODULOS A UTILIZAR.
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
 
################################# ENTORNO
import sys
sys.path.insert(0, "C:\\Users\\jjjimenez\\Documents\\quant\\libreria")
try:
    from sp500 import stoxx_600_banks_tickers
except Exception:
    stoxx_600_banks_tickers = ["JPM", "BAC", "GS", "MS", "C"]  # fallback demo
 
# DEFENSA EUROPEA. Se intenta cargar la lista YA VALIDADA que genera
# validar_tickers.py; si no existe (aún no lo has ejecutado), usa la lista
# de respaldo de aquí abajo.
#
# OJO con dos tickers trampa:
#   "BA.L"  = BAE Systems.  "BA" sin sufijo es BOEING (otra empresa, otro país)
#   "AM.PA" = Dassault Aviation. NO es "DSY.PA" (Dassault Systèmes = software)
DEFENSA_EUROPA_FALLBACK = [
    "BA.L", "RR.L", "QQ.L", "BAB.L", "CHG.L", "AVON.L", "SNR.L",
    "HO.PA", "AIR.PA", "SAF.PA", "AM.PA", "EXE.PA",
    "RHM.DE", "HAG.DE", "R3NK.DE", "MTX.DE",
    "LDO.MI", "FNC.MI", "AVIO.MI",
    "SAAB-B.ST", "KOG.OL", "IDR.MC",
]
try:
    from validar_tickers import cargar_universo
    defensa_europa_tickers = cargar_universo("defensa_europa")
    print(f"✅ Lista de defensa cargada VALIDADA: {len(defensa_europa_tickers)} tickers")
except Exception:
    defensa_europa_tickers = DEFENSA_EUROPA_FALLBACK
    print(f"ℹ️ Usando lista de defensa de respaldo ({len(defensa_europa_tickers)} tickers, "
          f"sin validar). Ejecuta validar_tickers.py para depurarla.")
 
# ENERGÍA EUROPEA (fósil: integradas + E&P).
# NO incluye servicios petroleros (Saipem, Subsea 7, Tenaris...) ni renovables
# (Ørsted, Neste...): son modelos de negocio distintos y meterlos en el mismo
# z-score compara peras con manzanas. La investigación es explícita sobre esto:
# el factor value es positivo en fósiles y ~cero/negativo en no fósiles.
# Si los quieres operar, hazlo como universos aparte (están en validar_tickers.py).
ENERGIA_EUROPA_FALLBACK = [
    # Integradas / majors
    "SHEL.L", "BP.L", "TTE.PA", "ENI.MI", "REP.MC",
    "EQNR.OL", "OMV.VI", "GALP.LS", "PKN.WA",
    # Exploración y producción (E&P)
    "AKRBP.OL", "VAR.OL", "DNO.OL",
    "HBR.L", "ENOG.L", "ITH.L", "TLW.L", "CNE.L",
]
try:
    energia_europa_tickers = cargar_universo("energia_europa")
    print(f"✅ Lista de energía cargada VALIDADA: {len(energia_europa_tickers)} tickers")
except Exception:
    energia_europa_tickers = ENERGIA_EUROPA_FALLBACK
    print(f"ℹ️ Usando lista de energía de respaldo ({len(energia_europa_tickers)} tickers, "
          f"sin validar). Ejecuta validar_tickers.py para depurarla.")
 
####################### LOGGING
import logging  # https://docs.python.org/3/library/logging.html
import os
 
# RUTAS ABSOLUTAS basadas en la ubicación del script, NO relativas al cwd.
# Con rutas relativas, lanzado desde cron con otro directorio de trabajo,
# escribías/leías ficheros distintos (y el log petaba al importar si la
# carpeta ../log no existía -> FileNotFoundError antes de ejecutar nada).
_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_DIR_LOG = os.path.join(_DIR_SCRIPT, "..", "log")
try:
    os.makedirs(_DIR_LOG, exist_ok=True)
    _RUTA_LOG = os.path.join(_DIR_LOG, "registro_qvm.log")
except Exception:
    _RUTA_LOG = os.path.join(_DIR_SCRIPT, "registro_qvm.log")  # fallback junto al script
 
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=_RUTA_LOG, level=logging.INFO, force=True,
                     format='%(asctime)s:%(levelname)s:%(message)s')
logging.warning('Paso por QualityValueMomentum, esto es una migita STOCXX v2')
 
#################################################### COLORES DE CONSOLA (ANSI)
# OJO: esto SOLO colorea la CONSOLA. Telegram NO admite texto en color:
# su API permite negrita, cursiva, subrayado, tachado, código y spoiler,
# pero no color de fuente. Para destacar en Telegram hay que usar emojis
# (🔴) o mayúsculas, que es lo que hacemos abajo.
class _C:
    ROJO = '\033[91m'
    VERDE = '\033[92m'
    AMARILLO = '\033[93m'
    AZUL = '\033[94m'
    NEGRITA = '\033[1m'
    FIN = '\033[0m'
 
    @classmethod
    def rojo(cls, t):
        return f"{cls.ROJO}{cls.NEGRITA}{t}{cls.FIN}"
 
    @classmethod
    def verde(cls, t):
        return f"{cls.VERDE}{t}{cls.FIN}"
 
    @classmethod
    def amarillo(cls, t):
        return f"{cls.AMARILLO}{t}{cls.FIN}"
 
 
# Windows: sin esto, cmd.exe antiguo imprime los códigos ANSI como basura
# (â[91m...) en vez de colorear. En PowerShell y Terminal moderno no hace falta.
if os.name == "nt":
    try:
        os.system("")   # activa el procesamiento de secuencias ANSI en Windows
    except Exception:
        pass
 
#### Variables globales
versionVersion = 0.2
globalVar = True
pdf_flag = True
 
#################################################### FILTROS Y PESOS POR SECTOR
# Cada sector mide "barato", "bueno" y "con momentum" de forma distinta.
# Esto es lo que tu v1 ignoraba por completo (aplicaba P/E 8-20 a TODO).
SECTOR_FILTERS = {
    'Financial Services': {
        # Recalibrado con datos reales de 2025-2026 (no aflojado a ciegas):
        # STOXX Europe 600 Banks forward P/E ~10.1 (media histórica 9.5),
        # EURO STOXX Banks 30-15 P/E ~12.15 (jul-2026), y el P/B agregado de
        # la banca europea alcanzó niveles de antes de 2008 tras subir 65%
        # en 2025. El rango (6,14)/(0.6,1.4) estaba calibrado a la "década
        # barata" 2009-2022, que ya no describe el mercado actual.
        'value': {'P/E': (5, 16), 'P/B': (0.5, 2.0)},
        'quality': {'ROE': (0.08, 1.0), 'DebtToEquity': (0, 1500)},  # bancos usan leverage alto
        # Bancos NO reportan FCF ni CurrentRatio como el resto de sectores
        # (yfinance casi siempre devuelve NaN aquí para financieras) ->
        # si exiges estos dos checks, el filtro se queda vacío SIEMPRE.
        'require_fcf_positive': False,
        'require_current_ratio_min': None,
        # IMPORTANTE: esto es rentabilidad, NO solvencia. yfinance no da
        # CET1/Tier1/NPL para bancos. Ver aviso en obtener_composite_quality_value_tickers().
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth", "RevenueGrowth"],
        'momentum': {'ADX': 25, 'lookback_dias': 56, 'holding_dias': 42},
        # Risk-parity aproximado: Quality pesa más (riesgo sistémico de bancos)
        # OJO: con quality_cols actual, este 50% protege de "poco rentable",
        # NO de "poco capitalizado/solvente" — son cosas distintas.
        'weights': {'quality': 0.50, 'value': 0.30, 'momentum': 0.20},
    },
    # SECTOR DEFENSA / AEROESPACIAL
    # Rangos MUY distintos a los de banca: el sector se revalorizó brutalmente
    # con el rearme europeo (2022-2026). Con los filtros de banca (P/E 5-16)
    # no pasaría ni un solo ticker.
    #
    # Nota sobre los pesos: Value baja al 20% y Momentum sube al 45% a
    # propósito. Con el sector en máximos, "barato" casi no existe -- solo
    # puedes rankear "menos caro que sus pares", que es señal mucho más débil.
    # Fingir que Value aporta aquí lo mismo que en banca sería engañarse.
    #
    # A favor: los industriales SÍ reportan FCF y CurrentRatio en yfinance
    # (al revés que los bancos), así que aquí esos filtros de calidad
    # funcionan de verdad y los dejamos activos.
    'Defensa': {
        'value': {'P/E': (10, 45), 'P/B': (1.0, 12.0)},
        'quality': {'ROE': (0.08, 1.0), 'DebtToEquity': (0, 300)},
        'require_fcf_positive': True,
        'require_current_ratio_min': 1.0,
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth", "RevenueGrowth"],
        'momentum': {'ADX': 25, 'lookback_dias': 56, 'holding_dias': 42},
        'weights': {'quality': 0.35, 'value': 0.20, 'momentum': 0.45},
    },
    # SECTOR ENERGÍA (fósil: integradas + E&P)
    #
    # Respaldo académico para aplicar QVM aquí:
    #  - El HML (factor value) es POSITIVO en empresas fósiles y ~cero/negativo
    #    en no fósiles. Al revés que en banca, donde el value lleva 10 años muerto.
    #  - El modelo de 5 factores de Fama-French es MÁS preciso en energía que
    #    en la mayoría de sectores.
    #  - La incertidumbre del crudo amplifica la prima de valor.
    #
    # Y lo que más importa en la práctica: las petroleras SÍ reportan FCF,
    # DebtToEquity y CurrentRatio en yfinance. En bancos tuvimos que desactivar
    # esos filtros, así que allí Quality_z mide rentabilidad pero NO solvencia.
    # Aquí el factor Quality mide de verdad lo que dice medir.
    #
    # Rango de P/E deliberadamente AMPLIO (3-20): sector muy cíclico. En la
    # parte alta del ciclo del crudo cotizan a P/E 4-8 y en la baja se disparan.
    # Un rango estrecho te dejaría fuera justo en los momentos interesantes.
    'Energia': {
        'value': {'P/E': (3, 20), 'P/B': (0.5, 3.5)},
        'quality': {'ROE': (0.06, 1.0), 'DebtToEquity': (0, 200)},
        'require_fcf_positive': True,       # AQUÍ SÍ funciona (a diferencia de bancos)
        'require_current_ratio_min': 0.8,   # las petroleras van justas de circulante
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth", "RevenueGrowth"],
        'momentum': {'ADX': 25, 'lookback_dias': 56, 'holding_dias': 42},
        # VALUE es el que más pesa: es el factor con respaldo empírico
        # específico en este sector (al revés que en defensa, donde lo bajé
        # al 20% porque nada está barato).
        'weights': {'quality': 0.30, 'value': 0.40, 'momentum': 0.30},
    },
    'Technology': {
        'value': {'P/E': (18, 55), 'P/B': (2, 12)},
        'quality': {'ROE': (0.15, 1.0), 'DebtToEquity': (0, 100)},
        'require_fcf_positive': True,
        'require_current_ratio_min': 1.0,
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"],
        'momentum': {'ADX': 20, 'lookback_dias': 28, 'holding_dias': 28},
        'weights': {'quality': 0.30, 'value': 0.25, 'momentum': 0.45},
    },
    'Utilities': {
        'value': {'P/E': (12, 19), 'P/B': (1.0, 2.0)},
        'quality': {'ROE': (0.06, 1.0), 'DebtToEquity': (0, 200)},
        'require_fcf_positive': False,   # utilities suelen tener FCF ajustado por capex regulado
        'require_current_ratio_min': None,
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"],
        'momentum': {'ADX': 25, 'lookback_dias': 84, 'holding_dias': 56},
        'weights': {'quality': 0.45, 'value': 0.35, 'momentum': 0.20},
    },
    'default': {
        'value': {'P/E': (8, 25), 'P/B': (0.8, 3)},
        'quality': {'ROE': (0.10, 1.0), 'DebtToEquity': (0, 300)},
        'require_fcf_positive': True,
        'require_current_ratio_min': 1.0,
        'quality_cols': ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"],
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
    # (eliminados: backtesting, n_past, flag01 -- residuos sin uso; y
    #  tickets_comprado, una lista hardcodeada que engañaba: el libro real
    #  de posiciones es self.posiciones_abiertas + el JSON de persistencia)
 
    # Reglas de salida / riesgo (lo que tu v1 no tenía)
    max_holding_dias = 42        # ~6 semanas, ajustable por sector
    adx_exit_threshold = 20      # si ADX cae debajo de esto, el momentum "murió"
    take_profit_pct = 0.25       # +25% asegura ganancia (opcional)
 
    # --- STOP LOSS POR ATR (sustituye al -10% fijo) ---
    # El -10% plano era un número redondo sin relación con cómo se mueve
    # realmente cada activo: en un banco de volatilidad diaria 1.2% son ~8
    # desviaciones típicas (no salta nunca, te comes la caída entera), y en
    # uno de 3.5% son ~3 (salta con ruido normal y te saca de posiciones sanas).
    # El ATR (Average True Range) mide el recorrido típico diario del activo,
    # así que "entrada - atr_multiplo x ATR" pone el stop a la MISMA distancia
    # estadística para todos, no a la misma distancia porcentual.
    usar_stop_atr = True         # False -> vuelve al stop porcentual clásico
    atr_window = 14              # ventana estándar del ATR
    atr_multiplo = 2.0           # 2xATR: valor habitual en swing trading
    stop_loss_pct = 0.10         # fallback: se usa si usar_stop_atr=False
                                 # o si el ATR no se puede calcular (datos insuficientes)
    stop_atr_pct_max = 0.20      # tope de seguridad: aunque el ATR sugiera más,
                                 # nunca arriesgar más de un 20% en un solo trade
    max_posiciones_abiertas = 10   # tope GLOBAL (bancos + defensa + lo que añadas)
    riesgo_max_por_trade = 0.02    # 2% del capital arriesgado por trade
 
    # Tope de concentración por posición. ANTES estaba escrito a mano dentro
    # de calcular_tamano_posicion() como "capital * 0.20", lo que lo hacía
    # invisible y peligroso: al subir de 5 a 10 posiciones, 10 x 20% = 200%
    # del capital. Debe cumplirse siempre:
    #     max_posiciones_abiertas x max_pct_por_posicion <= 1.0
    max_pct_por_posicion = 0.10    # 10 posiciones x 10% = 100% del capital
 
    # SUELO ABSOLUTO DE CALIDAD DE LA SEÑAL.
    #
    # El Score_total es un z-score ponderado: 0 = justo en la media del grupo,
    # negativo = PEOR que la media de sus propios pares.
    #
    # Sin este suelo, el código compraba "el mejor disponible" aunque fuera
    # malo en términos absolutos. Caso real: QQ.L entró con Score -0.46
    # (Quality -0.47, Momentum -0.81) simplemente porque los 3 candidatos
    # mejor puntuados habían sido rechazados por el filtro de ADX y quedaban
    # plazas libres. Comprar algo por debajo de la media de su sector solo
    # porque "es lo menos malo que hay hoy" no es una decisión de inversión.
    #
    # Con 0.0: se exige estar POR ENCIMA de la media del grupo. Si nadie lo
    # cumple, la plaza se queda vacía y te quedas en liquidez, que es una
    # posición perfectamente válida.
    min_score_compra = 0.0
    comision_bps = 10            # 10 bps por lado, ajusta a tu broker real
    slippage_bps = 5
 
    # ARCHIVO DE PERSISTENCIA: self.posiciones_abiertas es un dict EN MEMORIA.
    # Este script está pensado para correr como cron/tarea programada (mira
    # el sys.exit(32) al final del main) -> el proceso Python muere después
    # de cada ejecución -> sin persistir a disco, posiciones_abiertas se
    # reinicia VACÍO en cada corrida. Consecuencia real: vender_con_estrategia()
    # jamás encontraría una posición para aplicarle stop/time-exit, porque el
    # objeto "olvida" todo lo comprado en la ejecución anterior.
    posiciones_file = os.path.join(_DIR_SCRIPT, "posiciones_abiertas.json")
 
    # HISTÓRICO DE OPERACIONES (append-only).
    #
    # posiciones_abiertas.json es el estado ACTUAL (se sobrescribe en cada
    # cambio). Este otro fichero es el REGISTRO PERMANENTE: cada compra y cada
    # venta se añaden y no se borran nunca.
    #
    # Por qué importa: este script NO tiene backtest. Sin él, la única forma
    # de saber algún día si la estrategia funciona es acumular operaciones
    # reales con TODOS los datos que motivaron cada decisión. Si solo guardas
    # "compré X a 10", dentro de un año no podrás responder a la pregunta
    # importante: ¿los candidatos con Score alto rindieron mejor que los de
    # Score bajo? Por eso guardamos los tres z-scores, el ranking, los ratios
    # y el ADX de cada entrada.
    #
    # Formato JSONL (un JSON por línea) en vez de un array JSON único: si el
    # proceso muere a mitad de escritura, un array JSON queda corrupto y
    # pierdes TODO el histórico; en JSONL pierdes como mucho la última línea.
    # Ya nos pasó con posiciones_abiertas.json y un histórico no se puede
    # reconstruir.
    historial_file = os.path.join(_DIR_SCRIPT, "historial_operaciones.jsonl")
 
    # COBERTURA MÍNIMA DE DATOS para que un ticker sea evaluable.
    # Si yfinance no da al menos este nº de indicadores de Quality, el ticker
    # se EXCLUYE en vez de puntuarlo con datos parciales. Sin esto, un banco
    # con 1 de 5 indicadores obtenía una nota de calidad basada en ese único
    # dato y podía ganar el ranking a otro con información completa.
    min_cobertura_quality = 3
 
    def __init__(self, ticker_="AAPL", Y_supervised_='hull', para1=False, para2=1):
 
        self.para_02 = para2
        globalVar = True
 
        self.ticker = ticker_
        self.posiciones_abiertas = {}   # ticker -> dict(entrada, cantidad, fecha_entrada, stop, sector)
 
        # Etiqueta del universo que se está procesando ahora mismo. Se usa para
        # marcar TODOS los mensajes (consola y Telegram) y los nombres de los
        # gráficos. Sin esto, al procesar bancos y defensa en la misma corrida
        # recibirías dos tandas de avisos idénticos sin saber cuál es cuál.
        self.etiqueta_actual = "general"
 
        self.cargar_posiciones()        # recupera lo que había abierto de la corrida anterior
 
        return
 
    ################################################################
    # NOTIFICACIONES SEGURAS
    #
    # Telegram NUNCA debe poder tumbar una operación. Antes había llamadas
    # a send_message() sin proteger en comprar() y en vender_con_estrategia():
    # si Telegram fallaba (red caída, token expirado, o NameError al usar el
    # módulo como librería), la excepción abortaba la venta ANTES de cerrar
    # la posición -- se detectaba el stop loss y la posición seguía abierta.
    # Un aviso es cosmético; cerrar una posición es protección de capital.
    ################################################################
    def _notificar(self, texto, critico=False):
        """Envía a Telegram sin dejar que un fallo propague. Devuelve bool."""
        try:
            send_message(texto)
            return True
        except NameError:
            if critico:
                print("   ⚠️ send_message no disponible (módulo usado como librería): "
                      "aviso NO enviado. La operación SÍ se ha ejecutado.")
            return False
        except Exception as e:
            print(f"   ⚠️ Fallo enviando a Telegram: {e}. "
                  f"{'La operación SÍ se ha ejecutado.' if critico else ''}")
            return False
 
    def _notificar_imagen(self, ruta, caption=""):
        """Envía una imagen a Telegram sin propagar fallos."""
        try:
            enviar_png_telegram(ruta, caption=caption)
            return True
        except Exception as e:
            print(f"   ⚠️ No pude enviar {ruta} a Telegram: {e}")
            return False
 
    ################################################################
    # HISTÓRICO DE OPERACIONES — registro permanente append-only
    ################################################################
    def _registrar_operacion(self, registro):
        """
        Añade UNA operación al histórico. Nunca lanza excepción: un fallo
        escribiendo el log jamás debe tumbar una compra o una venta.
 
        Cada registro incluye TODO lo que motivó la decisión, no solo el
        precio: sin eso, dentro de un año no podrás analizar si la estrategia
        realmente funciona ni qué factor aportó.
        """
        import json
        try:
            registro = dict(registro)
            registro.setdefault("timestamp", datetime.now().isoformat())
            # numpy/pandas no son serializables directamente -> a tipo nativo
            limpio = {}
            for k, v in registro.items():
                if isinstance(v, (np.integer,)):
                    limpio[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    limpio[k] = None if pd.isna(v) else float(v)
                elif isinstance(v, (np.bool_,)):
                    limpio[k] = bool(v)
                elif isinstance(v, float) and pd.isna(v):
                    limpio[k] = None
                else:
                    limpio[k] = v
            with open(self.historial_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(limpio, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"   ⚠️ No pude escribir en el histórico ({self.historial_file}): {e}. "
                  f"La operación SÍ se ha ejecutado.")
 
    def cargar_historial(self, como_dataframe=True):
        """
        Lee el histórico completo. Devuelve DataFrame (o lista de dicts).
        Las líneas corruptas se saltan una a una, sin perder el resto.
        """
        import json
        if not os.path.exists(self.historial_file):
            print(f"ℹ️ Aún no existe {self.historial_file} (sin operaciones registradas).")
            return pd.DataFrame() if como_dataframe else []
 
        registros, malas = [], 0
        with open(self.historial_file, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    registros.append(json.loads(linea))
                except Exception:
                    malas += 1
        if malas:
            print(f"⚠️ {malas} línea(s) corruptas saltadas del histórico.")
        return pd.DataFrame(registros) if como_dataframe else registros
 
    def resumen_historial(self):
        """
        Estadísticas de las operaciones CERRADAS (round trips completos).
        Es lo más parecido a un backtest que tienes: resultados reales.
        """
        df = self.cargar_historial()
        if df.empty:
            return None
 
        ventas = df[df["tipo"] == "VENTA"] if "tipo" in df.columns else pd.DataFrame()
        compras = df[df["tipo"] == "COMPRA"] if "tipo" in df.columns else pd.DataFrame()
 
        print("\n" + "=" * 62)
        print("HISTÓRICO DE LA ESTRATEGIA")
        print("=" * 62)
        print(f"  Operaciones registradas: {len(df)}  ({len(compras)} compras, {len(ventas)} ventas)")
 
        if ventas.empty or "pnl_pct" not in ventas.columns:
            print("  Aún no hay operaciones CERRADAS: sin resultados que analizar.")
            print("=" * 62 + "\n")
            return df
 
        pnl = ventas["pnl_pct"].dropna()
        ganadoras = pnl[pnl > 0]
        perdedoras = pnl[pnl <= 0]
 
        print(f"\n  --- Operaciones cerradas: {len(pnl)} ---")
        print(f"  Win rate:          {len(ganadoras)/len(pnl)*100:>7.1f}%")
        print(f"  Ganancia media:    {ganadoras.mean():>+7.2f}%" if len(ganadoras) else "  Ganancia media:      n/a")
        print(f"  Pérdida media:     {perdedoras.mean():>+7.2f}%" if len(perdedoras) else "  Pérdida media:       n/a")
        print(f"  P&L medio:         {pnl.mean():>+7.2f}%")
        print(f"  P&L total (€):     {ventas['pnl_dinero'].sum():>+9.2f}" if "pnl_dinero" in ventas else "")
        print(f"  Mejor / peor:      {pnl.max():>+7.2f}% / {pnl.min():+.2f}%")
        if "dias_en_posicion" in ventas.columns:
            print(f"  Holding medio:     {ventas['dias_en_posicion'].mean():>7.0f} días")
 
        if "motivo_corto" in ventas.columns:
            print("\n  --- Motivos de salida ---")
            for m, c in ventas["motivo_corto"].value_counts().items():
                sub = ventas[ventas["motivo_corto"] == m]["pnl_pct"]
                print(f"     {m:<28} {c:>3} ops   P&L medio {sub.mean():+.2f}%")
 
        if "sector" in ventas.columns:
            print("\n  --- Por sector ---")
            for s, g in ventas.groupby("sector"):
                p = g["pnl_pct"].dropna()
                if len(p):
                    print(f"     {s:<22} {len(p):>3} ops   win {(p>0).mean()*100:>5.1f}%   P&L medio {p.mean():+.2f}%")
 
        # LA PREGUNTA CLAVE: ¿el Score_total predice el resultado?
        # Si no hay relación, el ranking no aporta y la estrategia es ruido.
        if "score_total" in ventas.columns and ventas["score_total"].notna().sum() >= 10:
            v = ventas.dropna(subset=["score_total", "pnl_pct"])
            corr = v["score_total"].corr(v["pnl_pct"])
            print("\n  --- ¿El Score predice el resultado? ---")
            print(f"     Correlación Score_total vs P&L: {corr:+.3f}  (n={len(v)})")
            if corr > 0.2:
                print("     ✅ Relación positiva: el ranking aporta información.")
            elif corr < -0.2:
                print("     🔴 Relación NEGATIVA: el ranking está seleccionando AL REVÉS.")
            else:
                print("     ⚠️ Sin relación clara. Con pocos datos es normal; si persiste")
                print("        con 50+ operaciones, el Score no está aportando nada.")
        else:
            print("\n  (Con 10+ operaciones cerradas podré decirte si el Score predice el P&L)")
 
        print("=" * 62 + "\n")
        return df
 
    def cargar_posiciones(self):
        """Carga posiciones_abiertas desde disco (JSON). Si no existe el
        archivo (primera vez que corres el script), arranca en blanco."""
        import json
        import os
        if not os.path.exists(self.posiciones_file):
            self.posiciones_abiertas = {}
            return
 
        try:
            with open(self.posiciones_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"⚠️ No pude LEER {self.posiciones_file}: {e}. "
                  f"Arranco con la cartera vacía.")
            self.posiciones_abiertas = {}
            return
 
        # RECONVERSIÓN ENTRADA POR ENTRADA.
        # Antes el bucle estaba dentro del mismo try: UNA sola posición con
        # fecha corrupta hacía saltar el except y dejaba posiciones_abiertas
        # VACÍO. Consecuencia real: el script creía no tener nada abierto,
        # compraba de nuevo, y el primer guardar_posiciones() sobrescribía el
        # JSON -> se perdía el rastro de las posiciones reales para siempre.
        # Ahora una entrada mala solo se descarta a sí misma.
        cargadas, corruptas = {}, []
        for ticker, pos in raw.items():
            try:
                pos["fecha_entrada"] = datetime.fromisoformat(pos["fecha_entrada"])
                cargadas[ticker] = pos
            except Exception as e:
                corruptas.append((ticker, str(e)))
 
        self.posiciones_abiertas = cargadas
        print(f"📂 Cargadas {len(cargadas)} posiciones abiertas desde {self.posiciones_file}")
 
        if corruptas:
            print(f"⚠️ {len(corruptas)} entrada(s) CORRUPTA(S) descartadas del JSON:")
            for tk, err in corruptas:
                print(f"     - {tk}: {err}")
            print(f"   ATENCIÓN: si esas posiciones existen en tu broker, el script "
                  f"YA NO las controla (sin stop ni time-exit). Revisa el JSON a mano.")
 
    def guardar_posiciones(self):
        """Persiste posiciones_abiertas a disco (JSON). Se llama tras CADA
        compra o venta -- si no, el fix de cargar_posiciones() de nada sirve
        (cargarías datos, pero nunca escribirías los cambios nuevos)."""
        import json
        try:
            serializable = {}
            for ticker, pos in self.posiciones_abiertas.items():
                pos_copia = dict(pos)
                pos_copia["fecha_entrada"] = pos["fecha_entrada"].isoformat()
                serializable[ticker] = pos_copia
            with open(self.posiciones_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error guardando {self.posiciones_file}: {e}. "
                  f"Las posiciones de esta corrida NO quedarán registradas para la próxima.")
 
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
                    "RevenueGrowth": info.get("revenueGrowth", np.nan),
                    "DividendYield": info.get("dividendYield", np.nan),
                    # Estos 3 campos yfinance los deja en NaN sistemáticamente
                    # para el sector financiero (bancos no reportan cash flow
                    # ni balance con la estructura estándar current assets/
                    # liabilities). Los dejamos porque los pide el screen
                    # absoluto en OTROS sectores (Tech, Utilities, default),
                    # pero para bancos son decorativos, ver aviso más abajo.
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
 
        for col in ["P/E", "P/B", "ROE", "ROA", "ProfitMargin", "EarningsGrowth",
                    "RevenueGrowth", "DividendYield", "DebtToEquity", "CurrentRatio", "FCF"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
        require_fcf = filtros.get('require_fcf_positive', True)
        require_cr_min = filtros.get('require_current_ratio_min', None)
        quality_cols = filtros.get('quality_cols', ["ROE", "ROA", "ProfitMargin", "EarningsGrowth"])
 
        # AVISO HONESTO: para 'Financial Services', FCF, DebtToEquity y
        # CurrentRatio vienen NaN sistemáticamente en yfinance (confirmado
        # empíricamente, no solo en teoría). Esto significa que el Quality_z
        # de bancos NO mide solvencia/capital ni apalancamiento real, aunque
        # el peso 'quality'=0.50 se justificó precisamente por "riesgo
        # sistémico". Sustituimos por RevenueGrowth (sí disponible) como
        # señal extra de calidad operativa, pero esto sigue siendo
        # rentabilidad, NO capital adequacy (CET1, Tier 1, ratio de morosidad).
        # Si quieres medir solvencia bancaria de verdad necesitas otra fuente
        # (informes Pillar 3 de cada banco, EBA transparency exercise, o un
        # proveedor especializado tipo S&P Capital IQ / SNL Financial).
        # yfinance simplemente no lo tiene para financieras.
        if sector == 'Financial Services':
            print("ℹ️ [Financial Services] Quality_z aquí = rentabilidad (ROE/ROA/margen/crecimiento), "
                  "NO solvencia. FCF/DebtToEquity/CurrentRatio no están disponibles para bancos en yfinance.")
 
        # 0) FILTRO DE COBERTURA DE DATOS — se aplica ANTES que nada.
        #
        # Motivo: un ticker sin datos NO es lo mismo que un ticker que suspende
        # el filtro, pero el código los trataba igual (NaN > x es False, así
        # que desaparecía en silencio y no sabías si era "caro" o "sin dato").
        #
        # Y peor: en el z-score de QUALITY, pandas .mean() IGNORA los NaN.
        # Un banco con 1 de 5 indicadores obtenía nota de calidad calculada
        # sobre ese único dato y competía de tú a tú con otro que tenía los 5
        # -- llegando incluso a ganarle el ranking. Eso es ruido disfrazado
        # de señal. Aquí exigimos un mínimo de cobertura real.
        n_lista = len(tickers)        # los que pediste
        n_descargados = len(df)       # los que yfinance devolvió (algunos fallan)
        n_fallo_descarga = n_lista - n_descargados
        print(f"🔎 [{sector}] Universo inicial: {n_descargados} tickers descargados de {n_lista} en la lista")
 
        # 0a) Indicadores IMPRESCINDIBLES: sin ellos no se puede ni valorar
        imprescindibles = ["P/E", "P/B", "ROE"]
        sin_datos_clave = df[df[imprescindibles].isna().any(axis=1)]
        excluidos_imprescindibles = []
        if not sin_datos_clave.empty:
            print(f"🚫 [{sector}] {len(sin_datos_clave)} ticker(s) EXCLUIDOS por falta de datos "
                  f"imprescindibles (no por suspender el filtro):")
            for _, fila_sd in sin_datos_clave.iterrows():
                faltantes = [c for c in imprescindibles if pd.isna(fila_sd[c])]
                excluidos_imprescindibles.append(fila_sd['Ticker'])
                print(f"     - {fila_sd['Ticker']}: sin {', '.join(faltantes)}")
        df = df[df[imprescindibles].notna().all(axis=1)].copy()
 
        # 0b) Cobertura mínima en las columnas de QUALITY
        cobertura = df[quality_cols].notna().sum(axis=1)
        df["Quality_n_datos"] = cobertura
        insuficientes = df[cobertura < self.min_cobertura_quality]
        excluidos_cobertura = []
        if not insuficientes.empty:
            print(f"🚫 [{sector}] {len(insuficientes)} ticker(s) EXCLUIDOS por cobertura "
                  f"insuficiente de Quality (<{self.min_cobertura_quality} de {len(quality_cols)} indicadores):")
            for _, fila_ins in insuficientes.iterrows():
                presentes = [c for c in quality_cols if pd.notna(fila_ins[c])]
                excluidos_cobertura.append(fila_ins['Ticker'])
                print(f"     - {fila_ins['Ticker']}: solo {len(presentes)}/{len(quality_cols)} "
                      f"({', '.join(presentes) if presentes else 'ninguno'})")
        df = df[cobertura >= self.min_cobertura_quality].copy()
 
        n_evaluables = len(df)
        pct = (n_evaluables / n_lista * 100) if n_lista else 0
 
        # ---- RESUMEN DE COBERTURA: consola + Telegram ----
        # Si este porcentaje es bajo, TODO el ranking posterior se calcula
        # sobre una muestra sesgada (solo los bancos que yfinance documenta
        # bien), y los z-scores comparan contra menos peers de los que crees.
        etiq = getattr(self, "etiqueta_actual", "general").upper()
        resumen = [
            "",
            f"📊 [{etiq}] COBERTURA DE DATOS — sector '{sector}'",
            "───────────────────────────────────────────────",
            f"  Tickers en la lista:       {n_lista}",
            f"  Con datos descargados:     {n_descargados}" + (f"   ({n_fallo_descarga} fallaron la descarga)" if n_fallo_descarga else ""),
            f"  Excluidos sin P/E, P/B o ROE:      {len(excluidos_imprescindibles)}",
            f"  Excluidos por Quality incompleto:  {len(excluidos_cobertura)}",
            f"  ✅ EVALUABLES:             {n_evaluables} de {n_lista}  ({pct:.0f}%)",
            "───────────────────────────────────────────────",
        ]
        if pct < 50:
            resumen.append(f"  ⚠️ Menos de la mitad del universo es evaluable. El ranking se")
            resumen.append(f"     calcula solo sobre esos {n_evaluables}: los z-scores comparan contra")
            resumen.append(f"     muchos menos peers de los que crees. Interpreta con cautela.")
            resumen.append("───────────────────────────────────────────────")
 
        texto_resumen = "\n".join(resumen)
        print(texto_resumen)
        # No silenciamos el fallo: si Telegram no recibe el aviso, quiero
        # verlo en consola. Un 'except: pass' aquí significaría creer que
        # estás avisado cuando no lo estás.
        try:
            send_message(texto_resumen)
            print("   (resumen de cobertura enviado a Telegram ✓)")
        except NameError:
            print("   ⚠️ send_message no está definido (módulo usado como librería): "
                  "resumen NO enviado a Telegram.")
        except Exception as e:
            print(f"   ⚠️ FALLÓ el envío a Telegram del resumen de cobertura: {e}")
 
        if df.empty:
            print(f"⚠️ Ningún ticker tiene datos suficientes para ser evaluado en '{sector}'.")
            return None
 
        # 1) FILTROS ABSOLUTOS — SECTOR-AWARE, aplicados paso a paso con
        #    diagnóstico (así ves EXACTAMENTE dónde se te vacía la tabla).
        #    A partir de aquí, si un ticker cae es porque SUSPENDE el filtro,
        #    no porque le falten datos (eso ya se filtró arriba).
        n0 = len(df)
 
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
 
        # GUARD CRÍTICO: el z-score (VALUE y QUALITY) es PEER-RELATIVE, es
        # decir, depende de la media/std del propio grupo que sobrevive.
        # Con n=1 la std es 0 -> división 0/0 -> NaN -> Score_total = NaN
        # -> "NaN > threshold" es SIEMPRE False, y el ticker desaparece sin
        # ningún error visible (esto es justo lo que te pasó con BMPS.MI al
        # mezclar bancos USA + europeos: los USA no pasaron el filtro de P/E
        # y solo quedó BMPS.MI solo, n=1, z-score matemáticamente indefinido).
        # Con n=2-4 el z-score existe pero es estadísticamente basura (std
        # calculado con 2-4 puntos es ruido puro). Exigimos un mínimo
        # razonable para que el ranking relativo tenga sentido.
        MIN_UNIVERSO_ZSCORE = 8
        if len(df) < MIN_UNIVERSO_ZSCORE:
            print(f"⚠️ [{sector}] Solo {len(df)} tickers sobreviven los filtros absolutos "
                  f"(mínimo recomendado: {MIN_UNIVERSO_ZSCORE}). El z-score con tan pocos "
                  f"puntos es matemáticamente inestable (o NaN con n=1) y NO es fiable. "
                  f"No mezcles universos heterogéneos (ej. bancos USA + bancos europeos) "
                  f"en la misma lista: usa un peer group amplio y homogéneo por sector/región.")
            return None
 
        # 2) VALUE: z-score inverso (barato = mejor)
        z_val = df[["P/E", "P/B"]].apply(lambda x: -(x - x.mean()) / x.std() if x.std() else 0.0)
        df["Composite Value"] = z_val.mean(axis=1)
        df["Value_z"] = self._zscore_seguro(df["Composite Value"])
 
        # 3) QUALITY: z-score directo (mayor = mejor), columnas configurables
        #    por sector (quality_cols, definido arriba desde SECTOR_FILTERS).
        #    Para bancos esto es RENTABILIDAD, no solvencia — ver aviso arriba.
        q_scores = df[quality_cols].apply(lambda x: (x - x.mean()) / x.std() if x.std() else 0.0)
        df["Quality"] = q_scores.mean(axis=1)
        df["Quality_z"] = self._zscore_seguro(df["Quality"])
 
        # 4) Blend previo Value+Quality (Momentum se añade después con calcular_score_final)
        df["ValueQuality_z"] = 0.5 * df["Value_z"] + 0.5 * df["Quality_z"]
        df["Ranking_VQ"] = df["ValueQuality_z"].rank(ascending=False)
        df.sort_values("ValueQuality_z", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
 
        return df[["Ticker", "Sector", "P/E", "P/B", "ROE", "ROA", "ProfitMargin",
                    "EarningsGrowth", "RevenueGrowth", "DividendYield",
                    "FCF", "DebtToEquity", "CurrentRatio",
                    "Composite Value", "Value_z", "Quality", "Quality_z",
                    "Quality_n_datos", "ValueQuality_z", "Ranking_VQ"]]
 
    ################################################################
    # MOMENTUM — igual que en tu v1 (regresión sobre SMA, no log-return simple)
    ################################################################
    def calcular_momentum_regresion_tickers(self, tickers, window_sma=20, window_reg=60):
        """
        Calcula el momentum (pendiente de la regresión del log-precio sobre
        la SMA) para una lista de tickers. Igual que tu v1 (idea correcta,
        la mantenemos).
        """
        resultados = []
        for t in tickers:
            try:
                data = yf.download(t, period=f"{max(window_reg*3, 180)}d", interval="1d",
                                    progress=False, auto_adjust=True)
                data = self._normalizar_ohlc(data)   # sin esto, data["Close"] puede ser DataFrame
                if data.empty:
                    print(f"⚠️ {t}: sin datos válidos.")
                    continue
 
                # Función pura (ver NÚCLEO DE CÁLCULO)
                beta = self._momentum_beta(data["Close"], window_sma, window_reg)
                if pd.isna(beta):
                    print(f"⚠️ {t}: datos insuficientes para regresión de momentum.")
                    continue
                if beta <= 0:
                    continue  # solo tendencias positivas (long-only)
                resultados.append({"Ticker": t, "Momentum_beta": beta})
            except Exception as e:
                print(f"Error al calcular momentum para {t}: {e}")
 
        df_mom = pd.DataFrame(resultados)
        if df_mom.empty:
            print("⚠️ Ningún ticker con momentum válido (ninguno en tendencia alcista).")
            return None
 
        # z-score SEGURO: con n=1 el .std() daba NaN -> Score_total NaN ->
        # 'NaN > umbral' False -> el candidato desaparecía sin ningún aviso.
        # Este era el mismo bug que te dejó a BMPS.MI fuera con la lista corta.
        if len(df_mom) < 3:
            print(f"⚠️ Solo {len(df_mom)} ticker(s) con momentum positivo. El Momentum_z "
                  f"resultante NO es estadísticamente significativo (comparas contra "
                  f"casi nada). Trátalo como informativo, no como señal fiable.")
        df_mom["Momentum_z"] = self._zscore_seguro(df_mom["Momentum_beta"])
 
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
 
        # Distribución del Score_total: en vez de comparar tu umbral de compra
        # contra un número fijo a ciegas, mira aquí dónde caen realmente tus
        # candidatos. Si tu umbral (ej. 0.5) está por encima del percentil 90
        # de tu propio universo, es NORMAL que no pase nadie — no es que la
        # estrategia esté "rota", es que le estás pidiendo el top 10% exacto.
        desc = df_final["Score_total"].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        print(f"📊 [{sector}] Distribución Score_total ({len(df_final)} candidatos): "
              f"min={desc['min']:.2f}  p25={desc['25%']:.2f}  mediana={desc['50%']:.2f}  "
              f"p75={desc['75%']:.2f}  p90={desc['90%']:.2f}  max={desc['max']:.2f}")
 
        return df_final
 
    ################################################################
    # NÚCLEO COMPARTIDO — funciones PURAS sobre DataFrames OHLC.
    #
    # Funciones PURAS: reciben datos, no descargan nada, no tocan estado.
    # Son la ÚNICA implementación de cada cálculo del fichero.
    # Regla: si un cálculo se necesita en más de un sitio, va AQUÍ. Tener
    # dos versiones del mismo indicador es como empiezan las incoherencias
    # (ya pasó con el ADX: había una copia propia y la de la librería 'ta').
    ################################################################
    @staticmethod
    def _normalizar_ohlc(data):
        """yfinance devuelve columnas MultiIndex cuando descargas con ciertos
        parámetros o varios tickers. Si no se aplana, data["Close"] devuelve
        un DataFrame de 1 columna en vez de una Series, y cualquier
        comparación posterior ('precio <= stop') produce una Series ->
        'ValueError: truth value of a Series is ambiguous' -> lo traga el
        except -> la posición nunca se cierra. Normalizamos SIEMPRE aquí."""
        if isinstance(data.columns, pd.MultiIndex):
            data = data.copy()
            data.columns = data.columns.droplevel(1)
        return data
 
    @staticmethod
    def _serie_atr(high, low, close, window=14):
        """ATR (Average True Range) como Series. Función pura: recibe OHLC,
        no descarga nada, así que se puede aplicar a cualquier serie
        histórica y obtener siempre el mismo número."""
        close_prev = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - close_prev).abs(),
            (low - close_prev).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()
 
    # NOTA: aquí había un _serie_adx() propio (implementación Wilder) que
    # escribí para el backtest. Al eliminar el backtest quedó sin usar, y
    # tener DOS implementaciones distintas del mismo indicador en el mismo
    # fichero es como empiezan las incoherencias. El ADX se calcula en un
    # único sitio: _analizar_tendencia(), con la librería `ta`.
 
    @staticmethod
    def _momentum_beta(serie_close, window_sma=20, window_reg=60):
        """Pendiente de la regresión lineal del log(SMA). Función pura.
        Devuelve np.nan si no hay datos suficientes (en vez de lanzar)."""
        from sklearn.linear_model import LinearRegression
        serie = serie_close.dropna()
        if len(serie) < window_sma + window_reg // 2:
            return np.nan
        sma = serie.rolling(window_sma).mean().dropna()
        if len(sma) < window_reg // 2:
            return np.nan
        y_vals = sma.values[-window_reg:]
        if np.any(y_vals <= 0):
            return np.nan
        y = np.log(y_vals)
        x = np.arange(len(y)).reshape(-1, 1)
        return float(LinearRegression().fit(x, y).coef_[0])
 
    @staticmethod
    def _zscore_seguro(serie, min_n=2):
        """z-score que NO devuelve NaN silencioso.
 
        El bug que apareció con BMPS.MI: .std() con n=1 es NaN -> Score_total
        NaN -> 'NaN > umbral' es False -> el candidato desaparece sin error.
        Con std=0 (todos los valores iguales) pasa lo mismo vía división por
        cero. Aquí devolvemos 0.0 (= "está en la media", neutro) y avisamos,
        en vez de propagar NaN."""
        n = serie.notna().sum()
        if n < min_n:
            print(f"⚠️ z-score con n={n} (<{min_n}): no es estadísticamente válido. "
                  f"Devuelvo 0.0 (neutro) en vez de NaN para no perder el candidato en silencio.")
            return pd.Series(0.0, index=serie.index)
        std = serie.std()
        if std == 0 or pd.isna(std):
            print("⚠️ z-score con desviación típica 0 (todos los valores iguales). Devuelvo 0.0 (neutro).")
            return pd.Series(0.0, index=serie.index)
        return (serie - serie.mean()) / std
 
    def calcular_stop_desde_atr(self, precio_entrada, atr_valor):
        """Calcula (stop_price, stop_pct, metodo, detalle) a partir de un ATR
        YA calculado. Si atr_valor es NaN -> fallback al stop porcentual."""
        if not self.usar_stop_atr or atr_valor is None or pd.isna(atr_valor):
            stop_pct = self.stop_loss_pct
            metodo = "PORCENTUAL" if not self.usar_stop_atr else "PORCENTUAL (fallback, ATR n/d)"
            return precio_entrada * (1 - stop_pct), stop_pct, metodo, f"{stop_pct*100:.1f}% fijo"
 
        distancia = self.atr_multiplo * atr_valor
        stop_pct = distancia / precio_entrada
        recortado = False
        if stop_pct > self.stop_atr_pct_max:
            stop_pct = self.stop_atr_pct_max
            distancia = precio_entrada * stop_pct
            recortado = True
 
        detalle = (f"ATR({self.atr_window})={atr_valor:.4f} ({atr_valor/precio_entrada*100:.2f}% del precio), "
                   f"stop = {precio_entrada:.2f} - {self.atr_multiplo}xATR = {distancia:.4f}")
        if recortado:
            detalle += f"  [RECORTADO al tope {self.stop_atr_pct_max*100:.0f}%]"
        return precio_entrada - distancia, stop_pct, "ATR", detalle
 
    def evaluar_reglas_salida(self, pos, precio_actual, adx_actual, fecha_actual, sigue_en_ranking=True):
        """REGLAS DE SALIDA — ÚNICA fuente de verdad.
 
        Devuelve (motivo|None, dict_de_checks). Evalúa TODAS las condiciones
        (no corta en la primera) para poder reportar el diagnóstico completo.
 
        Orden de prioridad: stop loss primero (es protección de capital, debe
        reportarse como tal aunque coincida con el time-exit ese mismo día).
        """
        dias = (fecha_actual - pos["fecha_entrada"]).days
        take_profit_price = pos["entrada"] * (1 + self.take_profit_pct)
 
        checks = {
            "dias": dias,
            "stop": bool(precio_actual <= pos["stop"]),
            "take_profit": bool(precio_actual >= take_profit_price),
            "tiempo": bool(dias >= self.max_holding_dias),
            "momentum": bool((not pd.isna(adx_actual)) and (adx_actual < self.adx_exit_threshold)),
            "fuera_ranking": bool(not sigue_en_ranking),
            "take_profit_price": take_profit_price,
        }
 
        if checks["stop"]:
            motivo = f"stop loss tocado ({precio_actual:.2f} <= {pos['stop']:.2f})"
        elif checks["take_profit"]:
            motivo = f"take profit alcanzado ({precio_actual:.2f} >= {take_profit_price:.2f}, +{self.take_profit_pct*100:.0f}%)"
        elif checks["tiempo"]:
            motivo = f"time-based exit ({dias}d >= {self.max_holding_dias}d)"
        elif checks["momentum"]:
            motivo = f"momentum muerto (ADX={adx_actual:.1f} < {self.adx_exit_threshold})"
        elif checks["fuera_ranking"]:
            motivo = "salió del top-N del ranking"
        else:
            motivo = None
 
        return motivo, checks
 
    ################################################################
    # STOP LOSS POR ATR — adapta el stop a la volatilidad de cada activo
    ################################################################
    def obtener_precio_actual(self, ticker):
        """Precio actual robusto. Devuelve float o np.nan (NUNCA una Series).
 
        Antes había dos fallbacks (en comprar y en vender) que hacían
        yf.download(...)["Close"].iloc[-1] SIN normalizar el MultiIndex ->
        devolvían Series -> comparaciones ambiguas -> excepción tragada."""
        try:
            precio = yf.Ticker(ticker).get_info().get("currentPrice")
            if precio:
                return float(precio)
        except Exception:
            pass
 
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            data = self._normalizar_ohlc(data)
            if data.empty:
                return np.nan
            valor = data["Close"].dropna().iloc[-1]
            return float(valor)
        except Exception as e:
            print(f"⚠️ No pude obtener precio de {ticker}: {e}")
            return np.nan
 
    def calcular_atr(self, ticker, window=None, periodo="6mo"):
        """
        Calcula el ATR (Average True Range) del ticker.
 
        True Range = max de:
          - High - Low                (rango del día)
          - |High - Close_anterior|   (gap alcista)
          - |Low  - Close_anterior|   (gap bajista)
 
        El ATR es la media móvil de ese TR. Mide el recorrido TÍPICO diario
        del activo en unidades de precio (no en %), incluyendo huecos de
        apertura -- por eso es mejor que la desviación típica de los cierres
        para colocar stops.
 
        Devuelve (atr_valor, atr_como_pct_del_precio) o (nan, nan) si falla.
        """
        if window is None:
            window = self.atr_window
 
        try:
            data = yf.download(ticker, period=periodo, interval="1d",
                                progress=False, auto_adjust=True)
            data = self._normalizar_ohlc(data)
 
            if data.empty or len(data) < window + 1:
                print(f"⚠️ {ticker}: datos insuficientes para ATR({window}).")
                return np.nan, np.nan
 
            # Usa la función pura compartida -> mismo número en todo el fichero
            atr = self._serie_atr(data["High"], data["Low"], data["Close"], window).iloc[-1]
            precio_actual = data["Close"].iloc[-1]
 
            if pd.isna(atr) or precio_actual <= 0:
                return np.nan, np.nan
 
            return float(atr), float(atr / precio_actual)
 
        except Exception as e:
            print(f"⚠️ Error calculando ATR de {ticker}: {e}")
            return np.nan, np.nan
 
    def calcular_stop_loss(self, ticker, precio_entrada):
        """
        Devuelve (stop_price, stop_pct, metodo_usado, detalle_str).
 
        Si usar_stop_atr=True y el ATR se puede calcular:
            stop = precio_entrada - atr_multiplo x ATR
        Si no (datos insuficientes, o usar_stop_atr=False):
            stop = precio_entrada x (1 - stop_loss_pct)   [fallback clásico]
 
        Se aplica un tope `stop_atr_pct_max`: si el ATR sugiere un stop más
        lejano que ese %, se recorta. Un activo muy volátil podría pedir un
        stop del 35-40%, y eso es más riesgo del que quieres en un solo trade
        aunque estadísticamente sea "el stop correcto" para ese activo.
        """
        atr, _ = self.calcular_atr(ticker) if self.usar_stop_atr else (np.nan, np.nan)
        # Delega en la función compartida (misma fórmula, tope y fallback).
        return self.calcular_stop_desde_atr(precio_entrada, atr)
 
    ################################################################
    # POSITION SIZING — lo que v1 NO tenía (compraba 1 acción fija)
    ################################################################
    def calcular_tamano_posicion(self, capital, precio, stop_loss_pct=None):
        """
        Fixed-fractional sizing: arriesga como mucho `riesgo_max_por_trade`
        del capital en cada trade, asumiendo que el stop se ejecuta.
        Limita además la concentración a max_pct_por_posicion del capital.
        """
        if stop_loss_pct is None:
            stop_loss_pct = self.stop_loss_pct
 
        riesgo_dinero = capital * self.riesgo_max_por_trade
        perdida_por_accion = precio * stop_loss_pct
        if perdida_por_accion <= 0:
            return 0
 
        cantidad = riesgo_dinero / perdida_por_accion
        valor_posicion = cantidad * precio
 
        # Tope de concentración (ya NO hardcodeado): con 10 posiciones y 10%
        # cada una, el capital comprometido no puede pasar del 100%.
        max_valor_posicion = capital * self.max_pct_por_posicion
        if valor_posicion > max_valor_posicion:
            cantidad = max_valor_posicion / precio
 
        return int(cantidad)
 
    ################################################################
    # TENDENCIA — ADX + DI direccional
    #
    # BUG CORREGIDO (ADX=nan): la librería `ta` propaga NaN a TODA la serie
    # si hay UN SOLO NaN en la entrada. Verificado empíricamente: con un
    # único NaN a mitad de serie, el ADX final pasa de 15,82 a nan.
    #
    # Por qué pasaba justo con bancos italianos/polacos y no con los de
    # Londres: yfinance devuelve filas vacías para los días en que ESE
    # mercado está cerrado pero otros abren (calendarios de festivos
    # distintos entre LSE, Borsa Italiana, GPW, OSE...). Basta un festivo
    # local en 6 meses para dejar el ADX del ticker en NaN para siempre.
    #
    # Solución: dropna() ANTES de calcular, y diagnóstico explícito cuando
    # aun así no se pueda calcular (en vez de devolver un nan mudo).
    ################################################################
    def _analizar_tendencia(self, ticker, direccion="UP", periodo="6mo"):
        """Motor común de UP y DOWN (antes eran 25 líneas duplicadas que
        solo se diferenciaban en el signo de la comparación DI+/DI-).
 
        Devuelve (cumple_condicion, valor_adx).
        """
        import ta
        try:
            data = yf.download(ticker, period=periodo, interval="1d",
                                progress=False, auto_adjust=True)
            data = self._normalizar_ohlc(data)
 
            if data.empty:
                print(f"⚠️ {ticker}: sin datos descargados.")
                return False, np.nan
 
            # CLAVE: eliminar filas incompletas antes de calcular indicadores.
            n_antes = len(data)
            data = data.dropna(subset=["High", "Low", "Close"]).copy()
            n_descartadas = n_antes - len(data)
            if n_descartadas > 0:
                print(f"   ℹ️ {ticker}: {n_descartadas} sesión(es) sin datos descartadas "
                      f"(festivos del mercado local). Quedan {len(data)}.")
 
            # ADX(14) necesita ~2x ventana para estabilizarse
            minimo = 14 * 2 + 5
            if len(data) < minimo:
                print(f"⚠️ {ticker}: solo {len(data)} sesiones válidas (mínimo {minimo} "
                      f"para un ADX(14) fiable). No se puede evaluar la tendencia.")
                return False, np.nan
 
            adx_ind = ta.trend.ADXIndicator(high=data["High"], low=data["Low"],
                                             close=data["Close"], window=14)
            adx = adx_ind.adx().iloc[-1]
            adx_pos = adx_ind.adx_pos().iloc[-1]
            adx_neg = adx_ind.adx_neg().iloc[-1]
 
            if pd.isna(adx) or pd.isna(adx_pos) or pd.isna(adx_neg):
                print(f"⚠️ {ticker}: el ADX sigue siendo NaN tras limpiar los datos "
                      f"({len(data)} sesiones). Revisa manualmente este ticker.")
                return False, np.nan
 
            if direccion == "UP":
                cumple = bool((adx > 25) and (adx_neg < adx_pos))
            else:
                cumple = bool((adx > 25) and (adx_neg > adx_pos))
 
            return cumple, float(adx)
 
        except Exception as e:
            print(f"Error al analizar {ticker}: {e}")
            return False, np.nan
 
    def analizar_tendencia_UP(self, ticker, periodo="6mo"):
        """True si hay tendencia ALCISTA fuerte: ADX > 25 y DI+ > DI-."""
        return self._analizar_tendencia(ticker, direccion="UP", periodo=periodo)
 
    def analizar_tendencia_DOWN(self, ticker, periodo="6mo"):
        """True si hay tendencia BAJISTA fuerte: ADX > 25 y DI- > DI+."""
        return self._analizar_tendencia(ticker, direccion="DOWN", periodo=periodo)
 
    ################################################################
    # COMPRA / VENTA con exit rules reales (v1 no tenía ninguna)
    ################################################################
    def comprar(self, ticker, sector='default', capital=100000, fila=None):
        """
        Compra SOLO si: (a) hay hueco en el libro (max_posiciones_abiertas),
        (b) la tendencia de corto plazo confirma (ADX+SMA), (c) hay
        position sizing con stop calculado. A diferencia de v1, aquí
        SÍ se registra la posición para poder aplicarle exit rules después.
 
        `fila` (opcional): fila de df_final (pandas Series) con TODOS los
        marcadores usados para seleccionar este ticker (Value_z, Quality_z,
        Momentum_z, Score_total, P/E, P/B, ROE, ROA, etc.). Si se pasa, se
        imprime en consola Y se manda por Telegram un desglose completo de
        qué se evaluó y qué dio cada cálculo -- igual que ya hacemos en
        vender_con_estrategia() para las señales de venta.
        """
        if ticker in self.posiciones_abiertas:
            print(f"ℹ️ {ticker}: ya está comprado (desde {self.posiciones_abiertas[ticker]['fecha_entrada'].date()}), no duplico posición.")
            return False
 
        if len(self.posiciones_abiertas) >= self.max_posiciones_abiertas:
            print(f"❌ {ticker}: portfolio lleno ({self.max_posiciones_abiertas} posiciones máx).")
            return False
 
        tendencia_ok, adx = self.analizar_tendencia_UP(ticker)
        if not tendencia_ok:
            print(f"❌ {ticker}: tendencia no confirma (ADX={adx:.1f}).")
            return False
 
        precio = self.obtener_precio_actual(ticker)
        if pd.isna(precio) or precio <= 0:
            print(f"❌ {ticker}: no pude obtener un precio válido.")
            return False
 
        # STOP PRIMERO, sizing DESPUÉS: el tamaño de posición depende de
        # dónde esté el stop (arriesgas riesgo_max_por_trade hasta el stop).
        # Si calculas el sizing con un 10% fijo pero luego pones el stop a
        # 2xATR = 6%, estarías arriesgando MENOS del 2% previsto; y si el ATR
        # da 18%, arriesgarías MUCHO MÁS. Tienen que ser el mismo número.
        stop_price, stop_pct, metodo_stop, detalle_stop = self.calcular_stop_loss(ticker, precio)
 
        cantidad = self.calcular_tamano_posicion(capital, precio, stop_loss_pct=stop_pct)
        if cantidad <= 0:
            print(f"❌ {ticker}: tamaño de posición calculado = 0.")
            return False
 
        self.posiciones_abiertas[ticker] = {
            "entrada": precio,
            "cantidad": cantidad,
            "fecha_entrada": datetime.now(),
            "stop": stop_price,
            "stop_pct": stop_pct,
            "stop_metodo": metodo_stop,
            "sector": sector,
            # Guardamos los factores de ENTRADA en la propia posición. Así,
            # cuando se venda, el registro de la venta puede incluir con qué
            # Score se compró -> es lo que permite después correlacionar
            # "Score alto" con "buen resultado". Sin esto, el histórico tendría
            # los datos de compra y de venta en registros sueltos sin enlazar.
            "score_entrada": (float(fila["Score_total"])
                              if fila is not None and pd.notna(fila.get("Score_total", np.nan)) else None),
            "quality_z_entrada": (float(fila["Quality_z"])
                                  if fila is not None and pd.notna(fila.get("Quality_z", np.nan)) else None),
            "value_z_entrada": (float(fila["Value_z"])
                                if fila is not None and pd.notna(fila.get("Value_z", np.nan)) else None),
            "momentum_z_entrada": (float(fila["Momentum_z"])
                                   if fila is not None and pd.notna(fila.get("Momentum_z", np.nan)) else None),
            "adx_entrada": float(adx) if not pd.isna(adx) else None,
        }
 
        self.guardar_posiciones()  # persiste a disco: si no, esto se olvida al terminar el proceso
 
        w = self.obtener_filtros_sector(sector)['weights']
 
        # --- REGISTRO EN EL HISTÓRICO ---
        # Guardamos TODO lo que motivó la compra, no solo precio y cantidad.
        # Sin los z-scores y los ratios, dentro de un año no podrás responder
        # a "¿los candidatos con Score alto rindieron mejor?", que es la única
        # pregunta que valida o tumba la estrategia.
        def _v(col):
            """Extrae un valor de `fila` de forma segura (None si no está)."""
            if fila is None:
                return None
            val = fila.get(col, None)
            return None if val is None or (isinstance(val, float) and pd.isna(val)) else val
 
        self._registrar_operacion({
            "tipo": "COMPRA",
            "ticker": ticker,
            "sector": sector,
            "universo": getattr(self, "etiqueta_actual", "general"),
            # --- ejecución ---
            "precio": round(float(precio), 4),
            "cantidad": int(cantidad),
            "valor_operacion": round(float(precio) * int(cantidad), 2),
            "capital_referencia": capital,
            # --- gestión de riesgo ---
            "stop": round(float(stop_price), 4),
            "stop_pct": round(float(stop_pct), 4),
            "stop_metodo": metodo_stop,
            "take_profit": round(float(precio) * (1 + self.take_profit_pct), 4),
            "riesgo_eur": round(float(precio) * int(cantidad) * float(stop_pct), 2),
            "adx_entrada": round(float(adx), 2) if not pd.isna(adx) else None,
            # --- factores que motivaron la selección ---
            "score_total": _v("Score_total"),
            "quality_z": _v("Quality_z"),
            "value_z": _v("Value_z"),
            "momentum_z": _v("Momentum_z"),
            "momentum_beta": _v("Momentum_beta"),
            "ranking": _v("Ranking_Total"),
            "peso_quality": w['quality'],
            "peso_value": w['value'],
            "peso_momentum": w['momentum'],
            # --- fundamentales del momento de la compra ---
            "pe": _v("P/E"),
            "pb": _v("P/B"),
            "roe": _v("ROE"),
            "roa": _v("ROA"),
            "margen_neto": _v("ProfitMargin"),
            "crecim_beneficios": _v("EarningsGrowth"),
            "crecim_ingresos": _v("RevenueGrowth"),
            "quality_n_datos": _v("Quality_n_datos"),
        })
 
        # --- Desglose de marcadores y cálculos, consola + Telegram ---
        # Si no se pasó `fila` (llamada directa sin pasar por el ranking),
        # igualmente mostramos lo que SÍ tenemos (precio, sizing, ADX).
        etiq = getattr(self, "etiqueta_actual", "general").upper()
        lineas_consola = [
            "",
            f"🟢 [{etiq}] SEÑAL DE COMPRA — {ticker}",
            "───────────────────────────────────────────────",
            f"  Universo:              {etiq}",
            f"  Sector (filtros):      {sector}",
            f"  Precio entrada:        {precio:.2f}",
            f"  Cantidad:              {cantidad}",
            f"  Stop loss [{metodo_stop}]:  {stop_price:.2f}  (-{stop_pct*100:.2f}%)",
            f"     └─ {detalle_stop}",
            f"  Riesgo del trade:      {cantidad * precio * stop_pct:.2f} ({self.riesgo_max_por_trade*100:.0f}% objetivo del capital)",
            f"  Take profit objetivo:  {precio * (1 + self.take_profit_pct):.2f}  (+{self.take_profit_pct*100:.0f}%)",
            f"  ADX (tendencia UP):    {adx:.2f}  (umbral 25, confirma alcista)",
        ]
 
        if fila is not None:
            lineas_consola += [
                "  --- VALUE (marcadores y resultado) ---",
                f"  P/E:                   {fila.get('P/E', float('nan')):.2f}",
                f"  P/B:                   {fila.get('P/B', float('nan')):.2f}",
                f"  Value_z (z-score):     {fila.get('Value_z', float('nan')):+.2f}",
                "  --- QUALITY (marcadores y resultado) ---",
                f"  ROE:                   {fila.get('ROE', float('nan')):.2%}" if pd.notna(fila.get('ROE', np.nan)) else "  ROE:                   n/a",
                f"  ROA:                   {fila.get('ROA', float('nan')):.2%}" if pd.notna(fila.get('ROA', np.nan)) else "  ROA:                   n/a",
                f"  Margen neto:           {fila.get('ProfitMargin', float('nan')):.2%}" if pd.notna(fila.get('ProfitMargin', np.nan)) else "  Margen neto:           n/a",
                f"  Crecim. beneficios:    {fila.get('EarningsGrowth', float('nan')):.2%}" if pd.notna(fila.get('EarningsGrowth', np.nan)) else "  Crecim. beneficios:    n/a",
                f"  Crecim. ingresos:      {fila.get('RevenueGrowth', float('nan')):.2%}" if pd.notna(fila.get('RevenueGrowth', np.nan)) else "  Crecim. ingresos:      n/a",
                f"  Quality_z (z-score):   {fila.get('Quality_z', float('nan')):+.2f}"
                + (f"   [calculado sobre {int(fila['Quality_n_datos'])} indicadores]"
                   if pd.notna(fila.get('Quality_n_datos', np.nan)) else ""),
                "  --- MOMENTUM (marcadores y resultado) ---",
                f"  Momentum_beta:         {fila.get('Momentum_beta', float('nan')):+.5f}  (pendiente regresión log-SMA20, ventana 60d)",
                f"  Momentum_z (z-score):  {fila.get('Momentum_z', float('nan')):+.2f}",
                "  --- SCORE FINAL (pesos del sector aplicados) ---",
                f"  Pesos usados:          Quality={w['quality']:.0%}  Value={w['value']:.0%}  Momentum={w['momentum']:.0%}",
                f"  Cálculo:               {w['quality']:.2f}×{fila.get('Quality_z', float('nan')):+.2f} + {w['value']:.2f}×{fila.get('Value_z', float('nan')):+.2f} + {w['momentum']:.2f}×{fila.get('Momentum_z', float('nan')):+.2f}",
                f"  Score_total:           {fila.get('Score_total', float('nan')):+.2f}",
                f"  Ranking en el universo: #{int(fila['Ranking_Total'])}" if pd.notna(fila.get('Ranking_Total', np.nan)) else "  Ranking en el universo: n/a",
            ]
        else:
            lineas_consola.append("  (Sin `fila` de ranking pasada a comprar() -- no hay desglose de Value/Quality/Momentum disponible)")
 
        lineas_consola.append("───────────────────────────────────────────────\n")
 
        texto_completo = "\n".join(lineas_consola)
        print(texto_completo)
 
        # Telegram: mismo contenido, sin las líneas separadoras decorativas
        # (para que no ocupe tanto en el chat, pero con todos los números)
        texto_telegram = texto_completo.replace("───────────────────────────────────────────────\n", "").replace("───────────────────────────────────────────────", "")
        # La posición YA está registrada y persistida arriba. Si Telegram falla
        # aquí, la compra es válida igualmente -> no dejamos que propague, o
        # abortaría el bucle de compras del universo entero.
        self._notificar(texto_telegram, critico=True)
 
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
 
        Reglas evaluadas (ver evaluar_reglas_salida, por orden de prioridad):
          1) Stop loss      -> precio <= stop (ATR o porcentual)
          2) Take profit    -> precio >= entrada x (1 + take_profit_pct)
          3) Time-based     -> dias_en_posicion >= max_holding_dias
          4) Momentum muerto-> ADX < adx_exit_threshold
        """
        tickers = list(self.posiciones_abiertas.keys())
        if not tickers:
            print("ℹ️ No hay posiciones abiertas.")
            return True
 
        for t in tickers:
            try:
                pos = self.posiciones_abiertas[t]
 
                # VALIDACIÓN de lo leído del JSON: si el fichero viene de una
                # versión anterior del script (o fue editado a mano), pueden
                # faltar campos. Sin esto, un KeyError aquí aborta la venta de
                # ESE ticker y la posición se queda abierta para siempre sin
                # que nadie le aplique stop ni time-exit.
                campos_obligatorios = ["entrada", "cantidad", "fecha_entrada", "stop"]
                faltan = [c for c in campos_obligatorios if c not in pos]
                if faltan:
                    print(f"⚠️ {t}: faltan campos {faltan} en {self.posiciones_file}. "
                          f"Salto esta posición (revisa el JSON a mano).")
                    continue
 
                if not isinstance(pos["fecha_entrada"], datetime):
                    print(f"⚠️ {t}: 'fecha_entrada' no es datetime tras cargar el JSON "
                          f"(es {type(pos['fecha_entrada']).__name__}). Salto esta posición.")
                    continue
 
                # stop_pct/stop_metodo son campos NUEVOS (stop por ATR). Si el
                # JSON es antiguo no existen: los derivamos del stop guardado
                # en vez de asumir el self.stop_loss_pct de clase, que podría
                # no ser el que realmente se aplicó al comprar.
                stop_pct_real = pos.get("stop_pct")
                if stop_pct_real is None:
                    stop_pct_real = (pos["entrada"] - pos["stop"]) / pos["entrada"]
                stop_metodo_real = pos.get("stop_metodo", "desconocido (JSON antiguo)")
 
                dias_en_posicion = (datetime.now() - pos["fecha_entrada"]).days
 
                precio_actual = self.obtener_precio_actual(t)
                if pd.isna(precio_actual) or precio_actual <= 0:
                    print(f"⚠️ {t}: sin precio válido hoy, no puedo evaluar salida. "
                          f"La posición SIGUE ABIERTA -- revísala manualmente.")
                    continue
 
                _, adx_actual = self.analizar_tendencia_UP(t)
 
                pnl_pct = (precio_actual - pos["entrada"]) / pos["entrada"] * 100
                pnl_dinero = (precio_actual - pos["entrada"]) * pos["cantidad"]
 
                # Reglas de salida centralizadas en evaluar_reglas_salida().
                motivo, checks = self.evaluar_reglas_salida(
                    pos, precio_actual, adx_actual, datetime.now(), sigue_en_ranking=True
                )
                check_tiempo = checks["tiempo"]
                check_momentum = checks["momentum"]
                check_stop = checks["stop"]
                check_take_profit = checks["take_profit"]
                take_profit_price = checks["take_profit_price"]
 
                if motivo:
                    sector_pos = pos.get('sector', 'default')
                    n_disparos = sum([check_stop, check_take_profit, check_tiempo, check_momentum])
                    signo = "🟢" if pnl_pct >= 0 else "🔴"
 
                    # Construimos el texto UNA sola vez y lo mandamos a consola
                    # y a Telegram, igual que hacemos en comprar(). Antes la
                    # consola recibía todo el desglose y Telegram solo una línea:
                    # justo al revés de lo útil, porque cuando salta un stop
                    # normalmente no estás delante de la consola.
                    lineas_venta = [
                        "",
                        f"⚠️ [{sector_pos.upper()}] SEÑAL DE VENTA — {t}",
                        "───────────────────────────────────────────────",
                        f"  Sector:                {sector_pos}",
                        f"  Fecha entrada:         {pos['fecha_entrada'].date()}   ({dias_en_posicion} días en cartera)",
                        f"  Cantidad:              {pos['cantidad']}",
                        f"  Entrada → actual:      {pos['entrada']:.2f} → {precio_actual:.2f}",
                        f"  {signo} P&L:                {pnl_pct:+.2f}%   ({pnl_dinero:+.2f})",
                        "  --- Chequeos evaluados (valores y resultado) ---",
                        f"  1) Stop loss [{stop_metodo_real}]: {precio_actual:.2f} vs {pos['stop']:.2f} "
                        f"(entrada -{stop_pct_real*100:.2f}%)  -> {'DISPARA' if check_stop else 'ok'}",
                        f"  2) Take profit:        {precio_actual:.2f} vs {take_profit_price:.2f} "
                        f"(entrada +{self.take_profit_pct*100:.0f}%)  -> {'DISPARA' if check_take_profit else 'ok'}",
                        f"  3) Time-based exit:    {dias_en_posicion} vs {self.max_holding_dias} días"
                        f"  -> {'DISPARA' if check_tiempo else 'ok'}",
                        f"  4) Momentum muerto:    ADX {adx_actual:.2f} vs umbral {self.adx_exit_threshold}"
                        f"  -> {'DISPARA' if check_momentum else 'ok'}",
                        f"  MOTIVO DE VENTA:       {motivo}",
                    ]
                    # Si saltan varios criterios a la vez, la salida NO es
                    # marginal: la posición estaba mal por varios frentes.
                    # Es información que la línea corta anterior ocultaba.
                    if n_disparos > 1:
                        lineas_venta.append(
                            f"  ⚠️ {n_disparos} de 4 criterios disparan a la vez: "
                            f"la posición estaba deteriorada por varios frentes, no es una salida marginal.")
                    lineas_venta.append("───────────────────────────────────────────────\n")
 
                    texto_venta = "\n".join(lineas_venta)
                    print(texto_venta)
 
                    # ORDEN CRÍTICO: cerrar y persistir PRIMERO, notificar DESPUÉS.
                    # Antes era al revés: si send_message fallaba, la excepción
                    # saltaba al except de abajo y la posición NUNCA se cerraba
                    # -- se había detectado el stop loss y seguía abierta, con
                    # el único rastro de un "❌ Error procesando venta".
                    # --- REGISTRO EN EL HISTÓRICO (antes de borrar la posición,
                    #     que es de donde salen los datos de entrada) ---
                    self._registrar_operacion({
                        "tipo": "VENTA",
                        "ticker": t,
                        "sector": sector_pos,
                        # --- ejecución ---
                        "precio": round(float(precio_actual), 4),
                        "cantidad": int(pos["cantidad"]),
                        "valor_operacion": round(float(precio_actual) * int(pos["cantidad"]), 2),
                        # --- RESULTADO (lo que de verdad importa) ---
                        "precio_entrada": round(float(pos["entrada"]), 4),
                        "pnl_pct": round(float(pnl_pct), 4),
                        "pnl_dinero": round(float(pnl_dinero), 2),
                        "ganadora": bool(pnl_pct > 0),
                        "fecha_entrada": pos["fecha_entrada"].isoformat(),
                        "dias_en_posicion": int(dias_en_posicion),
                        # --- por qué se vendió ---
                        "motivo": motivo,
                        "motivo_corto": motivo.split("(")[0].strip(),
                        "n_criterios_disparados": int(n_disparos),
                        "check_stop": bool(check_stop),
                        "check_take_profit": bool(check_take_profit),
                        "check_tiempo": bool(check_tiempo),
                        "check_momentum": bool(check_momentum),
                        # --- estado de los indicadores al salir ---
                        "adx_salida": round(float(adx_actual), 2) if not pd.isna(adx_actual) else None,
                        "stop": round(float(pos["stop"]), 4),
                        "stop_pct": round(float(stop_pct_real), 4),
                        "stop_metodo": stop_metodo_real,
                        # --- FACTORES DE ENTRADA (enlace compra<->venta):
                        #     esto es lo que permite responder a "¿el Score
                        #     alto predijo mejor resultado?" -- la pregunta que
                        #     valida o tumba toda la estrategia.
                        "score_total": pos.get("score_entrada"),
                        "quality_z": pos.get("quality_z_entrada"),
                        "value_z": pos.get("value_z_entrada"),
                        "momentum_z": pos.get("momentum_z_entrada"),
                        "adx_entrada": pos.get("adx_entrada"),
                    })
 
                    del self.posiciones_abiertas[t]
                    self.guardar_posiciones()
 
                    texto_telegram_venta = texto_venta.replace(
                        "───────────────────────────────────────────────\n", "").replace(
                        "───────────────────────────────────────────────", "")
                    self._notificar(texto_telegram_venta, critico=True)
 
                    """
                    import sys, importlib
                    sys.path.append("C:\\Users\\jjjimenez\\Documents\\quant\\999_Automatic\\999_Automatic")
                    automatic = importlib.import_module("automatic", "...")
                    alpacaAPI = automatic.tradeAPIClass(para2=automatic.CUENTA_J3_03)
                    alpacaAPI.placeOrderSell(t, pos["cantidad"])
                    """
                else:
                    print(f"✅ {t}: se mantiene | {dias_en_posicion}/{self.max_holding_dias}d | "
                          f"ADX={adx_actual:.1f} (sale <{self.adx_exit_threshold}) | "
                          f"precio={precio_actual:.2f} (stop {pos['stop']:.2f} [{stop_metodo_real}], "
                          f"TP {take_profit_price:.2f}) | P&L {pnl_pct:+.2f}%")
 
            except Exception as e:
                print(f"❌ Error procesando venta de {t}: {e}")
 
        return True
 
    ################################################################
    # GRAFICOS (mismo estilo que v1)
    ################################################################
    def graficar_spider(self, df_final, top_n=8, etiqueta="qvm"):
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
        ax.set_title(f"Perfil Quality / Value / Momentum — Top {top_n} ({etiqueta})", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)
        plt.tight_layout()
 
        # Nombre de fichero POR UNIVERSO: si no, al procesar defensa después de
        # bancos el segundo gráfico sobrescribía al primero y solo veías uno.
        nombre_fich = os.path.join(_DIR_SCRIPT, f"spider_{etiqueta}.png")
        fig.savefig(nombre_fich, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        self._notificar_imagen(nombre_fich, caption=f"Spider Quality/Value/Momentum — {etiqueta}")
 
    def graficar_ranking(self, df_final, top_n=10, etiqueta="qvm"):
        import matplotlib.pyplot as plt
        df_sorted = df_final.sort_values("Score_total", ascending=False).reset_index(drop=True)
 
        print(f"\n🏁 Ranking Quality+Value+Momentum — {etiqueta}:\n")
        cols_mostrar = ["Ticker", "Quality_z", "Value_z", "Momentum_z", "Score_total"]
        print(df_sorted[cols_mostrar].head(top_n).to_string(index=False))
 
        top_df = df_sorted.head(top_n)
        fig = plt.figure(figsize=(10, 6))
        plt.barh(top_df["Ticker"], top_df["Score_total"], color="dodgerblue", alpha=0.8)
        plt.xlabel("Score Total (Quality+Value+Momentum)")
        plt.title(f"Top {top_n} — QVM Score ({etiqueta})")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        # Línea del suelo de compra: desde que existe min_score_compra el
        # criterio SÍ es un umbral absoluto, así que pintarlo informa.
        plt.axvline(x=self.min_score_compra, color="green", linestyle="--", linewidth=2,
                    label=f"Suelo de compra ({self.min_score_compra:+.2f})")
        plt.legend(fontsize=8)
        nombre_fich = os.path.join(_DIR_SCRIPT, f"score_{etiqueta}.png")
        fig.savefig(nombre_fich, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        self._notificar_imagen(nombre_fich, caption=f"Ranking QVM — {etiqueta}")
 
    ################################################################
    # PROCESADO DE UN UNIVERSO — reutilizable para bancos, defensa, etc.
    ################################################################
    def procesar_universo(self, tickers, sector, etiqueta, max_posiciones_sector=None,
                           top_grafico=8):
        """
        Ejecuta el pipeline completo de SEÑALES DE COMPRA para un universo:
        Quality+Value -> Momentum -> Score -> gráficos -> compras.
 
        NO vende: la venta se hace UNA sola vez para toda la cartera antes
        de llamar a esta función (las reglas de salida son por ticker, no por
        sector; llamarla por universo evaluaría dos veces lo mismo).
 
        max_posiciones_sector: tope de posiciones ABIERTAS de este universo.
            Si es None, usa el global self.max_posiciones_abiertas.
            Sirve para que el primer sector procesado no se coma todas las
            plazas y deje al segundo sin ninguna (ver aviso en el main).
 
        Devuelve el df_final del universo (o None si no hubo candidatos).
        """
        # Marca TODOS los mensajes de aquí en adelante con esta etiqueta
        self.etiqueta_actual = etiqueta
 
        print(f"\n{'='*70}")
        print(f"  UNIVERSO: {etiqueta.upper()}   (sector de filtros: '{sector}')")
        print(f"  {len(tickers)} tickers en la lista")
        print(f"{'='*70}")
 
        try:
            send_message(f"🔎 [{etiqueta.upper()}] Analizando universo ({len(tickers)} tickers)")
        except Exception:
            pass
 
        df_qv = self.obtener_composite_quality_value_tickers(tickers, sector=sector)
        df_mom = self.calcular_momentum_regresion_tickers(tickers)
        df_final = self.calcular_score_final(df_qv, df_mom, sector=sector)
 
        if df_final is None or df_final.empty:
            msg = f"⚠️ [{etiqueta}] Sin candidatos hoy (no superan filtros o faltan datos)."
            print(msg)
            try:
                send_message(msg)
            except Exception:
                pass
            return None
 
        # Los gráficos son COSMÉTICOS: un fallo de matplotlib (backend sin
        # display en cron, permisos de escritura del PNG) no debe impedir que
        # se ejecuten las compras, que es la parte operativa.
        try:
            self.graficar_spider(df_final, top_n=top_grafico, etiqueta=etiqueta)
            self.graficar_ranking(df_final, top_n=top_grafico, etiqueta=etiqueta)
        except Exception as e:
            print(f"  ⚠️ No pude generar los gráficos de {etiqueta}: {e}. "
                  f"Continúo con el análisis (los gráficos no afectan a las decisiones).")
 
        # --- Plazas disponibles ---
        # Dos topes: el del sector y el global de la cartera. Manda el menor.
        abiertas_sector = sum(1 for p in self.posiciones_abiertas.values()
                              if p.get("sector") == sector)
        tope_sector = max_posiciones_sector if max_posiciones_sector is not None \
            else self.max_posiciones_abiertas
 
        plazas_sector = tope_sector - abiertas_sector
        plazas_global = self.max_posiciones_abiertas - len(self.posiciones_abiertas)
        plazas = min(plazas_sector, plazas_global)
 
        print(f"\n  Posiciones abiertas: {len(self.posiciones_abiertas)}/{self.max_posiciones_abiertas} total, "
              f"{abiertas_sector}/{tope_sector} en {etiqueta}")
 
        if plazas <= 0:
            razon = "tope del sector" if plazas_sector <= 0 else "cartera global llena"
            print(f"  ℹ️ Sin plazas libres para {etiqueta} ({razon}). No se compra.")
            return df_final
 
        ya_en_cartera = set(self.posiciones_abiertas.keys())
        disponibles = df_final[~df_final["Ticker"].isin(ya_en_cartera)]
 
        # SUELO DE CALIDAD: descartamos los que no superan min_score_compra
        # ANTES de intentar comprarlos. Sin esto, si los mejores candidatos
        # eran rechazados por el filtro de ADX, el código iba bajando por el
        # ranking hasta comprar cualquier cosa con tal de llenar la plaza --
        # incluso con Score negativo (= peor que la media de su propio grupo).
        # Los NaN se tratan APARTE: 'NaN > x' y 'NaN <= x' son AMBOS False, así
        # que un candidato con Score NaN no caería ni en aptos ni en
        # descartados -- desaparecería sin que los contadores cuadren. Es el
        # mismo patrón silencioso que _zscore_seguro fue escrito para evitar.
        con_nan = disponibles[disponibles["Score_total"].isna()]
        if not con_nan.empty:
            print(f"     ⚠️ {len(con_nan)} candidato(s) con Score_total = NaN, excluidos: "
                  f"{', '.join(con_nan['Ticker'])}. Revisa sus z-scores.")
        validos = disponibles[disponibles["Score_total"].notna()]
 
        aptos = validos[validos["Score_total"] > self.min_score_compra]
        descartados = validos[validos["Score_total"] <= self.min_score_compra]
 
        print(f"\n  🎯 {plazas} plaza(s) libre(s) en {etiqueta}.")
        print(f"     Candidatos no en cartera: {len(disponibles)}")
        print(f"     Superan el suelo (Score > {self.min_score_compra:+.2f}): {len(aptos)}")
 
        if len(descartados) > 0:
            peores = ", ".join(f"{r['Ticker']}({r['Score_total']:+.2f})"
                               for _, r in descartados.head(5).iterrows())
            print(f"     Descartados por Score insuficiente: {len(descartados)}  [{peores}]")
 
        if aptos.empty:
            msg = (f"⚠️ [{etiqueta.upper()}] Ningún candidato supera el Score mínimo "
                   f"({self.min_score_compra:+.2f}). NO se compra nada hoy: "
                   f"quedarse en liquidez es mejor que comprar el menos malo.")
            print(f"  {msg}")
            try:
                send_message(msg)
            except Exception:
                pass
            return df_final
 
        # Intentamos comprar bajando por el ranking. comprar() aplica ADEMÁS
        # su propio filtro de tendencia (ADX>25 y DI+>DI-), así que un
        # candidato puede ser rechazado ahí; en ese caso pasamos al siguiente,
        # pero SIEMPRE dentro de los que superan el suelo de Score.
        comprados = 0
        rechazados_adx = []
        for _, fila in aptos.iterrows():
            if comprados >= plazas:
                break
            print(f"  🟢 Evaluando compra de {fila['Ticker']} (Score={fila['Score_total']:+.2f})")
            if self.comprar(fila["Ticker"], sector=sector, fila=fila):
                comprados += 1
            else:
                rechazados_adx.append(fila["Ticker"])
 
        print(f"\n  📌 [{etiqueta}] Compradas {comprados} de {plazas} plaza(s) disponible(s).")
        if rechazados_adx:
            print(f"     Rechazados en la fase de compra (tendencia/precio/sizing): "
                  f"{', '.join(rechazados_adx)}")
        if comprados < plazas:
            print(f"     {plazas - comprados} plaza(s) quedan VACÍAS: o no había candidatos "
                  f"por encima del suelo, o los que había no confirman tendencia.")
 
        return df_final
 
 
#################################################### Clase FIN
if __name__ == '__main__':
 
    print('version(J): ', versionVersion)
 
    from telegram_bot import *
 
    objEstra = valueMomentumQualityClass("AMZN")
 
    _TITULO = "Proceso Quality+Value+Momentum — BANCOS + DEFENSA + ENERGÍA"
    # CONSOLA: rojo real vía ANSI.
    print("\n" + _C.rojo(f"🔴 {_TITULO}") + "\n")
    # TELEGRAM: no admite color, así que destacamos con emoji rojo.
    objEstra._notificar(f"🔴 {_TITULO}")
 
    #######################################################################
    #  CONFIGURACIÓN DE UNIVERSOS
    #
    #  TRES universos: bancos + defensa + energía.
    #
    #  Tope GLOBAL 12, con 5 por sector (5x3=15 > 12 a propósito: así ningún
    #  universo se bloquea por el orden en que se procesa, y el tope global
    #  sigue siendo el límite real de la cartera).
    #
    #  IMPORTANTE — la invariante que hay que respetar SIEMPRE:
    #       max_posiciones_abiertas x max_pct_por_posicion <= 1.0
    #  Con 12 posiciones, el máximo por posición baja a 8% (12 x 8% = 96%).
    #  Si subes posiciones y te olvidas de bajar el %, la cartera intentaría
    #  comprometer más capital del que hay. La comprobación de abajo lo avisa.
    #######################################################################
    objEstra.max_posiciones_abiertas = 12        # tope GLOBAL de la cartera
    objEstra.max_pct_por_posicion = 0.08         # 12 x 8% = 96% del capital
 
    UNIVERSOS = [
        {
            "etiqueta": "bancos",
            "sector": "Financial Services",
            "tickers": stoxx_600_banks_tickers,
            "max_posiciones": 5,
        },
        {
            "etiqueta": "defensa",
            "sector": "Defensa",
            "tickers": defensa_europa_tickers,
            "max_posiciones": 5,
        },
        {
            "etiqueta": "energia",
            "sector": "Energia",
            "tickers": energia_europa_tickers,
            "max_posiciones": 5,
        },
    ]
 
    print(f"⚙️  Cartera: máx {objEstra.max_posiciones_abiertas} posiciones "
          f"({' + '.join(str(u['max_posiciones']) + ' ' + u['etiqueta'] for u in UNIVERSOS)}), "
          f"máx {objEstra.max_pct_por_posicion:.0%} del capital por posición.")
 
    # --- COMPROBACIÓN DE COHERENCIA ---
    # Si el nº de posiciones x el % por posición pasa del 100%, la cartera
    # intenta comprometer más capital del que hay. Es el fallo que aparece
    # justo al subir el nº de posiciones y olvidarse de bajar el %.
    _exposicion = objEstra.max_posiciones_abiertas * objEstra.max_pct_por_posicion
    if _exposicion > 1.0:
        print(f"⚠️ INCOHERENCIA: {objEstra.max_posiciones_abiertas} posiciones x "
              f"{objEstra.max_pct_por_posicion:.0%} = {_exposicion:.0%} del capital. "
              f"Baja max_pct_por_posicion a {1.0/objEstra.max_posiciones_abiertas:.0%} o menos.")
    _suma_topes = sum(u["max_posiciones"] for u in UNIVERSOS)
    if _suma_topes < objEstra.max_posiciones_abiertas:
        print(f"ℹ️ La suma de topes por universo ({_suma_topes}) es menor que el tope "
              f"global ({objEstra.max_posiciones_abiertas}): nunca llenarás la cartera.")
 
    #######################################################################
    #  1) VENTA PRIMERO — UNA SOLA VEZ para TODA la cartera
    #
    #  Las reglas de salida son POR TICKER (stop, ADX, tiempo), no por sector,
    #  así que se evalúan todas juntas. Si llamáramos a vender dentro de cada
    #  universo, las posiciones se evaluarían dos veces (y con la lista de
    #  candidatos equivocada para el criterio "salió del ranking").
    #
    #  Va antes de comprar para liberar plazas y capital: si la cartera está
    #  llena de posiciones que hoy toca cerrar, comprar primero las rechazaría
    #  todas por "portfolio lleno" y se perdería el rebalanceo de la corrida.
    #######################################################################
    print("\n" + "#" * 70)
    print("#  FASE 1 — REVISIÓN DE VENTAS (toda la cartera, todos los sectores)")
    print("#" * 70)
    objEstra.etiqueta_actual = "cartera"
    objEstra.vender_con_estrategia()
 
    #######################################################################
    #  2) COMPRAS — un universo detrás de otro
    #######################################################################
    print("\n" + "#" * 70)
    print("#  FASE 2 — BÚSQUEDA DE SEÑALES DE COMPRA POR UNIVERSO")
    print("#" * 70)
 
    resultados = {}
    for u in UNIVERSOS:
        try:
            resultados[u["etiqueta"]] = objEstra.procesar_universo(
                tickers=u["tickers"],
                sector=u["sector"],
                etiqueta=u["etiqueta"],
                max_posiciones_sector=u["max_posiciones"],
            )
        except Exception as e:
            # Un universo que falle NO debe impedir que se procese el siguiente
            print(f"❌ Error procesando universo '{u['etiqueta']}': {e}")
            try:
                send_message(f"❌ Error en universo {u['etiqueta']}: {e}")
            except Exception:
                pass
            resultados[u["etiqueta"]] = None
 
    #######################################################################
    #  3) RESUMEN FINAL DE CARTERA (desglosado por sector)
    #######################################################################
    objEstra.etiqueta_actual = "cartera"
    por_sector = {}
    for tk, p in objEstra.posiciones_abiertas.items():
        por_sector.setdefault(p.get("sector", "?"), []).append(tk)
 
    lineas_fin = [
        "",
        "📋 CARTERA AL CIERRE DE LA EJECUCIÓN",
        "───────────────────────────────────────────────",
        f"  Total: {len(objEstra.posiciones_abiertas)}/{objEstra.max_posiciones_abiertas} posiciones",
    ]
    for sec, lista in sorted(por_sector.items()):
        lineas_fin.append(f"  {sec}: {len(lista)} -> {', '.join(lista)}")
    if not por_sector:
        lineas_fin.append("  (sin posiciones abiertas)")
    lineas_fin.append("───────────────────────────────────────────────")
 
    # Aviso de concentración: 5 posiciones en 1 solo sector es mucho riesgo
    # correlacionado (en una crisis bancaria caen todos los bancos a la vez).
    if por_sector:
        sec_mayor, lista_mayor = max(por_sector.items(), key=lambda x: len(x[1]))
        if len(lista_mayor) >= 4:
            lineas_fin.append(f"  ⚠️ {len(lista_mayor)} de {len(objEstra.posiciones_abiertas)} posiciones "
                              f"están en '{sec_mayor}'. Riesgo correlacionado alto:")
            lineas_fin.append(f"     en una crisis del sector caerían todas a la vez.")
            lineas_fin.append("───────────────────────────────────────────────")
 
    # Resumen del histórico acumulado: sin backtest, estas operaciones reales
    # son la ÚNICA evidencia de si la estrategia funciona.
    try:
        objEstra.resumen_historial()
    except Exception as e:
        print(f"⚠️ No pude generar el resumen del histórico: {e}")
 
    texto_fin = "\n".join(lineas_fin)
    print(texto_fin)
    try:
        send_message(texto_fin)
    except Exception as e:
        print(f"⚠️ No pude enviar el resumen final a Telegram: {e}")
 
    if DEBUG__:
        print("Pulsa una tecla para finalizar ")
 
    print('✅✅ This is it................ QVM v2')
    logging.warning('Paso por STOCXX QVM, esto es una migita FIN ')
    sys.exit(32)
 
else:
    print(' libreria')
    print('version(l): ', versionVersion)