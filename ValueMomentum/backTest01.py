
# pip install backtesting yfinance ta pandas numpy

import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
import ta

# 1) Datos desde Yahoo
TICKER = "USB"
df = yf.download(TICKER, period="5y", interval="1d", auto_adjust=True, progress=False)
df.columns = df.columns.droplevel(1)
df = df[['Open','High','Low','Close','Volume']].dropna().copy()
if getattr(df.index, "tz", None) is not None:
    df.index = df.index.tz_localize(None)

# 2) Función externa que devuelve la ORDEN (buy/close/hold)
def mi_funcion_orden(ctx: dict) -> dict:
    vals = [ctx['sma20'], ctx['sma50'], ctx['adx'], ctx['macd'], ctx['macd_sig']]
    if any(pd.isna(vals)):
        return {"action": "hold"}

    tendencia_fuerte = ctx['adx'] > 25 and ctx['adx_pos'] > ctx['adx_neg']
    cruce_alcista = ctx['sma20'] > ctx['sma50']
    macd_ok = ctx['macd'] > ctx['macd_sig']

    if not ctx['in_position'] and tendencia_fuerte and cruce_alcista and macd_ok:
        return {"action": "buy", "size": 1, "sl": ctx['close'] * 0.97, "tp": ctx['close'] * 1.06}

    perder_fuerza = ctx['adx'] < 20
    cruce_bajista = ctx['sma20'] < ctx['sma50']
    macd_mal = ctx['macd'] < ctx['macd_sig']

    if ctx['in_position'] and (cruce_bajista or macd_mal or perder_fuerza):
        return {"action": "close"}

    return {"action": "hold"}

# 3) Estrategia que usa esa función
def _SMA(series: pd.Series, n: int) -> pd.Series:
    return pd.Series(series).rolling(n).mean()

class EstrategiaTecnica(Strategy):
    sma_fast = 20
    sma_slow = 50
    adx_win  = 14

    def init(self):
        h, l, c = self.data.High, self.data.Low, self.data.Close
        self.sma20 = self.I(_SMA, c, self.sma_fast)
        self.sma50 = self.I(_SMA, c, self.sma_slow)

        adx = ta.trend.ADXIndicator(pd.Series(h), pd.Series(l), pd.Series(c), window=self.adx_win)
        self.adx     = self.I(lambda: adx.adx())
        self.adx_pos = self.I(lambda: adx.adx_pos())
        self.adx_neg = self.I(lambda: adx.adx_neg())

        macd = ta.trend.MACD(pd.Series(c))
        self.macd     = self.I(lambda: macd.macd())
        self.macd_sig = self.I(lambda: macd.macd_signal())

    def next(self):
        ctx = dict(
            close=float(self.data.Close[-1]),
            sma20=float(self.sma20[-1]), sma50=float(self.sma50[-1]),
            adx=float(self.adx[-1]), adx_pos=float(self.adx_pos[-1]), adx_neg=float(self.adx_neg[-1]),
            macd=float(self.macd[-1]), macd_sig=float(self.macd_sig[-1]),
            in_position=bool(self.position)
        )
        orden = mi_funcion_orden(ctx)

        if orden["action"] == "buy" and not self.position:
            self.buy(size=orden.get("size", 1), sl=orden.get("sl"), tp=orden.get("tp"))
        elif orden["action"] == "close" and self.position:
            self.position.close()

# 4) Backtest — ojo: usar 'spread', no 'slippage'
bt = Backtest(
    df,
    EstrategiaTecnica,
    cash=10_000,
    commission=0.0005,  # 5 bps por lado
    spread=0.0005,      # 5 bps de horquilla/slippage fijo
    trade_on_close=False,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats['_trades'].head())
#bt.plot(open_browser=True)
bt.plot(filename="mi_backtest.html", open_browser=True, plot_drawdown=True)

