import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Get all available currency pairs
all_symbols = mt5.symbols_get()
currency_pairs = [symbol.name for symbol in all_symbols if symbol.name.endswith(("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"))]

volume = 0.20
TIMEFRAME = mt5.TIMEFRAME_M15
DEVIATION = 20

def market_order(symbol, volume, order_type, sl_price, tp_price, deviation=DEVIATION):
    try:
        price = mt5.symbol_info_tick(symbol).ask if order_type == 0 else mt5.symbol_info_tick(symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": deviation,
            "magic": 100,
            "comment": "Python order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        
        print(f"Date: {datetime.now()}")
        print(f"Time: {datetime.now().time()}")
        print(f"Currency Symbol: {symbol}")
        print(f"Order Type: {'Buy' if order_type == mt5.ORDER_TYPE_BUY else 'Sell'}")
        print(f"SL: {sl_price}")
        print(f"TP: {tp_price}")
        print(f"Volume: {volume}")
        print(f"Order Result: {result}")

        if result is not None:
            print(f"Analyzing {symbol} - Order result: {result}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Analyzing {symbol} - Order placed successfully. Order ticket: {result.order}")
            else:
                print(f"Analyzing {symbol} - Failed to place order. Return code: {result.retcode}")
        else:
            print(f"Analyzing {symbol} - Failed to place order. No result returned.")
        return result

    except Exception as e:
        print(f"Analyzing {symbol} - An error occurred while placing the order: {e}")
        return None

def close_order(ticket, deviation=DEVIATION):
    positions = mt5.positions_get()

    for pos in positions:
        if pos.ticket == ticket:
            tick = mt5.symbol_info_tick(pos.symbol)
            price_dict = {0: tick.ask, 1: tick.bid}
            type_dict = {0: 1, 1: 0}
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": deviation,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            order_result = mt5.order_send(request)
            print(order_result)
            return order_result

    return 'Ticket does not exist'

def get_historical_data(symbol, timeframe, count=1000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return pd.DataFrame()
    rates_df = pd.DataFrame(rates)
    rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')
    return rates_df

def calculate_indicators(data):
    data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['close'].diff(1).clip(lower=0).rolling(14).mean() /
                                     data['close'].diff(1).clip(upper=0).abs().rolling(14).mean())))
    data['ATR'] = data[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1
    ).rolling(14).mean()
    return data

def prepare_data(symbol, timeframe):
    data = get_historical_data(symbol, timeframe)
    if data.empty:
        return pd.DataFrame(), pd.Series()
    data = calculate_indicators(data)
    data = data.dropna()

    # Features and target
    features = data[['EMA20', 'EMA50', 'RSI', 'ATR']]
    target = np.where(data['close'].shift(-1) > data['close'], 1, 0)

    return features, pd.Series(target, index=features.index)

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy}")
    return model

def get_signal(model, data):
    features = data[['EMA20', 'EMA50', 'RSI', 'ATR']].iloc[-1:].values
    prediction = model.predict(features)
    return 'buy' if prediction == 1 else 'sell'

def run_strategy(model):
    for symbol in currency_pairs:
        features, target = prepare_data(symbol, TIMEFRAME)
        if features.empty:
            continue
        signal = get_signal(model, features)
        last_close = get_historical_data(symbol, TIMEFRAME).iloc[-1]['close']

        if signal == 'buy':
            sl_price = last_close - (1.5 * features.iloc[-1]['ATR'])
            tp_price = last_close + (3 * features.iloc[-1]['ATR'])
            market_order(symbol, volume, mt5.ORDER_TYPE_BUY, sl_price, tp_price)
        elif signal == 'sell':
            sl_price = last_close + (1.5 * features.iloc[-1]['ATR'])
            tp_price = last_close - (3 * features.iloc[-1]['ATR'])
            market_order(symbol, volume, mt5.ORDER_TYPE_SELL, sl_price, tp_price)
        else:
            print(f"Analyzing {symbol} - Hold position")
        time.sleep(300)  # Wait for 5 minutes before placing the next trade

features, target = prepare_data(currency_pairs[0], TIMEFRAME)  # Prepare data using the first currency pair
model = train_model(features, target)  # Train the model

while True:
    run_strategy(model)
    print(f"Strategy executed at {datetime.now()}")
    time.sleep(120)  # Sleep for 2 minutes before the next iteration
