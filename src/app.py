import asyncio
import json
import requests
import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime, date, timedelta
import logging
import sqlite3
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from threading import Thread
import sys

# Configure Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading.log')
logging.getLogger().addHandler(logging.StreamHandler())

# Configuration (same as original)
TELEGRAM_BOT_TOKEN = "7734973822:AAF4bD4SxEqJ4CKaJzakF4gRD4vIZp8bS_8"
TELEGRAM_CHAT_ID = "@tradincapital"
TELEGRAM_MESSAGE_DELAY = 1
CAPITAL = 100000
RISK_PER_TRADE = 0.02
LOT_SIZE = 25
MAX_TRADES_PER_DAY = 3
TRADING_HOURS = {"start": "09:30", "end": "22:15"}
DAILY_LOSS_LIMIT = -0.05 * CAPITAL
USE_NEXT_EXPIRY_ON_EXPIRY_DAY = True
BASE_URL = "https://api.upstox.com/v2"
TOKEN_FILE = "access_token.txt"
DATE_FILE = "last_token_date.txt"
CONFIG_FILE = "upstox_config.json"
MAX_RETRIES = 3
RETRY_DELAY = 5
SQLITE_DB = "trade_ledger.db"

# Real-time data storage
rest_data = {
    "nifty_spot": 23850,
    "vix": 15.0,
    "prices_1min": [23850] * 50,
    "prices_5min": [23850] * 20,
    "option_chain": None,
    "last_update": None
}

# Global variables
ACCESS_TOKEN = None
trading_active = False
data_task = None

# Load Upstox config (same as original)
def load_upstox_config():
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading Upstox config: {e}")
    config = {
        "API_KEY": "your_api_key",
        "API_SECRET": "your_api_secret",
        "REDIRECT_URI": "your_redirect_uri"
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        logging.info("Upstox config saved to file")
    except Exception as e:
        logging.error(f"Error saving Upstox config: {e}")
    return config

upstox_config = load_upstox_config()
API_KEY = upstox_config["API_KEY"]
API_SECRET = upstox_config["API_SECRET"]
REDIRECT_URI = upstox_config["REDIRECT_URI"]

# Telegram and token functions (same as original)
def validate_telegram_config():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if not response.json().get("ok"):
            raise Exception("Invalid Bot Token")
        logging.info("Telegram Bot Token validated")
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "Validating Telegram configuration...", "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Telegram Chat ID validated")
        return True
    except Exception as e:
        logging.error(f"Telegram validation failed: {e}")
        send_telegram_message(f"Critical Error: Telegram validation failed: {e}")
        return False

def send_telegram_message(message):
    if len(message) > 4096:
        message = message[:4093] + "..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Telegram message sent")
        time.sleep(TELEGRAM_MESSAGE_DELAY)
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def load_token():
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                return f.read().strip()
        return None
    except Exception as e:
        logging.error(f"Error loading token: {e}")
        return None

def save_token(token):
    try:
        with open(TOKEN_FILE, 'w') as f:
            f.write(token)
        logging.info("Access token saved")
    except Exception as e:
        logging.error(f"Error saving token: {e}")

def load_last_token_date():
    try:
        if os.path.exists(DATE_FILE):
            with open(DATE_FILE, 'r') as f:
                return f.read().strip()
        return None
    except Exception as e:
        logging.error(f"Error loading last token date: {e}")
        return None

def save_last_token_date():
    try:
        with open(DATE_FILE, 'w') as f:
            f.write(str(date.today()))
        logging.info("Last token date saved")
    except Exception as e:
        logging.error(f"Error saving last token date: {e}")

def validate_access_token(token):
    url = f"{BASE_URL}/user/profile"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logging.info("Access token validated")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Token validation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2**attempt)
    return False

def generate_access_token(code):
    url = f"{BASE_URL}/login/authorization/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "code": code,
        "client_id": API_KEY,
        "client_secret": API_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        access_token = data.get("access_token")
        if not access_token:
            raise Exception("No access token in response")
        return access_token
    except Exception as e:
        logging.error(f"Failed to generate access token: {e}")
        send_telegram_message(f"Critical Error: Failed to generate access token: {e}")
        return None

def initialize_api():
    global ACCESS_TOKEN
    current_date = str(date.today())
    last_token_date = load_last_token_date()
    stored_token = load_token()

    if last_token_date == current_date and stored_token and validate_access_token(stored_token):
        ACCESS_TOKEN = stored_token
        logging.info("Using stored access token")
        return True
    elif stored_token and validate_access_token(stored_token):
        ACCESS_TOKEN = stored_token
        save_last_token_date()
        logging.info("Using stored access token")
        return True
    return False

# Other functions (same as original)
def get_next_wednesday():
    today = date.today()
    days_ahead = (2 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

def is_expiry_day(expiries, today):
    today_str = today.strftime("%Y-%m-%d")
    return today_str in expiries

def get_expiry_date():
    url = f"{BASE_URL}/option/contract"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("data"):
                expiries = sorted(set(item.get("expiry_date") for item in data["data"] if item.get("segment") == "NFO-OPT"))
                if not expiries:
                    logging.warning("No weekly expiries found, using fallback")
                    return get_next_wednesday()
                today = date.today()
                if USE_NEXT_EXPIRY_ON_EXPIRY_DAY and is_expiry_day(expiries, today):
                    for expiry in expiries:
                        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                        if expiry_date > today:
                            logging.info(f"Expiry day detected, using next expiry: {expiry}")
                            return expiry
                    return get_next_wednesday()
                expiry = expiries[0]
                logging.info(f"Fetched expiry: {expiry}")
                return expiry
            logging.warning("No valid data in response, using fallback")
            return get_next_wednesday()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching expiry (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2**attempt)
    logging.error("Failed to fetch expiry, using fallback")
    return get_next_wednesday()

async def fetch_rest_market_data(expiry_date):
    global rest_data
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    
    expiry_clean = expiry_date.replace('-', '')
    option_keys = [f"NSE_FO|NIFTY{expiry_clean}C{strike}" for strike in range(23000, 25000, 50)] + \
                  [f"NSE_FO|NIFTY{expiry_clean}P{strike}" for strike in range(23000, 25000, 50)]
    
    while trading_active:
        try:
            ltp_url = f"{BASE_URL}/market-quote/ltp"
            ltp_params = {"instrument_key": "NSE_INDEX|Nifty 50,NSE_INDEX|India VIX"}
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(ltp_url, headers=headers, params=ltp_params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("status") == "success" and data.get("data"):
                        rest_data["nifty_spot"] = data["data"].get("NSE_INDEX|Nifty 50", {}).get("last_price", rest_data["nifty_spot"])
                        rest_data["vix"] = data["data"].get("NSE_INDEX|India VIX", {}).get("last_price", rest_data["vix"])
                        rest_data["prices_1min"].append(rest_data["nifty_spot"])
                        rest_data["prices_1min"] = rest_data["prices_1min"][-50:]
                        if len(rest_data["prices_1min"]) % 5 == 0:
                            rest_data["prices_5min"].append(rest_data["nifty_spot"])
                            rest_data["prices_5min"] = rest_data["prices_5min"][-20:]
                    break
                except requests.exceptions.RequestException as e:
                    logging.error(f"LTP fetch failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * 2**attempt)
            
            chain_url = f"{BASE_URL}/option/contract"
            chain_params = {"instrument_key": "NSE_INDEX|Nifty 50"}
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get(chain_url, headers=headers, params=chain_params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("status") == "success" and data.get("data"):
                        rest_data["option_chain"] = []
                        for item in data["data"]:
                            if item.get("segment") == "NFO-OPT" and item.get("expiry_date") == expiry_date:
                                rest_data["option_chain"].append({
                                    "instrument_key": item.get("instrument_key"),
                                    "strike_price": float(item.get("strike_price")),
                                    "call_ltp": item.get("last_price") if "CE" in item.get("instrument_key") else None,
                                    "put_ltp": item.get("last_price") if "PE" in item.get("instrument_key") else None,
                                    "call_volume": item.get("volume", 1) if "CE" in item.get("instrument_key") else None,
                                    "put_volume": item.get("volume", 1) if "PE" in item.get("instrument_key") else None,
                                    "call_oi": item.get("open_interest", 10000) if "CE" in item.get("instrument_key") else None,
                                    "put_oi": item.get("open_interest", 10000) if "PE" in item.get("instrument_key") else None,
                                    "call_iv": item.get("implied_volatility", 25) if "CE" in item.get("instrument_key") else None,
                                    "put_iv": item.get("implied_volatility", 25) if "PE" in item.get("instrument_key") else None,
                                    "call_bid_price": item.get("bid_price", 95) if "CE" in item.get("instrument_key") else None,
                                    "put_bid_price": item.get("bid_price", 95) if "PE" in item.get("instrument_key") else None,
                                    "call_ask_price": item.get("ask_price", 105) if "CE" in item.get("instrument_key") else None,
                                    "put_ask_price": item.get("ask_price", 105) if "PE" in item.get("instrument_key") else None,
                                    "call_delta": item.get("delta", 0.5) if "CE" in item.get("instrument_key") else None,
                                    "put_delta": item.get("delta", -0.5) if "PE" in item.get("instrument_key") else None,
                                    "call_theta": item.get("theta", -1) if "CE" in item.get("instrument_key") else None,
                                    "put_theta": item.get("theta", -1) if "PE" in item.get("instrument_key") else None
                                })
                    break
                except requests.exceptions.RequestException as e:
                    logging.error(f"Option chain fetch failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * 2**attempt)
            
            rest_data["last_update"] = datetime.now()
            logging.debug(f"REST data updated: Spot={rest_data['nifty_spot']}, VIX={rest_data['vix']}")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"REST data fetch error: {e}")
            send_telegram_message(f"Critical Error: REST data fetch error: {e}")
            await asyncio.sleep(10)

def calculate_indicators(prices_1min, prices_5min):
    try:
        prices_1min = np.array(prices_1min, dtype=float)
        prices_5min = np.array(prices_5min, dtype=float)
        if len(prices_1min) < 50 or np.any(np.isnan(prices_1min)) or np.any(prices_1min <= 0):
            return None
        rsi_1min = talib.RSI(prices_1min, timeperiod=10)[-1]
        macd_1min, signal_1min, _ = talib.MACD(prices_1min, fastperiod=8, slowperiod=21, signalperiod=5)
        macd_1min = macd_1min[-1]
        signal_1min = signal_1min[-1]
        upper_1min, middle_1min, lower_1min = talib.BBANDS(prices_1min, timeperiod=20, nbdevup=2, nbdevdn=2)
        obv_1min = talib.OBV(prices_1min, np.ones_like(prices_1min))[-1]
        atr_1min = talib.ATR(prices_1min, prices_1min, prices_1min, timeperiod=14)[-1]
        if len(prices_5min) >= 20:
            rsi_5min = talib.RSI(prices_5min, timeperiod=10)[-1]
            macd_5min, signal_5min, _ = talib.MACD(prices_5min, fastperiod=8, slowperiod=21, signalperiod=5)
            macd_5min = macd_5min[-1]
            signal_5min = signal_5min[-1]
        else:
            rsi_5min = rsi_1min
            macd_5min = macd_1min
            signal_5min = signal_1min
        indicators = {
            "rsi_1min": rsi_1min if not np.isnan(rsi_1min) and 0 < rsi_1min < 100 else 50,
            "macd_1min": macd_1min if not np.isnan(macd_1min) else 0,
            "signal_1min": signal_1min if not np.isnan(signal_1min) else 0,
            "bb_upper_1min": upper_1min[-1] if not np.isnan(upper_1min[-1]) else prices_1min[-1] + 100,
            "bb_lower_1min": lower_1min[-1] if not np.isnan(lower_1min[-1]) else prices_1min[-1] - 100,
            "obv_1min": obv_1min if not np.isnan(obv_1min) else 0,
            "atr_1min": atr_1min if not np.isnan(atr_1min) and atr_1min > 0 else 50,
            "rsi_5min": rsi_5min,
            "macd_5min": macd_5min,
            "signal_5min": signal_5min
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return None

def calculate_fibonacci_levels(prices, vix):
    try:
        prices = np.array(prices, dtype=float)
        lookback = 30 if vix > 20 else 50
        high_price = np.max(prices[-lookback:])
        low_price = np.min(prices[-lookback:])
        price_range = high_price - low_price
        fib_levels = {
            "23.6%": high_price - 0.236 * price_range,
            "38.2%": high_price - 0.382 * price_range,
            "50.0%": high_price - 0.500 * price_range,
            "61.8%": high_price - 0.618 * price_range,
            "78.6%": high_price - 0.786 * price_range,
            "high": high_price,
            "low": low_price
        }
        return fib_levels
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {e}")
        return None

def check_market_context(vix):
    if vix > 25:
        logging.warning("High volatility (VIX > 25). Skipping trades.")
        return False
    return True

def analyze_sentiment(df, spot_price):
    try:
        df = pd.DataFrame(df)
        if df.empty:
            return None
        atm_strike = min(df['strike_price'], key=lambda x: abs(x - spot_price))
        atm_data = df[df['strike_price'] == atm_strike]
        if atm_data.empty:
            return None
        nearby_strikes = df[abs(df['strike_price'] - atm_strike) <= 200]
        pcr = nearby_strikes['put_oi'].sum() / nearby_strikes['call_oi'].sum() if nearby_strikes['call_oi'].sum() > 0 else 1.0
        support = df.loc[df['put_oi'].idxmax(), 'strike_price']
        resistance = df.loc[df['call_oi'].idxmax(), 'strike_price']
        iv = (atm_data['call_iv'].iloc[0] + atm_data['put_iv'].iloc[0]) / 2
        fib_levels = calculate_fibonacci_levels(rest_data["prices_1min"], rest_data["vix"])
        if fib_levels:
            fib_support = fib_levels["61.8%"]
            fib_resistance = fib_levels["38.2%"]
            if abs(spot_price - fib_support) < abs(spot_price - support):
                support = fib_support
            if abs(spot_price - fib_resistance) < abs(spot_price - resistance):
                resistance = fib_resistance
        direction_score = 0
        if pcr < 0.8:
            direction_score += 1
        elif pcr > 1.2:
            direction_score -= 1
        if spot_price > resistance:
            direction_score += 1
        elif spot_price < support:
            direction_score -= 1
        if iv > 30:
            direction_score += 0.5 * (1 if spot_price > atm_strike else -1)
        if fib_levels:
            for level in ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]:
                if abs(spot_price - fib_levels[level]) < 50:
                    direction_score += 0.5 if spot_price > fib_levels["50.0%"] else -0.5
                    break
        direction = "Bullish" if direction_score > 0 else "Bearish" if direction_score < 0 else "Neutral"
        sentiment = {
            "pcr": pcr,
            "support": support,
            "resistance": resistance,
            "iv": iv,
            "atm_strike": atm_strike,
            "call_instrument_key": atm_data['instrument_key'].iloc[0] if "CE" in atm_data['instrument_key'].iloc[0] else atm_data['instrument_key'].iloc[1],
            "put_instrument_key": atm_data['instrument_key'].iloc[1] if "PE" in atm_data['instrument_key'].iloc[1] else atm_data['instrument_key'].iloc[0],
            "call_bid_price": atm_data['call_bid_price'].iloc[0],
            "put_bid_price": atm_data['put_bid_price'].iloc[0],
            "call_ask_price": atm_data['call_ask_price'].iloc[0],
            "put_ask_price": atm_data['put_ask_price'].iloc[0],
            "call_delta": atm_data['call_delta'].iloc[0],
            "put_delta": atm_data['put_delta'].iloc[0],
            "oi_total": atm_data['call_oi'].iloc[0] + atm_data['put_oi'].iloc[0],
            "direction": direction,
            "direction_score": direction_score,
            "fib_levels": fib_levels
        }
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

def generate_trade_signal(sentiment, indicators, spot_price, prices_1min):
    if not sentiment or not check_market_context(rest_data["vix"]) or not indicators:
        return None
    atr = indicators['atr_1min']
    breakout_threshold = atr * (1.5 if rest_data["vix"] > 20 else 1.0)
    recent_move = spot_price - prices_1min[-5]
    is_breakout = abs(recent_move) >= breakout_threshold
    fib_levels = sentiment.get("fib_levels")
    confirmation_score = 0
    conditions_met = []
    fib_proximity = None
    if fib_levels:
        for level in ["23.6%", "38.2%", "50.0%", "61.8%", "78.6%"]:
            if abs(spot_price - fib_levels[level]) < atr / 2:
                fib_proximity = level
                confirmation_score += 1
                conditions_met.append(f"Fibonacci proximity ({level}: {fib_levels[level]:.2f})")
                break
    is_consolidation = indicators['bb_upper_1min'] - indicators['bb_lower_1min'] < 2.0 * atr
    if is_consolidation and 0.8 <= sentiment['pcr'] <= 2.0 and 15 <= sentiment['iv'] <= 25:
        straddle_score = 0
        if is_consolidation:
            straddle_score += 1
            conditions_met.append("Consolidation (BB width < 2.0 * ATR)")
        if 0.8 <= sentiment['pcr'] <= 2.0:
            straddle_score += 1
            conditions_met.append(f"PCR balanced ({sentiment['pcr']:.2f})")
        if 15 <= sentiment['iv'] <= 25:
            straddle_score += 1
            conditions_met.append(f"IV moderate ({sentiment['iv']:.2f})")
        confirmation_score += straddle_score
        if confirmation_score >= 4 and straddle_score >= 2:
            trade = {
                "type": "Straddle",
                "strike": sentiment['atm_strike'],
                "ce_entry": None,
                "pe_entry": None,
                "ce_instrument_key": sentiment['call_instrument_key'],
                "pe_instrument_key": sentiment['put_instrument_key'],
                "sl": 0.3,
                "target": 0.5,
                "time": datetime.now().strftime("%H:%M:%S"),
                "direction": sentiment['direction'],
                "confirmation_score": confirmation_score,
                "fib_proximity": fib_proximity
            }
            return trade
    is_bullish = (recent_move > 0 and spot_price > indicators['bb_upper_1min'] and
                  indicators['macd_1min'] > indicators['signal_1min'] and indicators['rsi_1min'] < 70 and
                  indicators['macd_5min'] > indicators['signal_5min'] and sentiment['call_delta'] > 0.5)
    bullish_score = 0
    if is_bullish:
        if recent_move > 0:
            bullish_score += 1
            conditions_met.append(f"Recent move up ({recent_move:.2f})")
        if spot_price > indicators['bb_upper_1min']:
            bullish_score += 1
            conditions_met.append(f"Above BB Upper ({spot_price:.2f} > {indicators['bb_upper_1min']:.2f})")
        if indicators['macd_1min'] > indicators['signal_1min']:
            bullish_score += 1
            conditions_met.append(f"MACD 1min bullish ({indicators['macd_1min']:.2f} > {indicators['signal_1min']:.2f})")
        if indicators['rsi_1min'] < 70:
            bullish_score += 1
            conditions_met.append(f"RSI not overbought ({indicators['rsi_1min']:.2f})")
        if indicators['macd_5min'] > indicators['signal_5min']:
            bullish_score += 1
            conditions_met.append(f"MACD 5min bullish ({indicators['macd_5min']:.2f} > {indicators['signal_5min']:.2f})")
        if sentiment['call_delta'] > 0.5:
            bullish_score += 1
            conditions_met.append(f"High call delta ({sentiment['call_delta']:.2f})")
        confirmation_score += bullish_score
        if confirmation_score >= 4 and bullish_score >= 4 and fib_proximity:
            trade = {
                "type": "CE",
                "strike": sentiment['atm_strike'],
                "ce_entry": None,
                "ce_instrument_key": sentiment['call_instrument_key'],
                "sl": 0.3,
                "target": 0.5,
                "time": datetime.now().strftime("%H:%M:%S"),
                "direction": "Bullish",
                "confirmation_score": confirmation_score,
                "fib_proximity": fib_proximity
            }
            if fib_levels:
                trade['sl'] = max(0.2, trade['sl'] * (0.9 if spot_price > fib_levels["61.8%"] else 1))
                trade['target'] = min(0.7, trade['target'] * (1.1 if spot_price > fib_levels["38.2%"] else 1))
            return trade
    is_bearish = (recent_move < 0 and spot_price < indicators['bb_lower_1min'] and
                  indicators['macd_1min'] < indicators['signal_1min'] and indicators['rsi_1min'] > 30 and
                  indicators['macd_5min'] < indicators['signal_5min'] and abs(sentiment['put_delta']) > 0.5)
    bearish_score = 0
    if is_bearish:
        if recent_move < 0:
            bearish_score += 1
            conditions_met.append(f"Recent move down ({recent_move:.2f})")
        if spot_price < indicators['bb_lower_1min']:
            bearish_score += 1
            conditions_met.append(f"Below BB Lower ({spot_price:.2f} < {indicators['bb_lower_1min']:.2f})")
        if indicators['macd_1min'] < indicators['signal_1min']:
            bearish_score += 1
            conditions_met.append(f"MACD 1min bearish ({indicators['macd_1min']:.2f} < {indicators['signal_1min']:.2f})")
        if indicators['rsi_1min'] > 30:
            bearish_score += 1
            conditions_met.append(f"RSI not oversold ({indicators['rsi_1min']:.2f})")
        if indicators['macd_5min'] < indicators['signal_5min']:
            bearish_score += 1
            conditions_met.append(f"MACD 5min bearish ({indicators['macd_5min']:.2f} > {indicators['signal_5min']:.2f})")
        if abs(sentiment['put_delta']) > 0.5:
            bearish_score += 1
            conditions_met.append(f"High put delta ({sentiment['put_delta']:.2f})")
        confirmation_score += bearish_score
        if confirmation_score >= 4 and bearish_score >= 4 and fib_proximity:
            trade = {
                "type": "PE",
                "strike": sentiment['atm_strike'],
                "pe_entry": None,
                "pe_instrument_key": sentiment['put_instrument_key'],
                "sl": 0.3,
                "target": 0.5,
                "time": datetime.now().strftime("%H:%M:%S"),
                "direction": "Bearish",
                "confirmation_score": confirmation_score,
                "fib_proximity": fib_proximity
            }
            if fib_levels:
                trade['sl'] = max(0.2, trade['sl'] * (0.9 if spot_price < fib_levels["38.2%"] else 1))
                trade['target'] = min(0.7, trade['target'] * (1.1 if spot_price < fib_levels["61.8%"] else 1))
            return trade
    return None

def execute_trade(trade, df, atr, vix):
    try:
        df = pd.DataFrame(df)
        atm_data = df[df['strike_price'] == trade['strike']]
        if atm_data.empty:
            return None
        ce_premium = atm_data['call_ask_price'].iloc[0]
        pe_premium = atm_data['put_ask_price'].iloc[0]
        ce_delta = atm_data['call_delta'].iloc[0]
        pe_delta = atm_data['put_delta'].iloc[0]
        if trade['type'] in ["PE", "Straddle"] and pe_premium < 20:
            logging.warning(f"Low PE premium: ₹{pe_premium:.2f}, skipping trade")
            return None
        if trade['type'] in ["CE", "Straddle"] and ce_premium < 20:
            logging.warning(f"Low CE premium: ₹{ce_premium:.2f}, skipping trade")
            return None
        iv_factor = atm_data['call_iv'].iloc[0] / 25
        delta_factor = max(abs(ce_delta), abs(pe_delta)) * 0.5
        sl_factor = 0.1 + (atr / 1000) * iv_factor * delta_factor * (0.8 if is_expiry_day([get_expiry_date()], date.today()) else 1)
        target_factor = 0.3 + (atr / 500) * iv_factor * (0.8 if is_expiry_day([get_expiry_date()], date.today()) else 1)
        if trade['type'] == "Straddle":
            trade['ce_entry'] = ce_premium
            trade['pe_entry'] = pe_premium
            total_premium = ce_premium + pe_premium
            trade['sl_price'] = total_premium * (1 - sl_factor)
            trade['target_price'] = total_premium * (1 + target_factor)
        elif trade['type'] == "CE":
            trade['ce_entry'] = ce_premium
            total_premium = ce_premium
            trade['sl_price'] = ce_premium * (1 - sl_factor)
            trade['target_price'] = ce_premium * (1 + target_factor)
        elif trade['type'] == "PE":
            trade['pe_entry'] = pe_premium
            total_premium = pe_premium
            trade['sl_price'] = pe_premium * (1 - sl_factor)
            trade['target_price'] = pe_premium * (1 + target_factor)
        max_loss = CAPITAL * RISK_PER_TRADE
        monetary_sl = (total_premium - trade['sl_price']) * LOT_SIZE
        lots = min(10, max_loss / monetary_sl * (0.8 if vix > 20 else 1)) if monetary_sl > 0 else 1
        trade['lots'] = int(lots)
        if trade['lots'] == 0:
            trade['lots'] = 1
        return trade
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

def save_trade_to_db(trade, indicators):
    try:
        conn = sqlite3.connect(SQLITE_DB)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT, type TEXT, strike REAL, ce_entry REAL, pe_entry REAL,
                ce_instrument TEXT, pe_instrument TEXT, sl_price REAL, target_price REAL,
                lots INTEGER, rsi REAL, macd REAL, atr REAL, vix REAL, direction TEXT,
                confirmation_score INTEGER, fib_proximity TEXT, status TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO trades (time, type, strike, ce_entry, pe_entry, ce_instrument, pe_instrument,
            sl_price, target_price, lots, rsi, macd, atr, vix, direction, confirmation_score, fib_proximity, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['time'], trade['type'], trade['strike'], trade.get('ce_entry'), trade.get('pe_entry'),
            trade.get('ce_instrument_key'), trade.get('pe_instrument_key'), trade['sl_price'],
            trade['target_price'], trade['lots'], indicators['rsi_1min'], indicators['macd_1min'],
            indicators['atr_1min'], rest_data['vix'], trade['direction'], trade['confirmation_score'],
            trade['fib_proximity'], 'OPEN'
        ))
        conn.commit()
        conn.close()
        logging.info("Trade saved to database")
    except Exception as e:
        logging.error(f"Error saving trade to database: {e}")

def generate_performance_dashboard():
    try:
        conn = sqlite3.connect(SQLITE_DB)
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        trades = df.to_dict('records')
        chart_data = {
            "type": "line",
            "data": {
                "labels": [t['time'] for t in trades],
                "datasets": [{
                    "label": "P&L",
                    "data": [(t['target_price'] - (t['ce_entry'] or t['pe_entry'] or 0)) * t['lots'] * LOT_SIZE for t in trades],
                    "borderColor": "#4CAF50",
                    "backgroundColor": "rgba(76, 175, 80, 0.2)",
                    "fill": True
                }]
            },
            "options": {
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": "P&L (₹)"}},
                    "x": {"title": {"display": True, "text": "Trade Time"}}
                }
            }
        }
        return chart_data
    except Exception as e:
        logging.error(f"Error generating performance dashboard: {e}")
        return None

async def trading_loop():
    global trading_active, data_task
    if not validate_telegram_config():
        send_telegram_message("Failed to validate Telegram configuration")
        return
    
    if not initialize_api():
        auth_url = f"{BASE_URL}/login/authorization/dialog?client_id={API_KEY}&redirect_uri={REDIRECT_URI}"
        send_telegram_message(f"Visit this URL to authorize: {auth_url}")
        return

    trades_today = 0
    daily_pnl = 0
    try:
        expiry_date = get_expiry_date()
    except Exception as e:
        logging.error(f"Failed to fetch expiry date: {e}")
        expiry_date = get_next_wednesday()
        send_telegram_message(f"Warning: Using fallback expiry date {expiry_date} due to error: {e}")
    
    logging.info(f"Starting trading for Nifty 50 (Expiry: {expiry_date})")
    data_task = asyncio.create_task(fetch_rest_market_data(expiry_date))
    try:
        while trading_active and trades_today < MAX_TRADES_PER_DAY:
            current_time = datetime.now().strftime("%H:%M")
            if not (TRADING_HOURS['start'] <= current_time <= TRADING_HOURS['end']):
                logging.info(f"Outside trading hours: {current_time}")
                await asyncio.sleep(60)
                continue
            if daily_pnl <= DAILY_LOSS_LIMIT:
                logging.warning("Daily loss limit reached, stopping trading")
                send_telegram_message("Trading stopped: Daily loss limit reached")
                break
            if rest_data["option_chain"] is None or (datetime.now() - rest_data["last_update"]).seconds > 300:
                logging.warning("No recent REST data, retrying")
                if data_task:
                    data_task.cancel()
                data_task = asyncio.create_task(fetch_rest_market_data(expiry_date))
                await asyncio.sleep(60)
                continue
            indicators = calculate_indicators(rest_data["prices_1min"], rest_data["prices_5min"])
            if indicators is None:
                logging.warning("Failed to calculate indicators, retrying")
                await asyncio.sleep(60)
                continue
            sentiment = analyze_sentiment(rest_data["option_chain"], rest_data["nifty_spot"])
            if sentiment is None:
                logging.warning("Failed to analyze sentiment, retrying")
                await asyncio.sleep(60)
                continue
            trade = generate_trade_signal(sentiment, indicators, rest_data["nifty_spot"], rest_data["prices_1min"])
            if trade:
                trade = execute_trade(trade, rest_data["option_chain"], indicators['atr_1min'], rest_data["vix"])
                if trade:
                    trades_today += 1
                    save_trade_to_db(trade, indicators)
                    message = f"<b>Confirmed Trade</b>\n" \
                             f"<b>Time:</b> {trade['time']}\n" \
                             f"<b>Type:</b> {trade['type']}\n" \
                             f"<b>Strike:</b> {trade['strike']}\n" \
                             f"<b>Expiry:</b> {expiry_date}\n" \
                             f"<b>Direction:</b> {trade['direction']}\n" \
                             f"<b>Score:</b> {trade['confirmation_score']}\n" \
                             f"<b>Fibonacci:</b> {trade['fib_proximity']}\n"
                    if trade['type'] == "Straddle":
                        message += f"<b>CE Entry:</b> ₹{trade['ce_entry']:.2f}\n" \
                                  f"<b>PE Entry:</b> ₹{trade['pe_entry']:.2f}\n" \
                                  f"<b>SL:</b> ₹{trade['sl_price']:.2f}, <b>Target:</b> ₹{trade['target_price']:.2f}\n"
                    else:
                        entry = trade['ce_entry'] if trade['type'] == "CE" else trade['pe_entry']
                        instrument = trade['ce_instrument_key'] if trade['type'] == "CE" else trade['pe_instrument_key']
                        message += f"<b>Entry:</b> ₹{entry:.2f}\n" \
                                  f"<b>SL:</b> ₹{trade['sl_price']:.2f}, <b>Target:</b> ₹{trade['target_price']:.2f}\n"
                    message += f"<b>Lots:</b> {trade['lots']}\n" \
                              f"<b>Indicators:</b> RSI={indicators['rsi_1min']:.2f}, MACD={indicators['macd_1min']:.2f}, " \
                              f"ATR={indicators['atr_1min']:.2f}, VIX={rest_data['vix']:.2f}"
                    send_telegram_message(message)
            await asyncio.sleep(60)
    except Exception as e:
        logging.error(f"Trading loop error: {e}")
        send_telegram_message(f"Critical Error: Trading stopped due to error: {e}")
    finally:
        if data_task:
            data_task.cancel()
        trading_active = False
        logging.info("Trading session ended")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html', trading_active=trading_active)

@app.route('/start_trading', methods=['POST'])
def start_trading():
    global trading_active, data_task
    if not trading_active:
        trading_active = True
        Thread(target=lambda: asyncio.run(trading_loop())).start()
        return jsonify({"status": "success", "message": "Trading started"})
    return jsonify({"status": "error", "message": "Trading already active"})

@app.route('/stop_trading', methods=['POST'])
def stop_trading():
    global trading_active, data_task
    if trading_active:
        trading_active = False
        if data_task:
            data_task.cancel()
        return jsonify({"status": "success", "message": "Trading stopped"})
    return jsonify({"status": "error", "message": "Trading not active"})

@app.route('/market_data')
def market_data():
    indicators = calculate_indicators(rest_data["prices_1min"], rest_data["prices_5min"])
    sentiment = analyze_sentiment(rest_data["option_chain"], rest_data["nifty_spot"]) if rest_data["option_chain"] else None
    return jsonify({
        "nifty_spot": rest_data["nifty_spot"],
        "vix": rest_data["vix"],
        "last_update": str(rest_data["last_update"]),
        "indicators": indicators,
        "sentiment": sentiment
    })

@app.route('/dashboard')
def dashboard():
    chart_data = generate_performance_dashboard()
    return render_template('dashboard.html', chart_data=json.dumps(chart_data))

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    global ACCESS_TOKEN
    if request.method == 'POST':
        code = request.form.get('code')
        if code:
            token = generate_access_token(code)
            if token and validate_access_token(token):
                ACCESS_TOKEN = token
                save_token(token)
                save_last_token_date()
                return jsonify({"status": "success", "message": "Authentication successful"})
            return jsonify({"status": "error", "message": "Authentication failed"})
    auth_url = f"{BASE_URL}/login/authorization/dialog?client_id={API_KEY}&redirect_uri={REDIRECT_URI}"
    return render_template('auth.html', auth_url=auth_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)