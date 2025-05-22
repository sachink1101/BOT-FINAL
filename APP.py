import requests
import pandas as pd
import numpy as np
import talib
import time
from datetime import datetime, date, timedelta
import logging
import urllib.parse
import csv
import os
import json
from pathlib import Path
import sys
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='trading.log')
logging.getLogger().addHandler(logging.StreamHandler())

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "7734973822:AAF4bD4SxEqJ4CKaJzakF4gRD4vIZp8bS_8"  # Bot token for @orboptionbot
TELEGRAM_CHAT_ID = "@tradincapital"  # Public channel username
TELEGRAM_MESSAGE_DELAY = 1  # Seconds between messages to avoid rate limits

# New configuration to toggle next expiry on expiry day
USE_NEXT_EXPIRY_ON_EXPIRY_DAY = True  # Set to True to use next expiry on expiry day

def validate_telegram_config():
    """Validate Telegram Bot Token and Chat ID."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if not response.json().get("ok"):
            raise Exception("Invalid Bot Token")
        logging.info("Telegram Bot Token validated")
    except Exception as e:
        logging.error(f"Telegram Bot Token validation failed: {e}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": "Validating Telegram configuration...", "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Telegram Chat ID validated")
        return True
    except Exception as e:
        logging.error(f"Telegram Chat ID validation failed: {e}. Please provide a valid numeric Chat ID.")
        return False

def send_telegram_message(message):
    """Send a message to the specified Telegram chat with rate limiting."""
    if len(message) > 4096:
        logging.warning("Telegram message exceeds 4096 characters, truncating")
        message = message[:4093] + "..."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Trade signal sent to Telegram")
        time.sleep(TELEGRAM_MESSAGE_DELAY)  # Rate limiting
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

# Load Upstox API credentials from config file or prompt
CONFIG_FILE = "upstox_config.json"
def load_upstox_config():
    """Load Upstox API credentials from a config file or prompt user."""
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading Upstox config: {e}")
    
    # Prompt for credentials (only in interactive mode)
    if sys.stdin.isatty():
        config = {
            "API_KEY": input("Enter Upstox API Key: ").strip(),
            "API_SECRET": input("Enter Upstox API Secret: ").strip(),
            "REDIRECT_URI": input("Enter Upstox Redirect URI: ").strip()
        }
    else:
        raise Exception("Non-interactive mode: Please create upstox_config.json with API_KEY, API_SECRET, and REDIRECT_URI")
    
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

# Configuration
CAPITAL = 100000
RISK_PER_TRADE = 0.02
LOT_SIZE = 25  # Current Nifty 50 lot size
MAX_TRADES_PER_DAY = 3
TRADING_HOURS = {"start": "00:00", "end": "23:59"}  # Relaxed for testing
ACCESS_TOKEN = "your_initial_access_token"  # Initial placeholder
BASE_URL = "https://api.upstox.com/v2"
SANDBOX = False
TOKEN_FILE = "access_token.txt"
DATE_FILE = "last_token_date.txt"
MAX_RETRIES = 3  # For API call retries
RETRY_DELAY = 5  # Seconds between retries
MAX_LOOP_RETRIES = 10  # Maximum retries for main loop failures

# Simulated real-time data storage
ws_data = {
    "nifty_spot": 23850,
    "prices": [23850] * 50,
    "vix": 15.0
}

# Helper functions for token management
def load_token():
    """Load the access token from a file."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                return f.read().strip()
        return None
    except Exception as e:
        logging.error(f"Error loading token: {e}")
        return None

def save_token(token):
    """Save the access token to a file."""
    try:
        with open(TOKEN_FILE, 'w') as f:
            f.write(token)
        logging.info("Access token saved to file")
    except Exception as e:
        logging.error(f"Error saving token: {e}")

def load_last_token_date():
    """Load the date of the last token validation."""
    try:
        if os.path.exists(DATE_FILE):
            with open(DATE_FILE, 'r') as f:
                return f.read().strip()
        return None
    except Exception as e:
        logging.error(f"Error loading last token date: {e}")
        return None

def save_last_token_date():
    """Save the current date as the last token validation date."""
    try:
        with open(DATE_FILE, 'w') as f:
            f.write(str(date.today()))
        logging.info("Last token date saved")
    except Exception as e:
        logging.error(f"Error saving last token date: {e}")

# Validate access token
def validate_access_token(token):
    """Validate the access token by checking user profile endpoint."""
    url = f"{BASE_URL}/user/profile"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            logging.info("Access token validated successfully")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Access token validation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return False

# Initialize API
def initialize_api():
    """Initialize Upstox API with a valid access token."""
    global ACCESS_TOKEN
    current_date = str(date.today())
    last_token_date = load_last_token_date()
    
    # Use stored token if validated today
    if last_token_date == current_date:
        stored_token = load_token()
        if stored_token and validate_access_token(stored_token):
            ACCESS_TOKEN = stored_token
            logging.info("Using stored access token from today")
            return True
    
    # Try loading token from file if not interactive
    stored_token = load_token()
    if stored_token and validate_access_token(stored_token):
        ACCESS_TOKEN = stored_token
        save_last_token_date()
        logging.info("Using stored access token")
        return True
    
    # Prompt for a new token in interactive mode
    if sys.stdin.isatty():
        print("Please provide a new Upstox access token for today (obtain from Upstox API dashboard):")
        new_token = input("Enter access token: ").strip()
    else:
        raise Exception("Non-interactive mode: Please update access_token.txt with a valid token")
    
    # Validate and save new token
    if validate_access_token(new_token):
        ACCESS_TOKEN = new_token
        save_token(new_token)
        save_last_token_date()
        logging.info("New access token validated and saved")
        return True
    else:
        raise Exception("Failed to initialize API: Invalid access token provided")

# Calculate next Wednesday for fallback expiry
def get_next_wednesday():
    """Calculate the next Wednesday for option expiry."""
    today = date.today()
    days_ahead = (2 - today.weekday()) % 7  # Wednesday is 2
    if days_ahead == 0:
        days_ahead = 7  # If today is Wednesday, use next Wednesday
    next_wednesday = today + timedelta(days=days_ahead)
    return next_wednesday.strftime("%Y-%m-%d")

# Check if today is an expiry day
def is_expiry_day(expiries, today):
    """Check if today matches any expiry date."""
    today_str = today.strftime("%Y-%m-%d")
    return today_str in expiries

# Fetch expiry date with logic for next expiry on expiry day
def get_expiry_date():
    """Fetch the nearest or next weekly expiry date for Nifty 50 options based on configuration."""
    url = f"{BASE_URL}/option/contract"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("data"):
                expiries = sorted(set(item.get("expiry") for item in data["data"] if item.get("weekly")))
                if not expiries:
                    logging.warning("No weekly expiries found, using fallback")
                    return get_next_wednesday()
                
                today = date.today()
                if USE_NEXT_EXPIRY_ON_EXPIRY_DAY and is_expiry_day(expiries, today):
                    # On expiry day, select the next expiry (at least one week away)
                    for expiry in expiries:
                        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
                        if expiry_date > today:
                            logging.info(f"Expiry day detected, using next expiry: {expiry}")
                            return expiry
                    logging.warning("No future expiry found, using fallback")
                    return get_next_wednesday()
                else:
                    # Use the nearest expiry
                    expiry = expiries[0]
                    logging.info(f"Fetched expiry: {expiry}")
                    return expiry
            logging.warning("Failed to fetch expiry, using fallback")
            return get_next_wednesday()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching expiry (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return get_next_wednesday()

# Fetch live Nifty 50 spot price
def fetch_spot_price():
    """Fetch live Nifty 50 spot price from Upstox API."""
    url = f"{BASE_URL}/market-quote/quotes"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    params = {"symbol": "NSE_INDEX|Nifty 50"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("data", {}).get("NSE_INDEX:Nifty 50"):
                spot_price = data["data"]["NSE_INDEX:Nifty 50"].get("last_price", ws_data["nifty_spot"])
                logging.info(f"Fetched spot price: {spot_price}")
                return spot_price
            logging.warning("Failed to fetch spot price, using fallback")
            return ws_data["nifty_spot"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching spot price (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return ws_data["nifty_spot"]

# Fetch option chain data
def fetch_option_chain(expiry_date):
    """Fetch option chain data for the given expiry date."""
    url = f"{BASE_URL}/option/chain"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Accept": "application/json"}
    params = {"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": expiry_date}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                logging.warning("Option chain API returned non-success status")
                return None
            df_data = []
            for item in data.get("data", []):
                call_volume = item.get("call_options", {}).get("market_data", {}).get("volume", 1) or 1
                put_volume = item.get("put_options", {}).get("market_data", {}).get("volume", 1) or 1
                pcr = item.get("pcr") or (put_volume / call_volume)
                df_data.append({
                    "strike_price": item.get("strike_price", 0),
                    "call_ltp": item.get("call_options", {}).get("market_data", {}).get("ltp", 100),
                    "put_ltp": item.get("put_options", {}).get("market_data", {}).get("ltp", 100),
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                    "call_oi": item.get("call_options", {}).get("market_data", {}).get("oi", 10000),
                    "put_oi": item.get("put_options", {}).get("market_data", {}).get("oi", 10000),
                    "call_iv": item.get("call_options", {}).get("option_greeks", {}).get("iv", 25),
                    "put_iv": item.get("put_options", {}).get("option_greeks", {}).get("iv", 25),
                    "pcr": pcr,
                    "call_instrument_key": item.get("call_options", {}).get("instrument_key", ""),
                    "put_instrument_key": item.get("put_options", {}).get("instrument_key", ""),
                    "call_bid_price": item.get("call_options", {}).get("market_data", {}).get("bid_price", 95),
                    "put_bid_price": item.get("put_options", {}).get("market_data", {}).get("bid_price", 95),
                    "call_ask_price": item.get("call_options", {}).get("market_data", {}).get("ask_price", 105),
                    "put_ask_price": item.get("put_options", {}).get("market_data", {}).get("ask_price", 105)
                })
            df = pd.DataFrame(df_data)
            logging.info(f"Option chain fetched for expiry {expiry_date}: {len(df)} strikes")
            return df
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching option chain (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None

# Calculate technical indicators
def calculate_indicators(prices):
    """Calculate technical indicators using TA-Lib."""
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < 50 or np.any(np.isnan(prices)) or np.any(prices <= 0):
            logging.error("Invalid price data for indicators")
            return None
        
        # Calculate indicators
        rsi = talib.RSI(prices, timeperiod=14)[-1]
        macd, signal, _ = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = macd[-1]
        signal = signal[-1]
        upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_upper = upper[-1]
        bb_lower = lower[-1]
        vwap = np.mean(prices[-20:])
        high = prices[-14:]
        low = prices[-14:]
        close = prices[-14:]
        atr = talib.ATR(high, low, close, timeperiod=14)[-1]
        
        # Validate outputs
        indicators = {
            "rsi": rsi if not np.isnan(rsi) and 0 < rsi < 100 else 50,
            "macd": macd if not np.isnan(macd) else 0,
            "signal": signal if not np.isnan(signal) else 0,
            "bb_upper": bb_upper if not np.isnan(bb_upper) else prices[-1] + 100,
            "bb_lower": bb_lower if not np.isnan(bb_lower) else prices[-1] - 100,
            "vwap": vwap if not np.isnan(vwap) else prices[-1],
            "atr": atr if not np.isnan(atr) and atr > 0 else 50
        }
        logging.debug(f"Calculated indicators: {indicators}")
        return indicators
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return None

# Calculate Fibonacci retracement levels
def calculate_fibonacci_levels(prices):
    """Calculate Fibonacci retracement levels based on recent price highs and lows."""
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < 50 or np.any(np.isnan(prices)) or np.any(prices <= 0):
            logging.error("Invalid price data for Fibonacci calculation")
            return None
        
        # Find the high and low over the last 50 prices
        high_price = np.max(prices[-50:])
        low_price = np.min(prices[-50:])
        price_range = high_price - low_price
        
        # Fibonacci levels: 38.2%, 50%, 61.8% (key levels for confirmation)
        fib_levels = {
            "38.2%": high_price - 0.382 * price_range,
            "50.0%": high_price - 0.500 * price_range,
            "61.8%": high_price - 0.618 * price_range,
            "high": high_price,
            "low": low_price
        }
        logging.debug(f"Fibonacci levels: {fib_levels}")
        return fib_levels
    except Exception as e:
        logging.error(f"Error calculating Fibonacci levels: {e}")
        return None

# Check market context
def check_market_context():
    """Check market volatility using VIX."""
    vix = ws_data["vix"]
    logging.debug(f"Checking market context: VIX = {vix}")
    if vix > 25:
        logging.warning("High volatility (VIX > 25). Skipping non-event trades.")
        return False
    return True

# Analyze market sentiment
def analyze_sentiment(df, spot_price):
    """Analyze market sentiment based on option chain data and Fibonacci levels."""
    try:
        # Find ATM strike
        atm_strike = min(df['strike_price'], key=lambda x: abs(x - spot_price))
        logging.debug(f"Spot price: {spot_price}, Selected ATM strike: {atm_strike}")
        atm_data = df[df['strike_price'] == atm_strike]
        if atm_data.empty:
            logging.warning(f"No ATM data found for strike: {atm_strike}")
            return None
        
        pcr = atm_data['pcr'].iloc[0]
        support = df.loc[df['put_oi'].idxmax(), 'strike_price']
        resistance = df.loc[df['call_oi'].idxmax(), 'strike_price']
        iv = (atm_data['call_iv'].iloc[0] + atm_data['put_iv'].iloc[0]) / 2
        oi_total = atm_data['call_oi'].iloc[0] + atm_data['put_oi'].iloc[0]
        
        # Calculate Fibonacci levels
        fib_levels = calculate_fibonacci_levels(ws_data["prices"])
        if fib_levels is None:
            logging.warning("Failed to calculate Fibonacci levels, using OI-based support/resistance")
            fib_support = support
            fib_resistance = resistance
        else:
            # Use Fibonacci 61.8% as support and 38.2% as resistance
            fib_support = fib_levels["61.8%"]
            fib_resistance = fib_levels["38.2%"]
            # Override OI-based support/resistance if Fibonacci levels are closer
            if abs(spot_price - fib_support) < abs(spot_price - support):
                support = fib_support
            if abs(spot_price - fib_resistance) < abs(spot_price - resistance):
                resistance = fib_resistance
        
        # Calculate direction score
        direction_score = 0
        if pcr < 0.8:
            direction_score += 1  # Bullish
        elif pcr > 1.2:
            direction_score -= 1  # Bearish
        if spot_price > resistance:
            direction_score += 1  # Breakout above resistance
        elif spot_price < support:
            direction_score -= 1  # Breakdown below support
        if iv > 30:
            direction_score += 0.5 * (1 if spot_price > atm_strike else -1)  # High IV amplifies direction
        # Fibonacci-based direction adjustment
        if fib_levels:
            if abs(spot_price - fib_levels["61.8%"]) < 50:
                direction_score += 0.5 if spot_price > fib_levels["50.0%"] else -0.5
            if abs(spot_price - fib_levels["38.2%"]) < 50:
                direction_score += 0.5 if spot_price > fib_levels["50.0%"] else -0.5
        
        direction = "Bullish" if direction_score > 0 else "Bearish" if direction_score < 0 else "Neutral"
        
        sentiment = {
            "pcr": pcr,
            "support": support,
            "resistance": resistance,
            "iv": iv,
            "atm_strike": atm_strike,
            "call_instrument_key": atm_data['call_instrument_key'].iloc[0],
            "put_instrument_key": atm_data['put_instrument_key'].iloc[0],
            "call_bid_price": atm_data['call_bid_price'].iloc[0],
            "put_bid_price": atm_data['put_bid_price'].iloc[0],
            "call_ask_price": atm_data['call_ask_price'].iloc[0],
            "put_ask_price": atm_data['put_ask_price'].iloc[0],
            "oi_total": oi_total,
            "direction": direction,
            "direction_score": direction_score,
            "fib_levels": fib_levels
        }
        logging.debug(f"Sentiment analysis with Fibonacci: {sentiment}")
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

# Generate trade signal with confirmation
def generate_trade_signal(sentiment, indicators, spot_price, prices):
    """Generate a confirmed trade signal when all conditions are met."""
    if not sentiment or not check_market_context() or not indicators:
        logging.debug("Trade signal generation skipped: Missing sentiment, market context, or indicators")
        return None
    
    atr = indicators['atr']
    breakout_threshold = atr
    recent_move = spot_price - prices[-5]
    is_breakout = abs(recent_move) >= breakout_threshold
    fib_levels = sentiment.get("fib_levels")
    
    # Initialize confirmation score
    confirmation_score = 0
    conditions_met = []
    
    # Fibonacci proximity check
    fib_proximity = None
    if fib_levels:
        for level in ["38.2%", "50.0%", "61.8%"]:
            if abs(spot_price - fib_levels[level]) < atr / 2:
                fib_proximity = level
                confirmation_score += 1
                conditions_met.append(f"Fibonacci proximity ({level}: {fib_levels[level]:.2f})")
                break
    
    # Trade type: Straddle
    is_consolidation = indicators['bb_upper'] - indicators['bb_lower'] < 2.5 * atr
    if (is_consolidation and 0.8 <= sentiment['pcr'] <= 2.0 and 18 <= sentiment['iv'] <= 30):
        straddle_score = 0
        if is_consolidation:
            straddle_score += 1
            conditions_met.append("Consolidation (BB width < 2.5 * ATR)")
        if 0.8 <= sentiment['pcr'] <= 2.0:
            straddle_score += 1
            conditions_met.append(f"PCR balanced ({sentiment['pcr']:.2f})")
        if 18 <= sentiment['iv'] <= 30:
            straddle_score += 1
            conditions_met.append(f"IV moderate ({sentiment['iv']:.2f})")
        confirmation_score += straddle_score
        if confirmation_score >= 3 and straddle_score >= 2:
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
            logging.info(f"Confirmed Straddle trade: Score={confirmation_score}, Conditions={conditions_met}")
            return trade
    
    # Trade type: CE (Bullish)
    is_bullish = (recent_move > 0 and spot_price > indicators['vwap'] and 
                  indicators['macd'] > indicators['signal'] and indicators['rsi'] < 65)
    bullish_score = 0
    if is_bullish:
        if recent_move > 0:
            bullish_score += 1
            conditions_met.append(f"Recent move up ({recent_move:.2f})")
        if spot_price > indicators['vwap']:
            bullish_score += 1
            conditions_met.append(f"Above VWAP ({spot_price:.2f} > {indicators['vwap']:.2f})")
        if indicators['macd'] > indicators['signal']:
            bullish_score += 1
            conditions_met.append(f"MACD bullish ({indicators['macd']:.2f} > {indicators['signal']:.2f})")
        if indicators['rsi'] < 65:
            bullish_score += 1
            conditions_met.append(f"RSI not overbought ({indicators['rsi']:.2f})")
        if sentiment['direction_score'] > 0:
            bullish_score += 1
            conditions_met.append(f"Sentiment bullish ({sentiment['direction_score']:.2f})")
        confirmation_score += bullish_score
        if confirmation_score >= 3 and bullish_score >= 3 and fib_proximity:
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
            # Adjust SL and target based on Fibonacci
            if fib_levels:
                trade['sl'] = max(0.2, trade['sl'] * (1 - 0.1 if spot_price > fib_levels["61.8%"] else 1))
                trade['target'] = min(0.7, trade['target'] * (1 + 0.1 if spot_price > fib_levels["38.2%"] else 1))
            logging.info(f"Confirmed CE trade: Score={confirmation_score}, Conditions={conditions_met}")
            return trade
    
    # Trade type: PE (Bearish)
    is_bearish = (recent_move < 0 and spot_price < indicators['vwap'] and 
                  indicators['macd'] < indicators['signal'] and indicators['rsi'] > 35)
    bearish_score = 0
    if is_bearish or (sentiment['pcr'] > 3.0 and sentiment['direction_score'] < 0):
        if recent_move < 0:
            bearish_score += 1
            conditions_met.append(f"Recent move down ({recent_move:.2f})")
        if spot_price < indicators['vwap']:
            bearish_score += 1
            conditions_met.append(f"Below VWAP ({spot_price:.2f} < {indicators['vwap']:.2f})")
        if indicators['macd'] < indicators['signal']:
            bearish_score += 1
            conditions_met.append(f"MACD bearish ({indicators['macd']:.2f} < {indicators['signal']:.2f})")
        if indicators['rsi'] > 35:
            bearish_score += 1
            conditions_met.append(f"RSI not oversold ({indicators['rsi']:.2f})")
        if sentiment['pcr'] > 3.0 or sentiment['direction_score'] < 0:
            bearish_score += 1
            conditions_met.append(f"Sentiment bearish (PCR={sentiment['pcr']:.2f}, Score={sentiment['direction_score']:.2f})")
        confirmation_score += bearish_score
        if confirmation_score >= 3 and bearish_score >= 3 and fib_proximity:
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
            # Adjust SL and target based on Fibonacci
            if fib_levels:
                trade['sl'] = max(0.2, trade['sl'] * (1 - 0.1 if spot_price < fib_levels["38.2%"] else 1))
                trade['target'] = min(0.7, trade['target'] * (1 + 0.1 if spot_price < fib_levels["61.8%"] else 1))
            logging.info(f"Confirmed PE trade: Score={confirmation_score}, Conditions={conditions_met}")
            return trade
    
    logging.debug(f"No confirmed trade: Score={confirmation_score}, Conditions={conditions_met}")
    return None

# Execute trade
def execute_trade(trade, df, atr):
    """Execute the trade with risk management and premium validation."""
    try:
        atm_data = df[df['strike_price'] == trade['strike']]
        if atm_data.empty:
            logging.error("No ATM data for trade execution")
            return None
        
        ce_premium = atm_data['call_ask_price'].iloc[0]
        pe_premium = atm_data['put_ask_price'].iloc[0]
        
        # Warn if premiums are too low
        if trade['type'] in ["PE", "Straddle"] and pe_premium < 20:
            logging.warning(f"Low PE premium: ₹{pe_premium:.2f} for strike {trade['strike']}")
        if trade['type'] in ["CE", "Straddle"] and ce_premium < 20:
            logging.warning(f"Low CE premium: ₹{ce_premium:.2f} for strike {trade['strike']}")
        
        # Adjust SL and target based on ATR and IV
        iv_factor = atm_data['call_iv'].iloc[0] / 25
        sl_factor = 0.1 + (atr / 1000) * iv_factor
        target_factor = 0.3 + (atr / 500) * iv_factor
        
        # Calculate trade parameters
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
        
        # Risk management: Calculate lots
        max_loss = CAPITAL * RISK_PER_TRADE
        monetary_sl = (total_premium - trade['sl_price']) * LOT_SIZE
        lots = min(10, max_loss / monetary_sl) if monetary_sl > 0 else 1
        trade['lots'] = int(lots)  # Round down to nearest integer
        if trade['lots'] == 0:
            logging.warning("Calculated lots is 0, setting to 1")
            trade['lots'] = 1
        
        logging.info(f"Executed confirmed trade: {trade}")
        return trade
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

# Save trade signals to CSV
def save_trade_to_csv(trade, indicators, filename):
    """Save trade details to a CSV file with unique filename."""
    os.makedirs("trade_signals", exist_ok=True)
    # Append timestamp to avoid conflicts
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{base}_{timestamp}{ext}"
    
    fields = ["Time", "Type", "Strike", "CE_Entry", "PE_Entry", "CE_Instrument", "PE_Instrument", 
              "SL_Price", "Target_Price", "Lots", "RSI", "MACD", "ATR", "VIX", "Direction", 
              "Confirmation_Score", "Fib_Proximity"]
    trade_data = {
        "Time": trade["time"],
        "Type": trade["type"],
        "Strike": trade["strike"],
        "CE_Entry": trade.get("ce_entry", None),
        "PE_Entry": trade.get("pe_entry", None),
        "CE_Instrument": trade.get("ce_instrument_key", None),
        "PE_Instrument": trade.get("pe_instrument_key", None),
        "SL_Price": trade["sl_price"],
        "Target_Price": trade["target_price"],
        "Lots": trade["lots"],
        "RSI": indicators["rsi"],
        "MACD": indicators["macd"],
        "ATR": indicators["atr"],
        "VIX": ws_data["vix"],
        "Direction": trade["direction"],
        "Confirmation_Score": trade["confirmation_score"],
        "Fib_Proximity": trade["fib_proximity"]
    }
    try:
        with open(unique_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(trade_data)
        logging.info(f"Trade saved to CSV: {unique_filename}")
    except Exception as e:
        logging.error(f"Error saving trade to CSV: {e}")

# Main trading loop
def main():
    """Main trading loop for Nifty 50 intraday trading with next expiry on expiry day."""
    # Validate Telegram configuration
    if not validate_telegram_config():
        if sys.stdin.isatty():
            print("Telegram configuration invalid. Please check Bot Token and Chat ID.")
            new_chat_id = input("Enter a valid Telegram Chat ID (numeric or @channel): ").strip()
            global TELEGRAM_CHAT_ID
            TELEGRAM_CHAT_ID = new_chat_id
            if not validate_telegram_config():
                raise Exception("Failed to validate Telegram configuration after retry")
        else:
            raise Exception("Non-interactive mode: Please update TELEGRAM_CHAT_ID with a valid ID")

    initialize_api()
    trades_today = 0
    expiry_date = get_expiry_date()
    logging.info(f"Starting intraday trading for Nifty 50 with Upstox API (Expiry: {expiry_date})...")
    
    csv_filename = f"trade_signals/nifty50_trades_{expiry_date}"
    
    loop_retries = 0
    while trades_today < MAX_TRADES_PER_DAY and loop_retries < MAX_LOOP_RETRIES:
        current_time = datetime.now().strftime("%H:%M")
        
        if not (TRADING_HOURS['start'] <= current_time <= TRADING_HOURS['end']):
            logging.info(f"Outside trading hours: {current_time}")
            time.sleep(60)
            continue
        
        df = fetch_option_chain(expiry_date)
        if df is None:
            logging.warning(f"Failed to fetch option chain for expiry {expiry_date}, retrying in 60s")
            loop_retries += 1
            time.sleep(60)
            continue
        
        # Fetch live spot price
        spot_price = fetch_spot_price()
        ws_data["nifty_spot"] = spot_price
        ws_data["prices"].append(spot_price)
        ws_data["prices"] = ws_data["prices"][-50:]
        ws_data["vix"] = max(10, min(25, ws_data["vix"] + np.random.normal(0, 1)))  # Simulated VIX
        
        indicators = calculate_indicators(ws_data["prices"])
        if indicators is None:
            logging.warning("Failed to calculate indicators, retrying in 60s")
            loop_retries += 1
            time.sleep(60)
            continue
        
        sentiment = analyze_sentiment(df, spot_price)
        if sentiment is None:
            logging.warning("Failed to analyze sentiment, retrying in 60s")
            loop_retries += 1
            time.sleep(60)
            continue
        
        trade = generate_trade_signal(sentiment, indicators, spot_price, ws_data["prices"])
        
        if trade:
            trade = execute_trade(trade, df, indicators['atr'])
            if trade:
                trades_today += 1
                loop_retries = 0  # Reset retries on successful trade
                save_trade_to_csv(trade, indicators, csv_filename)
                
                # Prepare Telegram message
                message = f"<b>Confirmed Trade</b>\n" \
                          f"<b>Time:</b> {trade['time']}\n" \
                          f"<b>Type:</b> {trade['type']}\n" \
                          f"<b>Strike:</b> {trade['strike']}\n" \
                          f"<b>Expiry:</b> {expiry_date}\n" \
                          f"<b>Market Direction:</b> {trade['direction']}\n" \
                          f"<b>Confirmation Score:</b> {trade['confirmation_score']}\n" \
                          f"<b>Fibonacci Proximity:</b> {trade['fib_proximity']}\n"
                
                if trade['type'] == "Straddle":
                    message += f"<b>CE Entry:</b> ₹{trade['ce_entry']:.2f}, Instrument: {trade['ce_instrument_key']}\n" \
                               f"<b>PE Entry:</b> ₹{trade['pe_entry']:.2f}, Instrument: {trade['pe_instrument_key']}\n" \
                               f"<b>SL:</b> ₹{trade['sl_price']:.2f}, <b>Target:</b> ₹{trade['target_price']:.2f}\n"
                else:
                    entry = trade['ce_entry'] if trade['type'] == "CE" else trade['pe_entry']
                    instrument = trade['ce_instrument_key'] if trade['type'] == "CE" else trade['pe_instrument_key']
                    message += f"<b>Entry:</b> ₹{entry:.2f}, Instrument: {instrument}\n" \
                               f"<b>SL:</b> ₹{trade['sl_price']:.2f}, <b>Target:</b> ₹{trade['target_price']:.2f}\n"
                
                message += f"<b>Lots:</b> {trade['lots']}\n" \
                           f"<b>Indicators:</b> RSI={indicators['rsi']:.2f}, MACD={indicators['macd']:.2f}, " \
                           f"ATR={indicators['atr']:.2f}, VIX={ws_data['vix']:.2f}"
                
                # Send to Telegram
                send_telegram_message(message)
                
                # Print to console
                print("\nCONFIRMED TRADE: ALL CONDITIONS MET. START TRADE ONLY AFTER 9:25 AM. DO NOT TRADE AFTER 2:50 PM. ONLY ONE STOP-LOSS TRADE ALLOWED PER DAY. FOLLOW THE RULES STRICTLY.")
                print(f"Time: {trade['time']}")
                print(f"Type: {trade['type']}")
                print(f"Strike: {trade['strike']}")
                print(f"Expiry: {expiry_date}")
                print(f"Market Direction: {trade['direction']}")
                print(f"Confirmation Score: {trade['confirmation_score']}")
                print(f"Fibonacci Proximity: {trade['fib_proximity']}")
                if trade['type'] == "Straddle":
                    print(f"CE Entry: ₹{trade['ce_entry']:.2f}, Instrument: {trade['ce_instrument_key']}")
                    print(f"PE Entry: ₹{trade['pe_entry']:.2f}, Instrument: {trade['pe_instrument_key']}")
                    print(f"SL: ₹{trade['sl_price']:.2f}, Target: ₹{trade['target_price']:.2f}")
                else:
                    entry = trade['ce_entry'] if trade['type'] == "CE" else trade['pe_entry']
                    instrument = trade['ce_instrument_key'] if trade['type'] == "CE" else trade['pe_instrument_key']
                    print(f"Entry: ₹{entry:.2f}, Instrument: {instrument}")
                    print(f"SL: ₹{trade['sl_price']:.2f}, Target: ₹{trade['target_price']:.2f}")
                print(f"Lots: {trade['lots']}")
                print(f"Indicators: RSI={indicators['rsi']:.2f}, MACD={indicators['macd']:.2f}, "
                      f"ATR={indicators['atr']:.2f}, VIX={ws_data['vix']:.2f}")
        
        time.sleep(60)
    
    if loop_retries >= MAX_LOOP_RETRIES:
        logging.error("Max loop retries reached, exiting")
    else:
        logging.info("Max trades reached for today.")

if __name__ == "__main__":
    main()