from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import random
from typing import List
from transformers import pipeline

app = FastAPI(title="PSX Quant Terminal API", version="1.0")

# --- INITIALIZE DATABASE ---
def init_db():
    conn = sqlite3.connect("trades.db", timeout=10)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE, password TEXT, 
                 plan TEXT, trx_id TEXT, promo_code TEXT, expiry_date TEXT, status TEXT)""")
    conn.commit()
    conn.close()

init_db()

# --- INITIALIZE FinBERT AI ---
print("Loading FinBERT Sentiment Model... (This takes a moment on first boot)")
try:
    nlp_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print(f"Failed to load AI: {e}")
    nlp_model = None

# --- DATA MODELS ---
class StockAnalysis(BaseModel):
    symbol: str
    live_price: float
    sma_50: float
    atr: float
    stop_loss: float
    take_profit: float
    news_headline: str
    sentiment: str
    confidence_score: float
    signal: str

# --- AI & QUANT ENGINE ---
def fetch_sentiment(symbol: str):
    """Simulates a live news scraper feed for the AI to process."""
    headlines = [
        f"{symbol} reports strong quarterly earnings, exceeding market expectations.",
        f"Regulatory concerns and tax hikes hit {symbol} operations.",
        f"{symbol} maintains standard dividend payout amidst average trading volume."
    ]
    headline = random.choice(headlines)
    
    if nlp_model:
        ai_result = nlp_model(headline)[0]
        return headline, ai_result['label'].upper(), round(ai_result['score'] * 100, 2)
    return headline, "NEUTRAL", 50.0

def calculate_quant_metrics(symbol: str):
    """Fuses mathematical risk boundaries with AI sentiment for a final signal."""
    try:
        df = yf.Ticker(f"{symbol}.KA").history(period="6mo")
        if df.empty or len(df) < 50: return None
            
        current_price = float(df['Close'].iloc[-1])
        sma_50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
        
        # ATR Calculation
        df['Prev_Close'] = df['Close'].shift(1)
        df['TR'] = df[['High', 'Low', 'Prev_Close']].apply(
            lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Prev_Close']), abs(x['Low'] - x['Prev_Close'])), axis=1
        )
        atr_14 = float(df['TR'].rolling(window=14).mean().iloc[-1])
        
        stop_loss = current_price - (1.5 * atr_14)
        take_profit = current_price + (2.0 * atr_14)
        
        # Get AI Sentiment
        headline, sentiment, confidence = fetch_sentiment(symbol)
        
        # Master Decision Logic (Math + AI)
        if current_price > sma_50 and sentiment == "POSITIVE" and confidence > 65.0:
            signal = "🟢 AI BUY"
        elif sentiment == "NEGATIVE":
            signal = "🔴 IGNORE (BEARISH NEWS)"
        else:
            signal = "🟡 HOLD / NEUTRAL"
            
        return {
            "symbol": symbol,
            "live_price": round(current_price, 2),
            "sma_50": round(sma_50, 2),
            "atr": round(atr_14, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "news_headline": headline,
            "sentiment": sentiment,
            "confidence_score": confidence,
            "signal": signal
        }
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

# --- CATEGORIZED STOCKS ---
STABLE_STOCKS = ["HUBC", "EFERT", "FFC", "MCB", "UBL"]

# --- API ENDPOINTS ---
@app.get("/api/matrix/stable", response_model=List[StockAnalysis])
def get_stable_matrix():
    results = [calculate_quant_metrics(sym) for sym in STABLE_STOCKS]
    return [res for res in results if res is not None]