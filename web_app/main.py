import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime, timedelta

os.makedirs("web_app/static", exist_ok=True)
os.makedirs("web_app/templates", exist_ok=True)

app = FastAPI(title="DỰ ĐOÁN GIÁ TÀI SẢN BẰNG AI")

app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

# Danh sách tài sản
ASSETS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance Coin",
    "GC=F": "Vàng",
    "^VNINDEX": "VN-Index (dùng ^VNI)"
}

# Load model chung (dùng cho tất cả)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(50, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(torch.load("models/lstm_baseline_window60.pth", map_location="cpu"))
model.eval()

# Cache dữ liệu gần nhất
data_cache = {}

def get_recent_data(ticker, days=60):
    if ticker not in data_cache or datetime.now() - data_cache[ticker]["time"] > timedelta(minutes=5):
        df = yf.download(ticker, period="3mo", interval="1d")["Close"].tail(days+10)
        scaler = joblib.load("data/processed/scaler.npy", allow_pickle=True).item()
        scaled = scaler.transform(df.values.reshape(-1,1))
        data_cache[ticker] = {"data": scaled[-days:], "scaler": scaler, "time": datetime.now()}
    return data_cache[ticker]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "assets": ASSETS})

@app.get("/predict/{ticker}/{days:int}")
async def predict(ticker: str, days: int):
    if ticker not in ASSETS or days not in [1,3,5,7]:
        return {"error": "Invalid request"}
    
    cache = get_recent_data(ticker)
    current = torch.FloatTensor(cache["data"]).unsqueeze(0)
    preds = []
    
    with torch.no_grad():
        temp = current.clone()
        for _ in range(days):
            pred = model(temp)
            preds.append(pred.item())
            temp = torch.cat([temp[:, 1:, :], pred.unsqueeze(2)], dim=1)
    
    preds_usd = cache["scaler"].inverse_transform(np.array(preds).reshape(-1,1))
    predictions = [round(float(p[0]), 2) for p in preds_usd]
    
    return {"ticker": ASSETS[ticker], "predictions": predictions}