import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import numpy as np

os.makedirs("web_app/static", exist_ok=True)
os.makedirs("web_app/templates", exist_ok=True)

app = FastAPI(title="DỰ ĐOÁN GIÁ BITCOIN - LSTM")

app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")

MODEL_PATH = "models\lstm_baseline_window60.pth"
SCALER_PATH = "processing\data\processed\scaler.npy"

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(50, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

scaler = np.load(SCALER_PATH, allow_pickle=True).item()
X_recent = np.load("processing\data\processed\X_test.npy")[-1:]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict/{days:int}")
async def predict(days: int):
    if days not in [1, 3, 5, 7]:
        return {"error": "Chỉ hỗ trợ 1, 3, 5, 7 ngày"}
    
    current = torch.FloatTensor(X_recent)
    preds = []
    with torch.no_grad():
        for _ in range(days):
            pred = model(current)
            preds.append(pred.item())
            current = torch.cat([current[:, 1:, :], pred.unsqueeze(2)], dim=1)
    
    preds_usd = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    predictions = [round(float(p[0]), 2) for p in preds_usd]
    
    return {"days": days, "predictions": predictions}