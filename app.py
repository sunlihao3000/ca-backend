# app.py
# FastAPI server for Security Event Analyzer with CORS enabled

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline
)
import torch

# ---------- Load model on startup ----------
MODEL_DIR = "./distil_model_ft"  # path to your fine-tuned model


device = "mps" if torch.backends.mps.is_available() else "cpu"
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
# device=0 uses MPS/Metal on Apple Silicon; -1 for CPU
clf = pipeline(
    "text-classification", model=model, tokenizer=tokenizer,
    device=0 if device == "mps" else -1
)

# ---------- FastAPI setup ----------
app = FastAPI(title="Security LLM API")

# Enable CORS for your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LogQuery(BaseModel):
    log_text: str

@app.post("/predict")
def predict(q: LogQuery):
    try:
        result = clf(q.log_text, truncation=True, max_length=512)[0]
        # result example: {'label': 'LABEL_1', 'score': 0.92}
        return {"label": result["label"], "score": float(result["score"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Run with: uvicorn app:app --reload -----------
