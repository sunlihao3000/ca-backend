import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from peft import PeftModel, PeftConfig

MODEL_DIR = "./deepseek_qwen_lora_ft"
DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"

# ── 1️⃣ Read LoRA config to find the original base model ────────────────
peft_cfg  = PeftConfig.from_pretrained(MODEL_DIR)
BASE_NAME = peft_cfg.base_model_name_or_path

# ── 2️⃣ Load base model + tokenizer ─────────────────────────────────────
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    device_map="auto" if DEVICE != "cpu" else {"": "cpu"},
)
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, trust_remote_code=True)

# ── 3️⃣ Attach the LoRA adapter ─────────────────────────────────────────
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

# ── 4️⃣ Fix id2label for human-friendly output ───────────────────────────
#    Ensure 0→"benign", 1→"malicious"
model.config.id2label = {0: "benign", 1: "malicious"}
model.config.label2id = {"benign": 0, "malicious": 1}

# ── 5️⃣ Build inference pipeline (no device arg here) ────────────────────
clf = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True,    # we need both class scores
)

# ── FastAPI setup ───────────────────────────────────────────────────────
app = FastAPI(title="Security LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LogQuery(BaseModel):
    log_text: str
    malicious_threshold: float = 0.1

@app.post("/predict")
def predict(q: LogQuery):
    try:
        # 1) run the pipeline
        raw = clf(
            q.log_text,
            truncation=True,
            max_length=512
        )
        # raw can be List[dict] OR List[List[dict]] depending on HF version
        # normalize to List[dict]:
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            scores_list = raw[0]
        else:
            scores_list = raw  # expected list of dicts

        # 2) extract scores
        benign_score    = next(item["score"] for item in scores_list if item["label"] == "benign")
        malicious_score = next(item["score"] for item in scores_list if item["label"] == "malicious")

        # 3) decide final label
        final_label = "malicious" if malicious_score >= q.malicious_threshold else "benign"

        return {
            "scores": {
                "benign":    benign_score,
                "malicious": malicious_score
            },
            "threshold": q.malicious_threshold,
            "label": final_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
