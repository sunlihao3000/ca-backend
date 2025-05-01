import pandas as pd
import json
from sklearn.utils import resample

# ================================
# 1️⃣ Load the Dataset
# ================================
file_path = "dataset.csv"  # Change to your CSV file path
df = pd.read_csv(file_path)

# ================================
# 2️⃣ Parse Timestamp & Basic Cleaning
# ================================
# Expected format: "Mar 13, 2025 @ 20:45:21.476"
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%b %d, %Y @ %H:%M:%S.%f",
    errors="coerce"
)

# Fill missing values and ensure required columns
# Source IP
if "data.srcip" in df.columns:
    df["data.srcip"] = df["data.srcip"].fillna("unknown")
else:
    df["data.srcip"] = "unknown"
# Source port
if "data.srcport" in df.columns:
    df["data.srcport"] = df["data.srcport"].fillna("unknown")
else:
    df["data.srcport"] = "unknown"
# Full log
if "full_log" not in df.columns:
    raise ValueError("Missing required column: full_log")
df = df.dropna(subset=["full_log"])  # keep only rows with full_log

# Geolocation fields
for col in ["GeoLocation.country_name", "GeoLocation.city_name", "GeoLocation.location"]:
    if col in df.columns:
        df[col] = df[col].fillna("unknown")
    else:
        df[col] = "unknown"

# Remove duplicates to avoid over-representation
df = df.drop_duplicates()

# ================================
# 3️⃣ Bias Mitigation (Optional)
# ================================
# Add any resampling here if needed based on labels
# Example: balance on a 'label' column (malicious vs benign)

# ================================
# 4️⃣ Prepare JSONL for Fine-Tuning
# ================================
jsonl_data = []
for _, row in df.iterrows():
    # Format timestamp for prompt
    ts = row["timestamp"]
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if pd.notna(ts) else ""

    # Clean the full_log by replacing newlines
    log_clean = str(row['full_log']).replace('\n', ' ')

    # Construct the input prompt with key fields
    inp = (
        f"Timestamp: {ts_str} | "
        f"SrcIP: {row['data.srcip']} | "
        f"SrcPort: {row['data.srcport']} | "
        f"Country: {row['GeoLocation.country_name']} | "
        f"City: {row['GeoLocation.city_name']} | "
        f"Location: {row['GeoLocation.location']} | "
        f"Log: {log_clean}"
    )
    # Use a 'label' column if present, otherwise leave output empty
    out = row['label'] if 'label' in df.columns else ""

    jsonl_data.append({"input": inp, "output": out})

# Save to JSONL file
jsonl_file_path = "fine_tuning_dataset_2.jsonl"
with open(jsonl_file_path, "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ JSONL dataset saved: {jsonl_file_path}")
