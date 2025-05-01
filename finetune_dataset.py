import pandas as pd
import json
from sklearn.utils import resample

# ================================
# 1️⃣ Load the Dataset
# ================================
file_path = "dataset.csv"  # Change this if needed

df = pd.read_csv(file_path)

# ================================
# 2️⃣ Data Cleaning
# ================================

# Ensure 'timestamp' column is properly formatted
try:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
except Exception as e:
    print(f"⚠️ Timestamp conversion failed: {e}")

# Ensure 'rule.description' is string type to avoid `.str.contains()` error
df["rule.description"] = df["rule.description"].astype(str)

# Fill missing values in 'data.srcip' with "unknown"
df["data.srcip"] = df["data.srcip"].fillna("unknown")

# Remove duplicate logs
df = df.drop_duplicates()

# ================================
# 3️⃣ Bias Analysis & Mitigation
# ================================

# Identify top frequent IPs
srcip_distribution = df["data.srcip"].value_counts(normalize=True) * 100
top_ips = srcip_distribution.head(10).index.tolist()

# Handling Overrepresented Log Types

# Select SSH authentication failure logs (ensure filtering works)
ssh_logs = df[df["rule.description"].str.contains("sshd|PAM", case=False, na=False)]

# Downsample SSH logs (only if they exist)
if len(ssh_logs) > 0:
    ssh_downsampled = resample(ssh_logs, replace=False, n_samples=min(len(ssh_logs), int(len(ssh_logs) * 0.5)), random_state=42)
else:
    ssh_downsampled = ssh_logs  # No resampling needed

# Select logs that are NOT related to SSH/PAM
non_ssh_logs = df[~df["rule.description"].str.contains("sshd|PAM", case=False, na=False)]

# Upsample non-SSH logs (only if they exist)
if len(non_ssh_logs) > 0:
    non_ssh_upsampled = resample(non_ssh_logs, replace=True, n_samples=min(len(df), int(len(non_ssh_logs) * 2)), random_state=42)
else:
    non_ssh_upsampled = non_ssh_logs  # No resampling needed

# Combine the balanced dataset
df_balanced = pd.concat([ssh_downsampled, non_ssh_upsampled])

# Normalize frequent attacker IPs
df_balanced["data.srcip"] = df_balanced["data.srcip"].apply(lambda x: "frequent_attacker" if x in top_ips else "new_ip")

# ================================
# 4️⃣ Preparing Data for Fine-Tuning
# ================================

# Ensure 'full_log' exists before processing
if "full_log" not in df_balanced.columns:
    print("❌ Error: 'full_log' column is missing.")
    exit(1)

# Convert logs into JSONL format
jsonl_data = df_balanced.apply(lambda row: {
    "input": f"Analyze this security event: {row['full_log']}",
    "output": f"Rule Triggered: {row['rule.description']} | Severity Level: {row['rule.level']} | Source: {row['data.srcip']}"
}, axis=1).tolist()

# Save to JSONL file
jsonl_file_path = "fine_tuning_dataset.jsonl"
with open(jsonl_file_path, "w") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ JSONL dataset saved as: {jsonl_file_path}")
