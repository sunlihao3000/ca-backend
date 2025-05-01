# Fine-Tune-DeepSeek
Fine Tune DeepSeek


python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 finetune_dataset.py

#run local pastapi server
uvicorn app:app --host 0.0.0.0 --port 8000
uvicorn app-thinking:app --host 0.0.0.0 --port 8000




