# Fine-Tune-DeepSeek
Fine Tune DeepSeek

# for deepseek mode
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 finetune_dataset.py

#run local fastapi server
uvicorn app:app --host 0.0.0.0 --port 8000
uvicorn app-thinking:app --host 0.0.0.0 --port 8000

# for distil_model_ft because the folder is too big, we are not able to upload to the github. so we uploaded to google drive
# you could go to the link below and download the model and unzip to the root of the project and called the folder 'distil_model_ft'
# then run this mode on local
https://drive.google.com/file/d/1dgZSpJ65twzqYkeck5bXOOH4d8QrH-Qa/view?usp=drive_link




