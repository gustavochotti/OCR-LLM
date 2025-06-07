# download_model.py
import os
import sys
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

def install(command: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + command.split())

def get_hf_token() -> str:
    token = input("Enter your Hugging Face token: ").strip()
    if not token.startswith("hf_"):
        raise ValueError("Invalid token format.")
    os.environ["HF_TOKEN"] = token
    return token

def download_and_save_model(model_id: str, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    install("-r requirements.txt")
    get_hf_token()
    download_and_save_model("google/gemma-2-2b-it", "gemma-2b-it")
