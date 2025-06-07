import os
import sys
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

def install(command: str):
    try:
        print(f"\nInstalling: {command}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + command.split())
        print(f"Successfully installed: {command}\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install '{command}': {e}")
        sys.exit(1)

def get_hf_token() -> str:
    token = input("Please, enter your Hugging Face token to start model download: ").strip()
    if not token.startswith("hf_"):
        raise ValueError("Invalid Hugging Face token. Please, check your token and try again.")
    os.environ["HF_TOKEN"] = token
    return token

def download_and_save_model(model_id: str, save_path: str):
    try:
        os.makedirs(save_path, exist_ok=True)

        print(f"\n🔽 Downloading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
        tokenizer.save_pretrained(save_path)

        print(f"\n🔽 Downloading model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
        model.save_pretrained(save_path)

        print(f"\n✅ Model saved locally at: {save_path}")
    except Exception as e:
        print(f"Error while downloading the model: {e}")
        raise

if __name__ == "__main__":
    print("Setting up environment...")

    # Step 1: Install dependencies
    install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    install("-r requirements.txt")

    # Step 2: Get token and download model
    get_hf_token()
    MODEL_ID = "google/gemma-2-2b-it"
    SAVE_PATH = "gemma-2b-it"

    # Step 3: Download and save
    download_and_save_model(MODEL_ID, SAVE_PATH)
