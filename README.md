# ocr2llm

A professional-grade OCR + LLM pipeline combining EasyOCR and Gemma-2B-IT to extract text from images and process it with a large language model. It enables use cases such as automatic summarization of scanned documents, information extraction from receipts or invoices, image-to-text transformation for accessibility, and custom prompting over OCR-extracted content for downstream language processing tasks.

## 📦 Project Structure

```
ocr2llm/
├── cli.py                  # Main CLI interface
├── config.py               # Configs (e.g., model path)
├── exceptions.py           # Custom exception classes
├── llm_handler.py          # Load & run LLM (Gemma)
├── ocr_utils.py            # EasyOCR + OpenCV preprocessing
├── torch_utils.py          # GPU/dtype helpers
│
├── examples/
│   ├── __init__.py  
│   └── main.py             # Minimal direct usage example
│
├── tests/
│   ├── __init__.py
│   ├── test_exceptions.py
│   └── test_integration.py
│
├── images/
│   └── receipt.jpg         # Example image
│
├── requirements.txt        # Dependencies
├── .gitignore              # Git exclusions
├── download_model.py       # Setup script to install deps & download model
├── README.md               # This file
```

---

## 📥 Clone the Repository
```bash
git clone https://github.com/gustavochotti/ocr2llm.git
cd ocr2llm
```

---

## 🔧 Installation & Setup

### Step-by-Step Execution Guide

#### Step 1: Get Your Hugging Face Token
You need a Hugging Face account and an access token to download the Gemma model.

1. Go to: https://huggingface.co
2. Log in or sign up.
3. Click your profile picture > **Settings**.
4. Go to **Access Tokens**.
5. Click **New token**, name it (e.g., `gemma-download`), set role as **read**.
6. Copy the token (starts with `hf_`).

#### Step 2: Accept the Gemma Model's Terms

1. Go to: https://huggingface.co/google/gemma-2-2b-it
2. Read and **accept the license terms**.

#### Step 3: Run the Script
```bash
python download_model.py
```
You will be prompted to paste your Hugging Face token. The model will download and be saved to `gemma-2b-it/`.

---

## 🚀 Usage

### CLI Usage
Run directly from terminal:
```bash
python -m cli --image example.png --prompt "Summarize this text:"
```

#### Optional arguments:
- `--max_tokens`: Number of tokens to generate (default: `768`)
- `--output`: Save the result to a file `response.txt`

### Full example
```bash
python -m cli --image images/receipt.jpg --prompt "List all prices and total." --max_tokens 512 --output
```

#### Output (example)
```txt
Item 1: $3.50
Item 2: $5.00
Total: $8.50
```

### Interactive Demo (main.py)
You can also run a minimal hardcoded example:
```bash
python examples/main.py
```
> Ensure `example.png` exists in the root directory or update the path.

---

## 🧪 Tests
Run unit and integration tests:
```bash
pytest tests/
```

---

## 🛠 Dependencies
Install requirements manually:
```bash
pip install -r requirements.txt
```
Or automatically using `download_model.py`, which installs all requirements and the model.

---

## 🧠 How it works
- **EasyOCR** handles multilingual OCR.
- **OpenCV** preprocesses the image (grayscale, blur, binarize).
- **Gemma-2B-IT** (locally) runs the prompt including OCR’d text.

> The full prompt format used internally:
```
<start_of_turn>user
[prompt + OCR text]
<end_of_turn>
<start_of_turn>model
```

---

## ✅ Output Control
- Clean output (special tokens skipped)
- Can stream and save to file
- Handles OCR/LLM exceptions gracefully

---

## 📂 License
This project is under the MIT License. Refer to each model’s license on Hugging Face for usage terms.

---

## 🙋 FAQ
- **Q**: What happens if I don't accept the model terms?
  - **A**: The model download will fail with a 403 authorization error.

- **Q**: Can I use another LLM?
  - **A**: Yes, edit `config.py` and change `LOCAL_MODEL_PATH` to another compatible model.

- **Q**: Why is OpenCV used?
  - **A**: To maximize OCR accuracy with noise-reduced, binarized inputs.

---

## ⭐️ Star This Repo
If you found this useful, consider giving it a star!

> Built with ❤️ by Gustavo Chotti – fine-tuned for OCR + LLM workflows.
