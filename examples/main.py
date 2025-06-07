# examples/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_utils import read_image
from llm_handler import generate_response, load_model

if __name__ == "__main__":
    image_path = "example.png"  # replace with the real path
    prompt = "Summarize the following text: "

    tokenizer, model = load_model()
    text = read_image(image_path)
    question = f"{prompt}
{text}"
    response = generate_response(question, tokenizer, model)

    print("Generated response:", response)
