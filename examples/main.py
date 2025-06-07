# main.py
from ocr_utils import read_image
from llm_handler import generate_response, load_model

if __name__ == "__main__":
    image_path = "example.png"
    prompt = "Summarize the following text:"

    tokenizer, model = load_model()
    text = read_image(image_path)
    question = f"{prompt}\n{text}"
    response = generate_response(question, tokenizer, model)

    print("\nGenerated response:\n", response)
