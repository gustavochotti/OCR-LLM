from ocr_utils import read_image
from llm_handler import generate_response, load_model

if __name__ == "__main__":
  tokenizer, model = load_model()
  image_path = "IMAGE.png"  # or the image path
  text = read_image(image_path)
  question = f"Summarize this text:\n{text}"

  generate_response(question, tokenizer, model)
