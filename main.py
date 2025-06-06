from ocr_utils import read_image
from llm_handler import generate_response, load_model

if __name__ == "__main__":
  tokenizer, model = load_model()
  image_path = "exemplo.png"
  text = read_image(image_path)
  question = f"Responda a seguinte pergunta:\n{text}"

  generate_response(question, tokenizer, model)
