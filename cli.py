# cli.py
import argparse
import logging
from ocr_utils import read_image
from llm_handler import generate_response, load_model
from exceptions import OCRProcessingError, ResponseGenerationError

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="OCR to LLM CLI")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", required=True, help="Prompt to pass to the LLM")
    parser.add_argument("--max_tokens", type=int, default=768, help="Maximum number of tokens to generate")
    parser.add_argument("--output", action="store_true", help="Save the response to a .txt file")
    args = parser.parse_args()

    try:
        tokenizer, model = load_model()
        text = read_image(args.image)
        question = f"{args.prompt}\n{text}"
        response = generate_response(question, tokenizer, model, max_tokens=args.max_tokens)

        print("\nGenerated Response:\n", response)

        if args.output:
            output_path = "response.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"\n[INFO] Response saved to {output_path}")

    except OCRProcessingError:
        print("[ERROR] Failed to read and extract text from the image.")
    except ResponseGenerationError:
        print("[ERROR] Failed to generate response from the language model.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print("[ERROR] An unexpected error occurred.")

if __name__ == "__main__":
    main()
