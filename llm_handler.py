# llm_handler.py
import torch
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from torch_utils import get_best_dtype, get_device
from config import LOCAL_MODEL_PATH
from exceptions import ResponseGenerationError

logging.basicConfig(level=logging.INFO)

def load_model():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=get_best_dtype(),
        device_map=get_device()
    )

    model.eval()
    return tokenizer, model

def generate_response(question: str, tokenizer, model, max_tokens: int = 768) -> str:
    try:
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            streamer=streamer
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logging.info("LLM response generated successfully.")
        return output_text

    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        raise ResponseGenerationError from e
