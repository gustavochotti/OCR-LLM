# test_integration.py
import pytest
from llm_handler import load_model, generate_response

def test_full_pipeline():
    tokenizer, model = load_model()
    text = "Este é um texto de teste extraído da imagem."
    question = f"Resuma este texto:\n{text}"
    response = generate_response(question, tokenizer, model)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
