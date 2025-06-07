# tests/test_integration.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from llm_handler import load_model, generate_response

def test_full_pipeline():
    tokenizer, model = load_model()
    text = "This is a test text extracted from the image."
    question = f"Summarize this text: {text}"
    response = generate_response(question, tokenizer, model)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
