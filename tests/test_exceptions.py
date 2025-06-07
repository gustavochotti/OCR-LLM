# tests/test_exceptions.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from exceptions import OCRProcessingError, ResponseGenerationError
from ocr_utils import read_image
from llm_handler import generate_response

def test_ocr_exception(monkeypatch):
    def mock_readtext(*args, **kwargs):
        raise RuntimeError("OCR error")
    monkeypatch.setattr("easyocr.Reader.readtext", mock_readtext)
    with pytest.raises(OCRProcessingError):
        read_image("fake_path.png")

def test_response_generation_exception():
    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Tokenization error")

    class DummyModel:
        device = "cpu"

    with pytest.raises(ResponseGenerationError):
        generate_response("test", DummyTokenizer(), DummyModel())
