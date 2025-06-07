# config.py
LOCAL_MODEL_PATH = "gemma-2b-it"

# exceptions.py
class OCRProcessingError(Exception):
    """Raised when the OCR process fails."""
    pass

class ResponseGenerationError(Exception):
    """Raised when the LLM response generation fails."""
    pass
  
