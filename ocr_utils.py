# ocr_utils.py
import easyocr
import cv2
import logging
from exceptions import OCRProcessingError

logging.basicConfig(level=logging.INFO)

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        logging.error(f"Failed to preprocess image: {e}")
        raise OCRProcessingError from e

def read_image(image_path):
    try:
        processed = preprocess_image(image_path)
        reader = easyocr.Reader(['pt'])
        result = reader.readtext(processed, detail=0)
        text = ' '.join(result)
        logging.info("Image read and text extracted successfully.")
        return text
    except Exception as e:
        logging.error(f"Error during OCR: {e}")
        raise OCRProcessingError from e
