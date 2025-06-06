import torch
import easyocr

def read_image(image_path):
  reader = easyocr.Reader(['pt'])
  result = reader.readtext(image_path, detail=0)  # return only text
  text = ' '.join(result)
  return text