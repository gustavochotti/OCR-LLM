import torch

def get_best_dtype() -> torch.dtype:
  if not torch.cuda.is_available():
    return torch.float32
  name = torch.cuda.get_device_name(0).lower()
  if any(x in name for x in ["a100", "h100", "gaudi"]):
    return torch.bfloat16
  
  return torch.float16

def get_device() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"
