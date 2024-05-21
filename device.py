import torch

def get_current_device():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  if device.type == "cuda":
    print("CUDA is available. Using GPU.")
    
  return device