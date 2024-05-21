import torch

class FeaturesCache:
  def __init__(self) -> None:
    '''
    cache_map will be used as follows:
      { filename: Tensor(...) }
      
    TODO: optimize this with frequency table
    '''
    self.cache_map = dict()
    
  def append(self, filename: str, tensor: torch.Tensor):
    self.cache_map[filename] = tensor
    
  def has(self, filename: str):
    if filename in self.cache_map:
      return True
    else:
      return False
    
  def get(self, filename: str):
    return self.cache_map[filename]