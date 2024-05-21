import cv2
import torch
import torchvision

class Transformer:
  def __init__(self):
     self.brightness_dense = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: x + 0.2),
        torchvision.transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
      ])
     self.to_pil = torchvision.transforms.ToPILImage()
  
  def apply_from_cv2_video_capture(self, image: tuple, resize: tuple[int] = (300, 300)):
    # Since the cv2 apply bgr, we flip it back 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize)
    return image
    
  def apply_after_face_detection(self, image: torch.Tensor, to_pil: bool = False):
    # Normalize from 0...255 to 0...1
    image = image / 255
    image = self.brightness_dense(image)
    if to_pil:
      image = self.to_pil(image)
    return image
    