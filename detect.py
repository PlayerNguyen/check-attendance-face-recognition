from cv2.typing import MatLike
from facenet_pytorch import MTCNN
import device

class Detect:

  def __init__(self) -> None:
    self.detector = MTCNN(selection_method="probability", margin=32, post_process=False, device=device.get_current_device())


  def detect(self, image, return_prob=False):

    if image is None:
      raise "Image cannot be none"

    return self.detector(image, return_prob=return_prob)
