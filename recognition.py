from facenet_pytorch import InceptionResnetV1
import device

class Recognizer:
  def __init__(self) -> None:  
    self.net = InceptionResnetV1(pretrained="vggface2", device=device.get_current_device()).eval()
  
  def recognize(self, image):
    return self.net(image)