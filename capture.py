import cv2
import argparse
from detect import Detect
from transfomer import Transformer
import os
import drawer

def ensure_dir(username: str):
  dirpath = os.path.join("datasets", username)
  os.makedirs(dirpath, exist_ok=True)

def start_capture(name: str, capture_device: int = 0, preview_windows: bool = True, detect_rate = 15):

  ensure_dir(name)

  detector = Detect()
  transformer = Transformer()
  capture = cv2.VideoCapture(capture_device)

  frame_count = 0
  interval = 0
  last_text = "Please look into a device."
  last_color = (255, 0, 0)

  while True:
    ret, image = capture.read()
    frame_count = frame_count + 1

    drawer.draw_bottom_indicator(image=image, text=last_text, capturer=capture, left_offset=64, color=last_color)

    # Apply detector
    if frame_count >= detect_rate:
      print("Refresh frame rate cycle")
      frame_count = 0
      _cropped_image = transformer.apply_from_cv2_video_capture(image, resize=(480, 480))
      face_image, probability = detector.detect(_cropped_image, return_prob=True)

      if face_image is not None:
        print("MTCNN probability: %f" % (probability))
        print("Detected a face, validation check for mapping")

        if probability < 0.99:
            # todo
            last_text = "Please look straight ahead."
            last_color = (0, 255, 255)
        else:
            face_image = transformer.apply_after_face_detection(face_image, to_pil=True)
            last_color = (0, 255, 0)
            last_text = "Hold still for capturing."

            # Save face data
            filename = os.path.join("datasets", name, "{}_{}.jpg".format(name, interval))
            face_image.save(filename)

            interval = interval + 1
            if interval >= 10:
              break

      else:
        last_text = "Please look into a device."
        last_color = (0, 0, 255)
    if preview_windows is True:
      cv2.imshow('Preview', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


  capture.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("capture.py")
  parser.add_argument("name", type=str, help="a student name")
  parser.add_argument("--capture-device", help="select a capture device", type=int)
  parser.add_argument("--no-gui", help="disable gui mode", action="store_true")
  parser.add_argument("--detect-rate", help="a rate of detect (using per frame)", type=int)
  args = parser.parse_args()

  capture_device = 0 if args.capture_device is None else args.capture_device
  open_preview_windows = False if args.no_gui is None else True
  detect_rate = 10 if args.detect_rate is None else args.detect_rate
  start_capture(name=args.name, capture_device=capture_device, preview_windows=open_preview_windows, detect_rate=detect_rate)
