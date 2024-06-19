import os
import cv2
from cv2.typing import MatLike
import torch
import argparse
from detect import Detect
import recognition
from PIL import Image
import numpy as np
from transfomer import Transformer
import pandas as pd
import storage
import cache
import drawer


def load_datasets():
  dirname = os.path.join("datasets")
  if os.path.exists(dirname) is False:
    raise "Dataset has not been initialized"

  # Load all datasets
  labels = os.listdir(dirname)
  return labels

def estimate_embeddings_to_datasets(labels: list[str],
      embeddings: torch.Tensor,
      recognizer: recognition.Recognizer,
      detector: Detect,
      transformer: Transformer,
      features_cache: cache.FeaturesCache,
      threshold: float = 0.6
):

  cosine_similar = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  collected_distances = []
  collected_labels = []

  for label in labels:
    distances = []
    dirname = os.path.join("datasets", label)
    files = os.listdir(dirname)

    for image in files:

      filename = os.path.join("datasets", label, image)
      current_embeddings = None

      if features_cache.has(filename) is False:
        print(filename)
        current_image = Image.open(filename)
        current_image = detector.detect(current_image) / 255
        current_embeddings = recognizer.recognize(current_image.unsqueeze(0))
        features_cache.append(filename, current_embeddings)
      else :
        current_embeddings = features_cache.get(filename)


      current_distance = cosine_similar(current_embeddings, embeddings)
      distances.append(current_distance.item())

    # Calculate average
    mean_distance = np.average(distances)

    if mean_distance >= threshold:
      collected_distances.append(mean_distance)
      collected_labels.append(label)

  return collected_distances, collected_labels

def print_distance_dict(prediction_result):
  data_frame = pd.DataFrame({"Name": prediction_result[1], "Cos. Distance": prediction_result[0]})
  print(data_frame)
  print()

def main(capture_device: int = 0, detect_rate: int = 20):
  labels = load_datasets()

  detector = Detect()
  transformer = Transformer()
  recognizer = recognition.Recognizer()
  frame_count = 0
  capture = cv2.VideoCapture(capture_device)

  # Features cache
  features_cache = cache.FeaturesCache()
  last_text = "None";
  last_color = (255, 255, 255)
  while True:
    _, image = capture.read()
    frame_count = frame_count + 1

    drawer.draw_bottom_indicator(image=image, text=last_text, capturer=capture, left_offset=64, color=last_color)
    cv2.imshow("Preview", image)

    if frame_count >= detect_rate:
      process_image = transformer.apply_from_cv2_video_capture(image, resize=(200, 200))
      face_image = detector.detect(image=process_image, return_prob=False)

      if face_image is not None:
        face_image = transformer.apply_after_face_detection(face_image)
        embeddings = recognizer.recognize(face_image.unsqueeze(0))

        prediction_result = estimate_embeddings_to_datasets(
          recognizer=recognizer,
          labels=labels, embeddings=embeddings,
          detector=detector,
          features_cache=features_cache,
          transformer=transformer,
          threshold=0.65
          )
        print_distance_dict(prediction_result=prediction_result)

        # If we found any face, search for the face with highest distance (since we use cosine)
        if len (prediction_result[0] ) != 0:
          max_index = np.argmax(prediction_result[0])

          distance = prediction_result[0][max_index]
          label_name = prediction_result[1][max_index]
          print("Found a person with name %s" % (label_name))

          last_text = label_name + " " + str("{0:.2f}").format(distance)
          last_color = (0, 255, 0)
          # Record a candidate with a time stamp
          storage.check_attendance(label_name)
        else:
          print("Face is not match with any face in dataset.")
          last_color = (0, 0, 255)
          # add more logic when not found any
          # draw a bottom indicator
          last_text = "Stranger"

      else:
          last_text = "Not found any face"
          last_color = (255, 255, 255)
      frame_count = 0
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  capture.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("predict.py")

  parser.add_argument("--capture-device", help="select a capture device", type=int)
  parser.add_argument("--detect-rate", help="a rate of detect (using per frame)", type=int)
  args = parser.parse_args()

  detect_rate = 20 if args.detect_rate is None else args.detect_rate
  capture_device = 0 if args.capture_device is None else args.capture_device

  # Initialize storage
  storage.initialize_storage()

  # Initialize main capture device
  main(capture_device=capture_device, detect_rate=detect_rate)
