import os
import numpy as np
import cv2
import tensorflow as tf
import random

movements = os.listdir("tennis_dataset")
movement_mapping = {'backhand': 0, 'forehand': 1, 'serve': 2, 'b_slice': 3,
                    'b_volley': 4, 'f_volley': 5, 'smash': 6}

HEIGHT, WEIGHT = 180, 320

train_frames = np.empty((0, 8, HEIGHT, WEIGHT, 3), dtype=np.float32)
train_classes = np.empty(0, dtype=np.int32)

test_frames = np.empty((0, 8, HEIGHT, WEIGHT, 3), dtype=np.float32)
test_classes = np.empty(0, dtype=np.int32)

val_frames = np.empty((0, 8, HEIGHT, WEIGHT, 3), dtype=np.float32)
val_classes = np.empty(0, dtype=np.int32)

for movement in movements:
    movement_path = "tennis_dataset" + "/" + movement

    videos = [f for f in os.listdir(movement_path) if f.endswith('.mp4')]
    random.shuffle(videos)
    counter = 1

    for video in videos:
      video_path = movement_path + "/" + video
      print(video_path)

      cap = cv2.VideoCapture(video_path)

      selected_frames = np.empty((0, HEIGHT, WEIGHT, 3), dtype=np.float32)
      frame_numbers = [2, 10, 18, 26, 34, 42, 50, 58]
      # frame_numbers = [2, 6, 10, 14, 17, 20, 23, 26, 29, 32, 35, 38, 42, 46, 50, 54]
      #   frame_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28,
      #                    29, 31, 32, 34, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
      for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if ret:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = cv2.resize(frame, (WEIGHT, HEIGHT))
          frame = tf.image.convert_image_dtype(frame, tf.float32)
          selected_frames = np.append(selected_frames, np.array([frame]), axis=0)
        else:
          print(f"Frame {frame_number} not found in {video_path}")

      cap.release()

      if counter <= 30:
        train_frames = np.append(train_frames, [selected_frames], axis=0)
        train_classes = np.append(train_classes, movement_mapping.get(movement))
      elif counter > 43:
        test_frames = np.append(test_frames, [selected_frames], axis=0)
        test_classes = np.append(test_classes, movement_mapping.get(movement))
      else:
        val_frames = np.append(val_frames, [selected_frames], axis=0)
        val_classes = np.append(val_classes, movement_mapping.get(movement))

      counter += 1

np.save('rgb8_train_frames.npy', train_frames)
np.save('rgb8_train_classes.npy', train_classes)

np.save('rgb8_test_frames.npy', test_frames)
np.save('rgb8_test_classes.npy', test_classes)

np.save('rgb8_val_frames.npy', val_frames)
np.save('rgb8_val_classes.npy', val_classes)