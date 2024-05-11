import os
import numpy as np
import cv2
import tensorflow as tf
import random

movements = os.listdir("tennis_dataset")

HEIGHT, WEIGHT = 180, 320

video_path = 'preparation.mp4'
cap = cv2.VideoCapture(video_path)

prep = []
classes_prep = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WEIGHT, HEIGHT))
    # frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = np.array(frame).astype(np.float32)
    prep.append(frame)
    classes_prep.append(0)

random.shuffle(prep)
frames = prep[:1500]
classes = classes_prep[:1500]
cap.release()

# selected_frames = np.empty((0, HEIGHT, WEIGHT, 3), dtype=np.float32)

counter = 1
flag = 0

for movement in movements:
    movement_path = "tennis_dataset" + "/" + movement

    videos = [f for f in os.listdir(movement_path) if f.endswith('.mp4')]
    random.shuffle(videos)
    counter = 1

    for video in videos:
      video_path = movement_path + "/" + video
      print(video_path)

      cap = cv2.VideoCapture(video_path)
      frame_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28,
                      29, 31, 32, 34, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
      for frame_number in frame_numbers:

        if len(frames) == 3000:
          flag = 1
          break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if ret:
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame = cv2.resize(frame, (WEIGHT, HEIGHT))
          # frame = tf.image.convert_image_dtype(frame, tf.float32)
          frame = np.array(frame).astype(np.float32)
          # selected_frames = np.append(selected_frames, np.array([frame]), axis=0)
          frames.append(frame)
          classes.append(1)
        else:
          print(f"Frame {frame_number} not found in {video_path}")

      cap.release()

      if flag == 1:
        break

      if counter == 7:
        break

      counter += 1

    if flag == 1:
      break

frames_array = np.array(frames)
classes_array = np.array(classes)

print(frames_array.shape, classes_array.shape)
print(frames_array[7], classes_array[7], classes_array[1777])


np.save('preparation_frames.npy', frames_array)
np.save('preparation_classes.npy', classes_array)