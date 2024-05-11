import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from functions import format_frames
from classes import Conv2Plus1D, ResizeVideo, ResidualMain, Project

model = load_model('rgb32_resnet3d.h5', compile=False)

video_path = 'test_forehand/forehand25.mp4'
cap = cv2.VideoCapture(video_path)

# print(f"Кількість кадрів у відео: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

selected_frames = np.empty((0, 135, 240, 3), dtype=np.float32)
frame_numbers = [2, 4, 6, 8, 10, 12, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28,
                 29, 31, 32, 34, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (240, 135))
        frame = tf.image.convert_image_dtype(frame, tf.float32)
        selected_frames = np.append(selected_frames, np.array([frame]), axis=0)
    else:
        print(f"Frame {frame_number} not found in {video_path}")

cap.release()

predictions = model.predict(np.expand_dims(selected_frames, axis=0))
predicted_class = np.argmax(predictions)

movement_mapping = {0: 'backhand', 1: 'forehand', 2: 'serve', 3: 'b_slice',
                    4: 'b_volley', 5: 'f_volley', 6: 'smash'}

print(f"Predicted class: {movement_mapping.get(predicted_class)}")