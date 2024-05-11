
# Code below based on https://www.kaggle.com/code/ibrahimserouis99/human-pose-estimation-with-movenet

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def loop(frame, keypoints, threshold=0.11):
    """
    Main loop : Draws the keypoints and edges for each instance
    """
    # Loop through the results
    for instance in keypoints:
        # Draw the keypoints and get the denormalized coordinates
        denormalized_coordinates = draw_keypoints(frame, instance, threshold)
        # Draw the edges
        draw_edges(denormalized_coordinates, frame, EDGE_COLORS, threshold)


def draw_keypoints(frame, keypoints, threshold=0.11):
    """Draws the keypoints on a image frame"""

    # Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
    denormalized_coordinates = np.squeeze(np.multiply(keypoints, [WIDTH, HEIGHT, 1]))
    # Iterate through the points
    for keypoint in denormalized_coordinates:
        # Unpack the keypoint values : y, x, confidence score
        keypoint_y, keypoint_x, keypoint_confidence = keypoint
        if keypoint_confidence > threshold:
            """"
            Draw the circle
            Note : A thickness of -1 px will fill the circle shape by the specified color.
            """
            cv2.circle(
                img=frame,
                center=(int(keypoint_x), int(keypoint_y)),
                radius=1,
                color=(255, 255, 255),
                thickness=-1
            )
    return denormalized_coordinates


def draw_edges(denormalized_coordinates, frame, edges_colors, threshold=0.11):
    """
    Draws the edges on a image frame
    """

    # Iterate through the edges
    for edge, color in edges_colors.items():
        # Get the dict value associated to the actual edge
        p1, p2 = edge
        # Get the points
        y1, x1, confidence_1 = denormalized_coordinates[p1]
        y2, x2, confidence_2 = denormalized_coordinates[p2]
        # Draw the line from point 1 to point 2, the confidence > threshold
        if (confidence_1 > threshold) & (confidence_2 > threshold):
            cv2.line(
                img=frame,
                pt1=(int(x1), int(y1)),
                pt2=(int(x2), int(y2)),
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA  # Gives anti-aliased (smoothed) line which looks great for curves
            )


def run_inference(height, weight, frame):
    """
    Runs inferences then starts the main loop for each frame
    """
    initial_shape = weight, height

    image = frame.copy()
    image = cv2.resize(image, (WIDTH, HEIGHT))
    input_image = tf.cast(tf.image.resize_with_pad(image, WIDTH, HEIGHT), dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    results = movenet(input_image)
    """
    Output shape :  [1, 6, 56] ---> (batch size), (instances), (xy keypoints coordinates and score from [0:50] 
    and [ymin, xmin, ymax, xmax, score] for the remaining elements)
    First, let's resize it to a more convenient shape, following this logic : 
    - First channel ---> each instance
    - Second channel ---> 17 keypoints for each instance
    - The 51st values of the last channel ----> the confidence score.
    Thus, the Tensor is reshaped without losing important information. 
    """

    keypoints = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))

    loop(image, keypoints, threshold=0.11)

    frame_rgb = cv2.cvtColor(
        cv2.resize(
            image, (initial_shape[0], initial_shape[1]),
            interpolation=cv2.INTER_LANCZOS4
        ),
        cv2.COLOR_BGR2RGB
    )
    # frame_rgb = cv2.resize(image, (initial_shape[0], initial_shape[1]), interpolation=cv2.INTER_LANCZOS4)

    return frame_rgb

cyan = (255, 255, 0)
magenta = (255, 0, 255)
yellow = (0, 255, 255)

WIDTH = HEIGHT = 256

EDGE_COLORS = {
    (0, 1): magenta,
    (0, 2): cyan,
    (1, 3): magenta,
    (2, 4): cyan,
    (0, 5): magenta,
    (0, 6): cyan,
    (5, 7): magenta,
    (7, 9): magenta,
    (6, 8): cyan,
    (8, 10): cyan,
    (5, 6): yellow,
    (5, 11): magenta,
    (6, 12): cyan,
    (11, 12): yellow,
    (11, 13): magenta,
    (13, 15): magenta,
    (12, 14): cyan,
    (14, 16): cyan
}

model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
movenet = model.signatures["serving_default"]