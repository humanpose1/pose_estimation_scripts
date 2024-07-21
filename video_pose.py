import argparse
import pathlib
import time

import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

def get_parser():
    parser = argparse.ArgumentParser("Compute pose for a video offline")
    parser.add_argument("--inputs", "-i", help="input video", type=pathlib.Path)
    parser.add_argument("--out", "-o", type=pathlib.Path, default=None)
    return parser

SIZE = (1920, 1080)

if __name__ == "__main__":
    args = get_parser().parse_args()
    cap = cv2.VideoCapture(args.inputs)

    fourcc = cv2.VideoWriter_fourcc(*'MV4V')
    out = args.inputs.parent / f"{args.inputs.stem}_landmark.mp4" if args.out is None else args.out
    out = cv2.VideoWriter(out, fourcc, 20.0, SIZE)
    with mp_pose.Pose(model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = time.time()
            results = pose.process(image)
            t1 = time.time()
            print(f"Time : {(t1-t)*1000:2.2f}ms, {image.shape=}")
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            annotated_image = cv2.resize(annotated_image, dsize=SIZE)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            out.write(annotated_image)
    out.release()
    cap.release()
        
        
        
        


