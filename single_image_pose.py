import argparse
import pathlib
import time

import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose

def get_parser():
    parser = argparse.ArgumentParser("Compute pose for a single image offline")
    parser.add_argument("--inputs", "-i", help="input image", type=pathlib.Path)
    parser.add_argument("--out", "-o", type=pathlib.Path, default=None)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    with mp_pose.Pose(model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        image = cv2.cvtColor(cv2.imread(args.inputs), cv2.COLOR_BGR2RGB)
        t = time.time()
        results = pose.process(image)
        t1 = time.time()
        print(f"Time : {(t1-t)*1000:2.2f}ms")
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        out = args.inputs.parent / f"{args.inputs.stem}_landmark.jpg" if args.out is None else args.out
        print(out)
        cv2.imwrite(out, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


