import cv2
import numpy as np
from slam import VisualSLAM

def main():
    # Camera matrix (example values - replace with your camera's calibration)
    camera_matrix = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])

    # Initialize SLAM system
    slam = VisualSLAM(r'G:\Nihar\imew\main\GRU\config\config.yaml', camera_matrix)
    # slam.tracker.camera_matrix = camera_matrix  # Set camera matrix

    # Use webcam or video file
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or file path for video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        slam.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()