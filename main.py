import cv2
import numpy as np
from slam import VisualSLAM

def list_available_cameras():
    """List all available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def select_camera():
    """Let user select a camera"""
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("No cameras found!")
        return None
        
    print("\n=== Available Cameras ===")
    print("0: Internal Laptop Camera (usually)")
    print("1: First External Camera")
    print("2: Second External Camera (if available)")
    print(f"\nDetected cameras at indexes: {available_cameras}")
    
    while True:
        try:
            idx = int(input("\nSelect camera index (or -1 to exit): "))
            if idx == -1:
                return None
            if idx in available_cameras:
                return idx
            print("Invalid camera index! Please try again.")
        except ValueError:
            print("Please enter a valid number!")

def init_camera(camera_idx):
    """Initialize camera with given index"""
    cap = cv2.VideoCapture(camera_idx)
    
    # Try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nCamera initialized with:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    return cap

def main():
    try:
        print("\n=== Real-time Visual SLAM System ===")
        
        # Camera selection
        camera_idx = select_camera()
        if camera_idx is None:
            print("Camera selection cancelled.")
            return
            
        # Initialize camera
        cap = init_camera(camera_idx)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_idx}")
            return
            
        # Camera matrix (example values - replace with your camera's calibration)
        # Note: You might need different matrices for different cameras
        camera_matrix = np.array([
            [718.856, 0, 607.1928],
            [0, 718.856, 185.2157],
            [0, 0, 1]
        ])

        # Initialize SLAM system
        slam = VisualSLAM(r'C:\Users\nihar\Documents\github\Real_time_visual_slam_system\config\config.yaml', camera_matrix)

        # Create windows once, outside the loop
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Current Features', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        
        # Resize windows to reasonable dimensions
        cv2.resizeWindow('Frame', 640, 480)
        cv2.resizeWindow('Current Features', 640, 480)
        cv2.resizeWindow('Matches', 1280, 480)

        print("\n=== Controls ===")
        print("Q/ESC: Quit application")
        print("R: Reset 3D view")
        print("T: Top view")
        print("S: Side view")
        print("F: Front view")
        print("P: Save current view as PNG")

        frame_count = 0
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            frame_count += 1
            
            # Process frame
            slam.process_frame(frame)
            
            # Display the original frame
            cv2.imshow('Frame', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Check for exit commands
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Exiting...")
                break
                
            # Handle 3D visualization controls
            elif key == ord('r'):
                print("Resetting view...")
                slam.visualizer.reset_view()
            elif key == ord('t'):
                print("Switching to top view...")
                slam.visualizer.top_view()
            elif key == ord('s'):
                print("Switching to side view...")
                slam.visualizer.side_view()
            elif key == ord('f'):
                print("Switching to front view...")
                slam.visualizer.front_view()
            elif key == ord('p'):
                filename = f"reconstruction_{frame_count}.png"
                print(f"Saving view as {filename}...")
                slam.visualizer.save_view(filename)
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nCleaning up...")
        # Cleanup
        if 'cap' in locals():
            cap.release()
        if 'slam' in locals():
            slam.visualizer.close()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()