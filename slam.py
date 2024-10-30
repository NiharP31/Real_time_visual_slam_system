# src/feature/detector.py
import cv2
import numpy as np
import yaml

class FeatureDetector:
    def __init__(self, config):
        self.config = config
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=config.get('num_features', 1000),
            scaleFactor=config.get('scale_factor', 1.2),
            nlevels=config.get('num_levels', 8)
        )
        
    def detect_and_compute(self, image):
        """
        Detect keypoints and compute descriptors
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        if image is None:
            return None, None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        return keypoints, descriptors

# src/feature/matcher.py
class FeatureMatcher:
    def __init__(self, config):
        self.config = config
        # Initialize BF matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def match_features(self, desc1, desc2):
        """
        Match features between two frames
        
        Args:
            desc1, desc2 (np.ndarray): Feature descriptors
            
        Returns:
            list: Matches
        """
        if desc1 is None or desc2 is None:
            return []
            
        matches = self.matcher.match(desc1, desc2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Apply ratio test if needed
        if self.config.get('use_ratio_test', True):
            good_matches = []
            for m in matches:
                if m.distance < self.config.get('ratio_threshold', 0.75):
                    good_matches.append(m)
            return good_matches
        
        return matches

# src/tracking/tracker.py
class PoseTracker:
    def __init__(self, config, camera_matrix):
        self.config = config
        self.camera_matrix = camera_matrix
        self.last_pose = np.eye(4)
        
    def estimate_pose(self, kpts1, kpts2, matches):
        """
        Estimate camera pose from matched features
        
        Args:
            kpts1, kpts2: Keypoints from two frames
            matches: Feature matches
            
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        # Extract matched points
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
        
        # Calculate essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(-1)
        
        # Update pose
        self.last_pose = T @ self.last_pose
        
        return self.last_pose

# src/visualization/viewer.py
import open3d as o3d

class MapVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
    def update_map(self, points, poses):
        """
        Update the 3D visualization
        
        Args:
            points (np.ndarray): 3D points in the map
            poses (list): Camera poses
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add camera poses
        for pose in poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1,
                origin=[0, 0, 0]
            )
            frame.transform(pose)
            self.vis.add_geometry(frame)
        
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

# slam.py - Update the VisualSLAM class
class VisualSLAM:
    def __init__(self, config_path, camera_matrix):  # Add camera_matrix parameter
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.detector = FeatureDetector(self.config['feature_detector'])
        self.matcher = FeatureMatcher(self.config['feature_matcher'])
        self.tracker = PoseTracker(self.config['tracker'], camera_matrix)  # Pass camera_matrix
        self.visualizer = MapVisualizer()
        
        # Initialize map storage
        self.keyframes = []
        self.map_points = []
        
    def process_frame(self, frame):
        """
        Process a new frame
        
        Args:
            frame (np.ndarray): Input image
        """
        # Detect features
        kpts, descs = self.detector.detect_and_compute(frame)
        
        if len(self.keyframes) == 0:
            # Initialize system with first frame
            self.keyframes.append((frame, kpts, descs))
            return
            
        # Match with last keyframe
        last_kpts = self.keyframes[-1][1]
        last_descs = self.keyframes[-1][2]
        matches = self.matcher.match_features(descs, last_descs)
        
        # Estimate pose
        pose = self.tracker.estimate_pose(kpts, last_kpts, matches)
        
        # Update map
        self.update_map(frame, pose, kpts, matches)
        
        # Visualize
        self.visualizer.update_map(self.map_points, self.keyframes)
