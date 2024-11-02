# src/feature/detector.py
import cv2
import numpy as np
import yaml

import cv2
import numpy as np

class FeatureDetector:
    def __init__(self, config):
        self.config = config
        # Optimize ORB parameters for better speed/accuracy balance
        self.orb = cv2.ORB_create(
            nfeatures=config.get('num_features', 2000),      # Reduced features for speed
            scaleFactor=config.get('scale_factor', 1.2),
            nlevels=config.get('num_levels', 8),            
            edgeThreshold=config.get('edge_threshold', 19),  # Adjusted for better corners
            firstLevel=config.get('first_level', 0),
            WTA_K=config.get('wta_k', 2),
            patchSize=config.get('patch_size', 31),
            fastThreshold=config.get('fast_threshold', 15)   # Lower for more features
        )
        
        # Initialize CLAHE for better feature detection
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
    
    def detect_and_compute(self, image):
        """Detect keypoints and compute descriptors with improved visualization"""
        if image is None:
            return None, None
            
        # Make a copy of the input image for visualization
        vis_image = image.copy()
            
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Enhance image
        # 1. Apply CLAHE for better contrast
        gray = self.clahe.apply(gray)
        
        # 2. Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0.7)
        
        # Detect and compute
        mask = self.create_grid_mask(gray.shape[:2])
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        
        # Visualize features
        if self.config.get('show_keypoints', True):
            # Draw keypoints
            vis_image = cv2.drawKeypoints(
                vis_image, 
                keypoints, 
                None,
                color=(0, 255, 0),  # Green color
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            
            # Add keypoint count
            text = f'Keypoints: {len(keypoints)}'
            cv2.putText(
                vis_image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show visualization
            cv2.imshow('Current Features', vis_image)
        
        return keypoints, descriptors
        
    def create_grid_mask(self, shape):
        """Create a grid mask to ensure even feature distribution"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        grid_size = 32  # Size of grid cells
        
        for y in range(0, height - grid_size, grid_size):
            for x in range(0, width - grid_size, grid_size):
                mask[y:y+grid_size, x:x+grid_size] = 255
                
        return mask

class FeatureMatcher:
    def __init__(self, config):
        self.config = config
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    def match_features(self, desc1, desc2, img1, img2, kpts1, kpts2):
        """Match features with improved visualization"""
        if desc1 is None or desc2 is None or img1 is None or img2 is None:
            return []
            
        try:
            # Ensure images are in color for visualization
            if len(img1.shape) == 2:
                img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                img1_color = img1.copy()
                
            if len(img2.shape) == 2:
                img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            else:
                img2_color = img2.copy()
            
            # Match features
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # Further filter matches using geometric constraints
            if len(good_matches) >= 8:
                src_pts = np.float32([kpts1[m.queryIdx].pt for m in good_matches])
                dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good_matches])
                
                # Use RANSAC to filter matches
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                
                if H is not None and mask is not None:
                    # Only keep inlier matches
                    good_matches = [m for i, m in enumerate(good_matches) if mask[i][0]]
            
            # Create match visualization
            if self.config.get('show_matches', True) and len(good_matches) > 0:
                # Draw matches
                match_img = cv2.drawMatches(
                    img1_color, kpts1,
                    img2_color, kpts2,
                    good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    matchColor=(0, 255, 0),      # Green color for matches
                    singlePointColor=(255, 0, 0)  # Red color for keypoints
                )
                
                # Add text with match count
                text = f'Matches: {len(good_matches)}'
                cv2.putText(
                    match_img, text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2
                )
                
                # Display match visualization
                cv2.imshow('Matches', match_img)
            
            return good_matches
            
        except Exception as e:
            print(f"Error in matching: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def visualize_matches(self, desc1, desc2, matches):
        """Placeholder for match visualization"""
        pass  # Implement if needed

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
        if len(matches) < 5:
            return self.last_pose
            
        # Extract matched points and ensure they're properly shaped
        pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        
        # Ensure points are contiguous in memory
        pts1 = np.ascontiguousarray(pts1)
        pts2 = np.ascontiguousarray(pts2)
        
        try:
            # Calculate essential matrix with error handling
            E, mask = cv2.findEssentialMat(
                pts1, pts2, 
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None:
                return self.last_pose
                
            # Recover pose with error handling
            retval, R, t, mask = cv2.recoverPose(
                E, pts1, pts2, 
                self.camera_matrix, 
                mask=mask
            )
            
            if not retval or R is None or t is None:
                return self.last_pose
                
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.reshape(-1)
            
            # Update pose
            self.last_pose = T @ self.last_pose
            
            return self.last_pose
            
        except cv2.error as e:
            print(f"OpenCV error during pose estimation: {e}")
            return self.last_pose
        except Exception as e:
            print(f"Error during pose estimation: {e}")
            return self.last_pose

# src/visualization/viewer.py
import open3d as o3d
import numpy as np

import open3d as o3d
import numpy as np

class MapVisualizer:
    def __init__(self):
        print("\n=== Open3D Visualization Controls ===")
        print("Left mouse button: Rotate view")
        print("Right mouse button: Pan/move view")
        print("Mouse wheel: Zoom in/out")
        print("Ctrl + Left mouse: Pan/move view")
        print("Shift + Left mouse: Rotate around model")
        print("\nKeyboard Controls:")
        print("R: Reset view")
        print("T: Top view")
        print("S: Side view")
        print("F: Front view")
        print("Q: Close window")
        
        # Initialize visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("3D Map", width=1024, height=768)
        
        # Set rendering options for better visibility
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Black background
        opt.point_size = 5.0  # Larger points
        opt.line_width = 2.0  # Thicker lines
        
        # Initialize geometries
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # Add coordinate frame - smaller size for laptop camera scale
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # Smaller coordinate frame
            origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.coord_frame)
        
        # Camera trajectory as line set
        self.camera_trajectory = o3d.geometry.LineSet()
        self.vis.add_geometry(self.camera_trajectory)
        
        # Initialize camera view
        self.setup_camera()
        
    def setup_camera(self):
        """Setup default camera view"""
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.3)
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_lookat([0.0, 0.0, 0.0])
        ctr.set_up([0.0, -1.0, 0.0])
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def set_view_angle(self, front, lookat=None, up=None, zoom=None):
        """Set specific view angle"""
        ctr = self.vis.get_view_control()
        if lookat is None:
            lookat = [0, 0, 0]
        if up is None:
            up = [0, -1, 0]
        
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        if zoom is not None:
            ctr.set_zoom(zoom)
            
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def top_view(self):
        """Switch to top view"""
        self.set_view_angle([0, -1, 0], up=[0, 0, -1])
        
    def side_view(self):
        """Switch to side view"""
        self.set_view_angle([1, 0, 0])
        
    def front_view(self):
        """Switch to front view"""
        self.set_view_angle([0, 0, -1])
        
    def reset_view(self):
        """Reset camera view for laptop camera setup"""
        self.setup_camera()
        
    def create_camera_frustum(self, pose, size=0.1):
        """Create a camera frustum at the given pose"""
        points = np.array([
            [0, 0, 0],  # Camera center
            [-size, -size, size*2],  # Bottom-left
            [size, -size, size*2],   # Bottom-right
            [size, size, size*2],    # Top-right
            [-size, size, size*2],   # Top-left
        ])
        
        lines = np.array([
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # Lines between corners
        ])
        
        # Create line set
        frustum = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        
        # Transform to pose
        frustum.transform(pose)
        
        return frustum
        
    def update_map(self, points, poses):
        """Update the 3D visualization with scaled parameters for laptop camera"""
        if len(points) == 0:
            return
            
        # Scale points for better visibility
        if isinstance(points, list):
            points = np.array(points)
        
        # Scale the points to be more visible
        points = points * 0.1  # Scale down points for better visibility
        
        # Filter points
        valid_mask = np.all(np.abs(points) < 1.0, axis=1) & (points[:, 2] > 0)
        filtered_points = points[valid_mask]
        
        if len(filtered_points) == 0:
            return
            
        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # Color points based on depth for better visualization
        colors = np.zeros((len(filtered_points), 3))
        normalized_depths = (filtered_points[:, 2] - filtered_points[:, 2].min()) / \
                          (filtered_points[:, 2].max() - filtered_points[:, 2].min() + 1e-6)
        colors[:, 1] = 0.2 + 0.8 * normalized_depths  # Green channel varies with depth
        colors[:, 2] = 0.2  # Add some blue for better visibility
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Update camera trajectory
        if len(poses) >= 2:
            # Scale poses for better visibility
            scaled_poses = [pose.copy() for pose in poses]
            for pose in scaled_poses:
                pose[:3, 3] *= 0.1  # Scale translation
            
            points = [pose[:3, 3] for pose in scaled_poses]
            lines = [[i, i+1] for i in range(len(poses)-1)]
            
            self.camera_trajectory.points = o3d.utility.Vector3dVector(points)
            self.camera_trajectory.lines = o3d.utility.Vector2iVector(lines)
            self.camera_trajectory.colors = o3d.utility.Vector3dVector(
                [[1.0, 0.0, 0.0] for _ in lines]  # Red trajectory
            )
        
        # Update visualization
        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.camera_trajectory)
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def close(self):
        """Close the visualization window"""
        self.vis.destroy_window()
        
    def save_view(self, filename="reconstruction.png"):
        """Save current view to an image file"""
        self.vis.capture_screen_image(filename, False)

# slam.py - Update the VisualSLAM class
import cv2
import numpy as np
import yaml
from typing import List, Tuple

class VisualSLAM:
    def __init__(self, config_path, camera_matrix):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.detector = FeatureDetector(self.config['feature_detector'])
        self.matcher = FeatureMatcher(self.config['feature_matcher'])
        self.tracker = PoseTracker(self.config['tracker'], camera_matrix)
        self.visualizer = MapVisualizer()
        
        # Initialize map storage
        self.keyframes = []  # [(frame, keypoints, descriptors, pose), ...]
        self.map_points = []
        self.camera_poses = [np.eye(4)]  # Start with identity pose
        self.camera_matrix = camera_matrix

        self.frame_count = 0  # Add frame counter
        
    class VisualSLAM:
        def triangulate_points(self, kpts1, kpts2, pose1, pose2, matches):
            """Triangulate 3D points optimized for laptop camera"""
            print("\n=== Triangulation Process ===")
            
            try:
                if len(matches) < 8:
                    print("Not enough matches for triangulation")
                    return np.array([])
                    
                # Extract matched points
                pts1 = np.float32([kpts1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kpts2[m.trainIdx].pt for m in matches])
                
                print(f"Processing {len(matches)} matched points")
                
                # Use Essential Matrix with smaller threshold for laptop camera
                E, inlier_mask = cv2.findEssentialMat(
                    pts1, pts2,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=0.5  # Reduced threshold for smaller movements
                )
                
                if inlier_mask is None:
                    print("No inliers found in Essential Matrix estimation")
                    return np.array([])
                    
                inliers = inlier_mask.ravel().astype(bool)
                pts1 = pts1[inliers]
                pts2 = pts2[inliers]
                
                print(f"Filtered to {len(pts1)} inlier points")
                
                # Extract projection matrices
                P1 = self.camera_matrix @ np.hstack((pose1[:3, :3], pose1[:3, 3:4]))
                P2 = self.camera_matrix @ np.hstack((pose2[:3, :3], pose2[:3, 3:4]))
                
                # Normalize points
                pts1_normalized = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.camera_matrix, None)
                pts2_normalized = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.camera_matrix, None)
                
                # Triangulate
                points_4d = cv2.triangulatePoints(
                    P1, P2,
                    pts1_normalized.reshape(-1, 2).T,
                    pts2_normalized.reshape(-1, 2).T
                )
                
                # Convert to 3D
                points_3d = (points_4d[:3, :] / points_4d[3, :]).T
                
                # Filter points - adjusted for laptop camera
                valid_mask = (
                    (np.abs(points_3d[:, 0]) < 2.0) &    # X bounds
                    (np.abs(points_3d[:, 1]) < 2.0) &    # Y bounds
                    (points_3d[:, 2] > 0.1) &            # Min depth
                    (points_3d[:, 2] < 3.0)              # Max depth
                )
                
                filtered_points = points_3d[valid_mask]
                
                print(f"Points after filtering: {len(filtered_points)}")
                
                return filtered_points
                
            except Exception as e:
                print(f"Error during triangulation: {e}")
                import traceback
                traceback.print_exc()
                return np.array([])
            
    def update_map(self, frame, pose, kpts, descs, matches):
        """Update the map with new keyframe and points"""
        print("\n=== Map Update Status ===")
        print(f"Number of points in map: {len(self.map_points)}")
        print(f"Number of keyframes: {len(self.keyframes)}")
        
        # Only create new keyframe periodically
        if self.frame_count % self.config.get('keyframe_interval', 3) == 0:
            print("\n=== Creating New Keyframe ===")
            
            try:
                # Get last keyframe
                last_keyframe = self.keyframes[-1]
                
                # Triangulate points between frames
                if len(matches) >= 8:
                    # Extract matched points
                    pts1 = np.float32([kpts[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([last_keyframe['keypoints'][m.trainIdx].pt for m in matches])
                    
                    # Create projection matrices
                    P1 = self.camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = self.camera_matrix @ np.hstack((pose[:3, :3], pose[:3, 3:4]))
                    
                    # Triangulate points
                    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
                    points_3d = (points_4d[:3, :] / points_4d[3, :]).T
                    
                    # Filter points
                    valid_mask = (
                        (np.abs(points_3d[:, 0]) < 2.0) &
                        (np.abs(points_3d[:, 1]) < 2.0) &
                        (points_3d[:, 2] > 0.1) &
                        (points_3d[:, 2] < 3.0)
                    )
                    
                    valid_points = points_3d[valid_mask]
                    
                    if len(valid_points) > 0:
                        print(f"Adding {len(valid_points)} new points to map")
                        self.map_points.extend(valid_points.tolist())
                        self.camera_poses.append(pose.copy())
                        
                        # Add new keyframe
                        self.keyframes.append({
                            'frame': frame.copy(),
                            'keypoints': kpts,
                            'descriptors': descs,
                            'pose': pose.copy()
                        })
                        
                        # Update visualization
                        points_array = np.array(self.map_points)
                        print("Point cloud bounds:")
                        print(f"X: [{points_array[:,0].min():.2f}, {points_array[:,0].max():.2f}]")
                        print(f"Y: [{points_array[:,1].min():.2f}, {points_array[:,1].max():.2f}]")
                        print(f"Z: [{points_array[:,2].min():.2f}, {points_array[:,2].max():.2f}]")
                        
                        self.visualizer.update_map(points_array, self.camera_poses)
                    else:
                        print("No valid points after filtering")
                
            except Exception as e:
                print(f"Error in map update: {e}")
                import traceback
                traceback.print_exc()
    
    def process_frame(self, frame):
        """
        Process a new frame with enhanced debugging
        """
        self.frame_count += 1
        print(f"\n=== Processing Frame {self.frame_count} ===")
        
        if frame is None:
            print("Error: Received None frame")
            return
            
        # Store a copy of the frame
        frame = frame.copy()
        
        # Detect features
        kpts, descs = self.detector.detect_and_compute(frame)
        print(f"Detected {len(kpts)} keypoints")
        
        # Draw features on frame copy
        frame_with_features = frame.copy()
        frame_with_features = cv2.drawKeypoints(
            frame_with_features, kpts, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        # Add keypoint count
        cv2.putText(
            frame_with_features,
            f"Keypoints: {len(kpts)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show features
        cv2.imshow('Current Features', frame_with_features)
        
        # First frame handling
        if len(self.keyframes) == 0:
            print("Initializing first keyframe")
            self.keyframes.append({
                'frame': frame.copy(),
                'keypoints': kpts,
                'descriptors': descs,
                'pose': np.eye(4)
            })
            return
        
        # Match with last keyframe
        last_keyframe = self.keyframes[-1]
        matches = self.matcher.match_features(
            descs, last_keyframe['descriptors'],
            frame, last_keyframe['frame'],
            kpts, last_keyframe['keypoints']
        )
        
        print(f"Found {len(matches)} matches with last keyframe")
        
        if len(matches) < self.config.get('min_matches', 8):
            print(f"Not enough matches: {len(matches)}")
            return
            
        # Estimate pose
        try:
            # Extract matched points
            pts1 = np.float32([kpts[m.queryIdx].pt for m in matches])
            pts2 = np.float32([last_keyframe['keypoints'][m.trainIdx].pt for m in matches])
            
            # Estimate Essential Matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, 
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None:
                print("Failed to estimate Essential Matrix")
                return
                
            # Recover pose from Essential Matrix
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix, mask=mask)
            
            # Create 4x4 pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.ravel()
            
            print("Estimated pose matrix:")
            print(pose)
            
            # Update map with matches
            self.update_map(frame, pose, kpts, descs, matches)
            
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            import traceback
            traceback.print_exc()