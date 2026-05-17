import os
import sys
import json
import struct
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow (PIL) is not installed. Please install it using: pip install Pillow")
    sys.exit(1)

import matplotlib.pyplot as plt

# Import existing tool
from annotation_tool import SimpleAnnotationTool
from point_cloud_io import PointCloudIO

class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

    def project(self, points_3d, image_width=None, image_height=None):
        """
        Project 3D points (in camera coordinate system) to 2D image coordinates.
        points_3d: (N, 3) array
        image_width, image_height: Actual dimensions of the image being processed.
                                   If provided, coordinates will be scaled if they differ from camera model dimensions.
        Returns: (N, 2) array of pixel coordinates
        """
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        elif self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params
            fx = fy = f
        else:
            raise NotImplementedError(f"Camera model {self.model} not implemented")
            
        x = points_3d[:, 0] / points_3d[:, 2]
        y = points_3d[:, 1] / points_3d[:, 2]
        u = fx * x + cx
        v = fy * y + cy
        
        # Scale coordinates if image size differs from model size
        if image_width is not None and image_height is not None:
            if image_width != self.width or image_height != self.height:
                scale_x = image_width / self.width
                scale_y = image_height / self.height
                u *= scale_x
                v *= scale_y
                
        return np.stack([u, v], axis=1)

class ImagePose:
    def __init__(self, id, qw, qx, qy, qz, tx, ty, tz, camera_id, name):
        self.id = id
        # Colmap quaternions are [w, x, y, z] -> Scipy expects [x, y, z, w]
        self.q = np.array([qx, qy, qz, qw]) 
        self.t = np.array([tx, ty, tz])
        self.camera_id = camera_id
        self.name = name
        
        # Convert quaternion to rotation matrix using scipy
        self.rot = R.from_quat(self.q)
        self.rot_mat = self.rot.as_matrix()

    def transform_world_to_camera(self, points_world):
        """
        Transform points from world coordinates to camera coordinates.
        points_world: (N, 3) array
        Returns: (N, 3) array
        """
        # P_c = R * P_w + t
        # For N points: (N, 3) @ (3, 3).T + (3,)
        return np.dot(points_world, self.rot_mat.T) + self.t

class ColmapReader:
    CAMERA_MODEL_ID_TO_NAME_AND_NUM_PARAMS = {
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
        6: ("FULL_OPENCV", 12),
        7: ("FOV", 5),
        8: ("SIMPLE_RADIAL_FISHEYE", 4),
        9: ("RADIAL_FISHEYE", 5),
        10: ("THIN_PRISM_FISHEYE", 12),
    }

    def __init__(self, sparse_dir):
        self.sparse_dir = Path(sparse_dir)
        self.cameras = {}
        self.images = {}
        self.load_cameras()
        self.load_images()

    def load_cameras(self):
        camera_txt = self.sparse_dir / "cameras.txt"
        camera_bin = self.sparse_dir / "cameras.bin"

        if camera_txt.exists():
            with open(camera_txt, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.strip().split()
                    camera_id = int(parts[0])
                    model = parts[1]
                    width = int(parts[2])
                    height = int(parts[3])
                    params = [float(p) for p in parts[4:]]
                    self.cameras[camera_id] = Camera(camera_id, model, width, height, params)
            return

        if not camera_bin.exists():
            raise FileNotFoundError(f"Neither cameras.txt nor cameras.bin found in {self.sparse_dir}")

        with open(camera_bin, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
                if model_id not in self.CAMERA_MODEL_ID_TO_NAME_AND_NUM_PARAMS:
                    raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
                model_name, num_params = self.CAMERA_MODEL_ID_TO_NAME_AND_NUM_PARAMS[model_id]
                params = list(struct.unpack("<" + "d" * num_params, f.read(8 * num_params)))
                self.cameras[camera_id] = Camera(camera_id, model_name, int(width), int(height), params)

    def load_images(self):
        image_txt = self.sparse_dir / "images.txt"
        image_bin = self.sparse_dir / "images.bin"

        if image_txt.exists():
            with open(image_txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("#") or not line:
                        i += 1
                        continue

                    parts = line.split()
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    name = parts[9]

                    self.images[image_id] = ImagePose(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)

                    # Skip the points2D line
                    i += 2
            return

        if not image_bin.exists():
            raise FileNotFoundError(f"Neither images.txt nor images.bin found in {self.sparse_dir}")

        with open(image_bin, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                image_props = struct.unpack("<idddddddi", f.read(64))
                image_id = image_props[0]
                qw, qx, qy, qz = image_props[1:5]
                tx, ty, tz = image_props[5:8]
                camera_id = image_props[8]

                name_bytes = bytearray()
                while True:
                    ch = f.read(1)
                    if ch == b"\x00":
                        break
                    if ch == b"":
                        raise ValueError("Invalid images.bin: unexpected EOF while reading image name")
                    name_bytes.extend(ch)
                name = name_bytes.decode("utf-8")

                self.images[image_id] = ImagePose(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name)

                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.seek(24 * num_points2d, 1)

class GlassAnnotator:
    def __init__(self, pcd_path, colmap_dir, output_dir):
        self.pcd_path = pcd_path
        self.colmap_dir = Path(colmap_dir)
        self.sparse_dir = self.colmap_dir / "sparse" / "0"
        self.images_dir = self.colmap_dir / "images"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pcd_io = PointCloudIO()
        if not self.pcd_io.load_point_cloud(pcd_path):
            raise FileNotFoundError(f"Could not load point cloud: {pcd_path}")
        self.pcd = self.pcd_io.get_point_cloud()
        
        print("Loading Colmap data...")
        self.colmap_reader = ColmapReader(self.sparse_dir)
        print(f"Loaded {len(self.colmap_reader.cameras)} cameras and {len(self.colmap_reader.images)} images.")

        self.glass_labels = [] # List of {'label': str, 'points': np.array}

    def run(self):
        print("\n=== Glass Annotation Tool ===")
        print("Please annotate the 4 corners of each glass panel.")
        print("NOTE: Please select the 4 points in ORDER (e.g., Clockwise or Counter-Clockwise) to form a valid polygon.")
        
        glass_count = 0
        while True:
            glass_count += 1
            label = f"Glass {glass_count}"
            print(f"\n--- Annotating {label} ---")
            print("Please select 4 corner points in the visualization window (Shift + Left Click). Press Q to finish selection.")
            
            tool = SimpleAnnotationTool(self.pcd)
            coords = tool.run()
            
            if len(coords) != 4:
                print(f"Warning: You selected {len(coords)} points. 4 points are required for a glass panel.")
                retry = input("Retry? (y/n): ").strip().lower()
                if retry == 'y':
                    glass_count -= 1
                    continue
                else:
                    if input("Skip this glass? (y/n): ").strip().lower() == 'y':
                        glass_count -= 1
                        continue
                    else:
                         # User wants to keep it even if not 4 points? 
                         # No, strictly require 4 points as per requirement.
                         print("Ignoring invalid selection. Please retry.")
                         glass_count -= 1
                         continue
            
            self.glass_labels.append({
                'label': label,
                'points': coords
            })
            
            if input("Annotate another glass? (y/n): ").strip().lower() != 'y':
                break
        
        if not self.glass_labels:
            print("No glass annotated. Exiting.")
            return

        print(f"\nAnnotation finished. Processing {len(self.glass_labels)} glass panels...")
        self.process_images()

    def process_images(self):
        # Define 10 distinct colors for different glass labels
        colors = [
            (255, 0, 0, 100),      # Red
            (0, 255, 0, 100),      # Green
            (0, 0, 255, 100),      # Blue
            (255, 255, 0, 100),    # Yellow
            (255, 0, 255, 100),    # Magenta
            (0, 255, 255, 100),    # Cyan
            (255, 128, 0, 100),    # Orange
            (128, 0, 255, 100),    # Purple
            (0, 128, 255, 100),    # Light Blue
            (128, 255, 0, 100)     # Lime
        ]
        
        # Create output directories
        output_overlay_dir = self.output_dir / "overlay"
        output_color_mask_dir = self.output_dir / "color_masks"
        output_bw_mask_dir = self.output_dir / "bw_masks"
        output_overlay_dir.mkdir(parents=True, exist_ok=True)
        output_color_mask_dir.mkdir(parents=True, exist_ok=True)
        output_bw_mask_dir.mkdir(parents=True, exist_ok=True)

        # Prepare for JSON report
        per_image_components = {}
        id_to_color_map = {}
        color_to_id_list = []
        
        # Calculate global plane from all annotated points (simple fit)
        all_points = []
        for g in self.glass_labels:
            all_points.append(g['points'])
        
        plane_n = [0, 0, 1] # Default normal
        plane_d = 0
        if all_points:
            all_points_np = np.vstack(all_points)
            if len(all_points_np) >= 3:
                # Center the points
                centroid = np.mean(all_points_np, axis=0)
                u, s, vh = np.linalg.svd(all_points_np - centroid)
                normal = vh[2, :]
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, centroid)
                plane_n = normal.tolist()
                plane_d = float(d)

        # Prepare a file to write coordinates
        coords_file_path = self.output_dir / "mask_coordinates.txt"
        with open(coords_file_path, "w", encoding="utf-8") as f_coords:
            f_coords.write("Image Name, Glass Label, Point 1 (x,y), Point 2 (x,y), Point 3 (x,y), Point 4 (x,y)\n")

            for img_id, img_pose in self.colmap_reader.images.items():
                image_name = img_pose.name
                image_path = self.images_dir / image_name
                
                if not image_path.exists():
                    print(f"Warning: Image {image_name} not found at {image_path}")
                    continue
                
                # Initialize component list for this image
                per_image_components[image_name] = []

                try:
                    # Load image
                    with Image.open(image_path) as img:
                        img = img.convert("RGBA")
                        draw_overlay = ImageDraw.Draw(img, "RGBA")
                        
                        # Create color mask (for instance IDs) and black-white mask
                        width, height = img.size
                        color_mask_img = Image.new("RGB", (width, height), (0, 0, 0))
                        draw_color_mask = ImageDraw.Draw(color_mask_img)
                        bw_mask_img = Image.new("L", (width, height), 0)
                        draw_bw_mask = ImageDraw.Draw(bw_mask_img)
                        
                        camera = self.colmap_reader.cameras[img_pose.camera_id]
                        
                        annotated = False
                        
                        for i, glass in enumerate(self.glass_labels):
                            points_3d = glass['points']
                            # Add ID to label
                            global_id = i + 1
                            label = f"{glass['label']} (ID: {global_id})"
                            
                            color_overlay = colors[i % len(colors)]
                            # For mask, use full opacity
                            color_mask = color_overlay[:3]
                            
                            # Store color mapping if not already stored
                            if global_id not in id_to_color_map:
                                # Convert RGBA (255, 0, 0, 100) to RGB list [255, 0, 0]
                                rgb_color = list(color_overlay[:3])
                                id_to_color_map[global_id] = rgb_color
                                color_to_id_list.append({
                                    "color": rgb_color,
                                    "stage3_id_after_rgb_sort": global_id # Assuming simple mapping for now
                                })

                            # Transform to camera coordinates
                            points_cam = img_pose.transform_world_to_camera(points_3d)
                            
                            # 1. Check if points are in front of camera (z > 0)
                            if np.any(points_cam[:, 2] <= 0):
                                continue # Behind camera
                            
                            # Project to image
                            points_2d = camera.project(points_cam, width, height)
                            
                            # 2. Check if the polygon is mostly within the image
                            # Calculate bounding box of the projected points
                            min_x, min_y = np.min(points_2d, axis=0)
                            max_x, max_y = np.max(points_2d, axis=0)
                            
                            # If the bounding box is completely outside, skip
                            if max_x < 0 or min_x > width or max_y < 0 or min_y > height:
                                continue
                                
                            # 3. Visibility Check (Occlusion Handling)
                            # A simple check: if the projected area is too large (e.g., covers > 80% of image) 
                            # or if the points are extremely spread out, it's likely an invalid projection (e.g. object is too close or behind)
                            # Calculate polygon area
                            # (Shoelace formula)
                            x = points_2d[:, 0]
                            y = points_2d[:, 1]
                            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                            
                            image_area = width * height
                            if area > image_area * 0.8: # If glass covers more than 80% of image, it's suspicious for this dataset
                                 continue

                            # Sort points to form a convex polygon (counter-clockwise)
                            centroid_2d = np.mean(points_2d, axis=0)
                            angles = np.arctan2(points_2d[:, 1] - centroid_2d[1], points_2d[:, 0] - centroid_2d[0])
                            sort_indices = np.argsort(angles)
                            points_2d = points_2d[sort_indices]

                            # Format coordinates for output
                            # Round to integer for pixel coordinates
                            points_int = np.round(points_2d).astype(int)
                            
                            # Write to file
                            coord_str = ", ".join([f"({p[0]},{p[1]})" for p in points_int])
                            f_coords.write(f"{image_name}, {label}, {coord_str}\n")
                            
                            # Also print to console in Python slice format style as requested
                            # Although user asked for text output, I'll print a helpful format
                            print(f"[{image_name}] {label} Coordinates:")
                            # Determine min/max for potential slicing usage (just as a helper)
                            x_coords = points_int[:, 0]
                            y_coords = points_int[:, 1]
                            min_x, max_x = np.min(x_coords), np.max(x_coords)
                            min_y, max_y = np.min(y_coords), np.max(y_coords)
                            print(f"  Points: {points_int.tolist()}")
                            print(f"  Bounding Box Slice Suggestion: [{min_y}:{max_y}, {min_x}:{max_x}]")

                            # Store component info for JSON report
                            # Calculate 3D centroid (anchor)
                            anchor3d = np.mean(points_3d, axis=0).tolist()
                            
                            per_image_components[image_name].append({
                                "comp_idx": i, # Using glass index as component index within image context (though distinct from global ID)
                                "area": int(area),
                                "centroid_uv": [float(centroid_2d[0]), float(centroid_2d[1])],
                                "global_id": global_id,
                                "anchor_source": "manual_annotation_3d_centroid",
                                "anchor3d": anchor3d
                            })

                            # Draw polygon on overlay
                            polygon = [tuple(p) for p in points_2d]
                            draw_overlay.polygon(polygon, fill=color_overlay, outline=color_overlay[:3]+(255,))
                            
                            # Draw polygon on color and black-white masks
                            draw_color_mask.polygon(polygon, fill=color_mask)
                            draw_bw_mask.polygon(polygon, fill=255)
                            
                            # Draw label on overlay
                            # Only draw label if centroid is inside image
                            if 0 <= centroid_2d[0] <= width and 0 <= centroid_2d[1] <= height:
                                draw_overlay.text((centroid_2d[0], centroid_2d[1]), label, fill=(255, 255, 255, 255))
                            
                            annotated = True
                        
                        if annotated:
                            # Save overlay image
                            output_overlay_path = output_overlay_dir / f"annotated_{image_name}"
                            if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg'):
                                 img = img.convert("RGB")
                            img.save(output_overlay_path)
                            
                            # Save masks using the exact original image name
                            output_color_mask_path = output_color_mask_dir / image_name
                            output_bw_mask_path = output_bw_mask_dir / image_name
                            color_mask_img.save(output_color_mask_path)
                            bw_mask_img.save(output_bw_mask_path)
                            
                            print(f"Saved images for {image_name}")

                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
            
            print(f"\nCoordinate data saved to: {coords_file_path}")
            
            # Generate JSON Report
            print("Generating JSON report...")
            
            # Get first camera intrinsics (assuming mostly single camera setup or consistent model)
            first_cam_id = list(self.colmap_reader.cameras.keys())[0]
            first_cam = self.colmap_reader.cameras[first_cam_id]
            intrinsics_dict = {
                "fx": 0, "fy": 0, "cx": 0, "cy": 0,
                "width": first_cam.width, "height": first_cam.height
            }
            if first_cam.model == "PINHOLE":
                intrinsics_dict["fx"], intrinsics_dict["fy"] = first_cam.params[0], first_cam.params[1]
                intrinsics_dict["cx"], intrinsics_dict["cy"] = first_cam.params[2], first_cam.params[3]
            elif first_cam.model == "SIMPLE_PINHOLE":
                intrinsics_dict["fx"] = intrinsics_dict["fy"] = first_cam.params[0]
                intrinsics_dict["cx"], intrinsics_dict["cy"] = first_cam.params[1], first_cam.params[2]
            
            report = {
                "intrinsics": intrinsics_dict,
                "plane": {"n": plane_n, "d": plane_d},
                "anchor_merge_dist": 0.0, # Not applicable for manual annotation
                "global_windows": len(self.glass_labels),
                "id_to_color": {str(k): v for k, v in id_to_color_map.items()},
                "color_to_id_sorted_by_rgb": sorted(color_to_id_list, key=lambda x: (x['color'][0], x['color'][1], x['color'][2])),
                "images": per_image_components
            }
            
            report_path = self.output_dir / "geo_link_report.json"
            with open(report_path, "w", encoding="utf-8") as f_json:
                json.dump(report, f_json, ensure_ascii=False, indent=2)
            
            print(f"JSON report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate glass on point cloud and project to images.")
    parser.add_argument("--pcd", type=str, required=True, help="Path to points3D.ply")
    parser.add_argument("--colmap_dir", type=str, required=True, help="Path to Colmap directory (containing 'sparse' and 'images')")
    parser.add_argument("--output", type=str, default="output_images", help="Output directory for annotated images")
    
    args = parser.parse_args()
    
    annotator = GlassAnnotator(args.pcd, args.colmap_dir, args.output)
    annotator.run()
