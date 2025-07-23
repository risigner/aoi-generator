import json
from pathlib import Path
import cv2
import numpy as np


class BaseAfricanFabricMaskGenerator:
    def __init__(self, image_shape, mask_save_dir="masks"):
        self.image_shape = image_shape
        self.mask_save_dir = Path(mask_save_dir)
        self.mask_save_dir.mkdir(exist_ok=True)

    def generate_advanced_mask(self, polygon_points_list, dilate_iterations=1, 
                             edge_preserve_strength=0.8, feather_size=3):
        """Generate mask with advanced edge preservation and feathering"""
        
        # Create base mask
        mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
        # Handle both single polygon and multiple segments
        if isinstance(polygon_points_list[0], (list, tuple)) and len(polygon_points_list[0]) == 2:
            # Single polygon (backward compatibility)
            cv2.fillPoly(mask, [np.array(polygon_points_list, dtype=np.int32)], 255)
        else:
            # Multiple segments - group points into segments of 3+ consecutive points
            # This is a simple approach - you might want to modify based on your segment tracking
            current_segment = []
            for point in polygon_points_list:
                current_segment.append(point)
                # Simple heuristic: if we have enough points for a triangle, fill it
                # You'll want to replace this with proper segment tracking from the drawer
                if len(current_segment) >= 3:
                    cv2.fillPoly(mask, [np.array(current_segment, dtype=np.int32)], 255)
                    current_segment = []
        
        # Very light dilation to avoid cutting off fabric edges
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        
        # Advanced feathering using distance transform
        mask_float = mask.astype(np.float32) / 255.0
        
        # Create feathered edges
        if feather_size > 0:
            # Use bilateral filter to create smooth but edge-aware feathering
            mask_feathered = cv2.bilateralFilter(mask, -1, feather_size*10, feather_size*2)
            
            # Blend original mask with feathered version
            mask = (mask_float * edge_preserve_strength + 
                   (mask_feathered.astype(np.float32) / 255.0) * (1 - edge_preserve_strength))
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        
        return mask
    def save_mask(self, mask, image_path, polygon_points):
        """Save mask and metadata for reuse"""
        image_name = Path(image_path).stem
        
        # Save mask as image
        mask_path = self.mask_save_dir / f"{image_name}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Save metadata
        metadata = {
            'image_path': str(image_path),
            'polygon_points': polygon_points,
            'image_shape': self.image_shape,
            'mask_path': str(mask_path)
        }
        
        metadata_path = self.mask_save_dir / f"{image_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Mask saved: {mask_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return mask_path, metadata_path
    
    def save_mask_with_custom_image(self, mask, image_path, polygon_points):
        """Save mask with a #f6f6f6 background and metadata for reuse"""
        image_name = Path(image_path).stem
        
        # Convert mask to 3-channel RGB with #f6f6f6 background
        bg_color = np.array([246, 246, 246], dtype=np.uint8)
        mask_rgb = np.full((*mask.shape, 3), bg_color, dtype=np.uint8)

        # Apply white foreground where mask is active
        white_foreground = np.array([255, 255, 255], dtype=np.uint8)
        mask_indices = mask > 0
        mask_rgb[mask_indices] = white_foreground

        # Save RGB mask as image
        mask_path = self.mask_save_dir / f"{image_name}_mask.png"
        cv2.imwrite(str(mask_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

        # Save metadata
        metadata = {
            'image_path': str(image_path),
            'polygon_points': polygon_points,
            'image_shape': self.image_shape,
            'mask_path': str(mask_path)
        }
        metadata_path = self.mask_save_dir / f"{image_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Mask saved: {mask_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return mask_path, metadata_path

    def load_mask(self, image_path,gender):
        """Load previously saved mask"""
        image_name = Path(image_path).stem
        mask_path = self.mask_save_dir / f"{gender}/{image_name}_mask.png"
        metadata_path = self.mask_save_dir / f"{gender}/{image_name}_metadata.json"
        print(mask_path)
        if mask_path.exists() and metadata_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"Loaded existing mask: {mask_path}")
            return mask, metadata
        
        return None, None
    
    def load_mask_with_custom_image(self, image_path):
        """Load saved mask and metadata given the original image path"""
        image_name = Path(image_path).stem
        print(image_name)
        metadata_path = self.mask_save_dir / f"male/{image_name}_metadata.json"
        mask_path = self.mask_save_dir / f"male/{image_name}_mask.png"

        if not metadata_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Metadata or mask file not found for {image_name}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load RGB mask
        mask_rgb = cv2.imread(str(mask_path))
        if mask_rgb is None:
            raise ValueError(f"Could not read mask image at {mask_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        # Optional: validate image_shape
        if tuple(metadata['image_shape']) != tuple(self.image_shape):
            print("[Warning] Loaded image shape doesn't match current config")

        return mask_rgb, metadata

