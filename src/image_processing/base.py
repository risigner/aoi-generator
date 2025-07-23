
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import numpy as np
import cv2
from PIL import Image



class ImageProcessor:
    """Utility class for image processing and saving"""
    
    @staticmethod
    def trim_recolored_image(original_image, recolored_image, mask, trim_strength=0.95):
        """Trim recolored images to remove bleeding outside mask boundaries"""
        
        # Create a tighter mask by eroding the original mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tight_mask = cv2.erode(mask, kernel, iterations=1)
        
        # Create smooth transition mask
        transition_mask = cv2.GaussianBlur(tight_mask.astype(np.float32), (5, 5), 0) / 255.0
        transition_mask = np.clip(transition_mask * trim_strength, 0, 1)
        
        # Apply trimming
        trimmed_result = original_image.copy().astype(np.float32)
        
        for i in range(3):
            trimmed_result[:,:,i] = (original_image[:,:,i].astype(np.float32) * (1 - transition_mask) + 
                                   recolored_image[:,:,i].astype(np.float32) * transition_mask)
        
        return trimmed_result.astype(np.uint8)
    
    @staticmethod
    def save_images_high_res(images_dict, base_name, output_dir="output", 
                           save_ultra_high_res=True, scale_factor:float=2.0):
        """Save images in ultra high resolution and 1080x1080"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        for name, image in images_dict.items():
            # Clean filename
            clean_name = name.replace(" ", "_").replace("/", "_")
            
            # Save ultra high resolution (original size)
            if save_ultra_high_res:
                ultra_high_path = output_path / f"{base_name}_{clean_name}_ultra_high.png"
                # Ensure uint8
                image = np.clip(image, 0, 255).astype(np.uint8)
                
                # Scale image
                if scale_factor != 1.0:
                    height, width = image.shape[:2]
                    new_size = (int(width * scale_factor), int(height * scale_factor))
                    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)  # High-quality scaling

                # Convert to PIL and save with maximum quality
                pil_image = Image.fromarray(image)
                pil_image.save(ultra_high_path, 'PNG', optimize=False, compress_level=0)
                saved_files[f"{name}_ultra_high"] = str(ultra_high_path)
                print(f"Saved ultra high-res: {ultra_high_path}")
        
        return saved_files

    @staticmethod
    def fill_transparent_background(image: np.ndarray, alpha_channel: np.ndarray, bg_color=(246, 246, 246)) -> np.ndarray:
        """Replaces transparent pixels with a solid background color"""
        if alpha_channel is None:
            return image

        bg = np.full_like(image, bg_color, dtype=np.uint8)
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return np.where(mask_3ch, image, bg)
    
    @staticmethod
    def fill_with_image_background(image: np.ndarray, alpha_channel: np.ndarray, bg_image_path: str) -> np.ndarray:
        """Replaces transparent pixels with an image background"""
        if alpha_channel is None:
            return image
        
        # Load background image
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            raise ValueError(f"Could not load background image: {bg_image_path}")
        
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
        # Resize background to match foreground dimensions
        h, w = image.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        
        # Apply alpha blending
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        return np.where(mask_3ch, image, bg_resized)
    
    @staticmethod
    def fill_with_image_background_from_array(image: np.ndarray, alpha_channel: np.ndarray, bg_image: np.ndarray) -> np.ndarray:
        """Replaces transparent pixels with an image background (from numpy array)"""
        if alpha_channel is None:
            return image
        
        # Resize background to match foreground dimensions
        h, w = image.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        
        # Apply alpha blending
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        return np.where(mask_3ch, image, bg_resized)
    
    @staticmethod
    def fill_with_tiled_background(image: np.ndarray, alpha_channel: np.ndarray, bg_image_path: str) -> np.ndarray:
        """Replaces transparent pixels with a tiled background pattern"""
        if alpha_channel is None:
            return image
        
        # Load background tile
        bg_tile = cv2.imread(bg_image_path)
        if bg_tile is None:
            raise ValueError(f"Could not load background image: {bg_image_path}")
        
        bg_tile = cv2.cvtColor(bg_tile, cv2.COLOR_BGR2RGB)
        
        # Create tiled background
        h, w = image.shape[:2]
        tile_h, tile_w = bg_tile.shape[:2]
        
        # Calculate how many tiles we need
        tiles_y = (h + tile_h - 1) // tile_h
        tiles_x = (w + tile_w - 1) // tile_w
        
        # Create tiled background
        tiled_bg = np.tile(bg_tile, (tiles_y, tiles_x, 1))
        tiled_bg = tiled_bg[:h, :w]  # Crop to exact size
        
        # Apply alpha blending
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        return np.where(mask_3ch, image, tiled_bg)
    
    @staticmethod
    def fill_with_blurred_background(image: np.ndarray, alpha_channel: np.ndarray, bg_image_path: str, blur_strength: int = 15) -> np.ndarray:
        """Replaces transparent pixels with a blurred version of a background image"""
        if alpha_channel is None:
            return image
        
        # Load and process background
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            raise ValueError(f"Could not load background image: {bg_image_path}")
        
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
        # Resize and blur background
        h, w = image.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        bg_blurred = cv2.GaussianBlur(bg_resized, (blur_strength, blur_strength), 0)
        
        # Apply alpha blending
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        return np.where(mask_3ch, image, bg_blurred)
    
    @staticmethod
    def fill_with_gradient_overlay(image: np.ndarray, alpha_channel: np.ndarray, bg_image_path: str, gradient_alpha: float = 0.3) -> np.ndarray:
        """Combines background image with a gradient overlay"""
        if alpha_channel is None:
            return image
        
        # Load background
        bg_image = cv2.imread(bg_image_path)
        if bg_image is None:
            raise ValueError(f"Could not load background image: {bg_image_path}")
        
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
        # Resize background
        h, w = image.shape[:2]
        bg_resized = cv2.resize(bg_image, (w, h))
        
        # Create gradient (top to bottom)
        gradient = np.linspace(0, 255, h, dtype=np.uint8)
        gradient = np.tile(gradient.reshape(-1, 1, 1), (1, w, 3))
        
        # Blend background with gradient
        bg_with_gradient = cv2.addWeighted(bg_resized, 1-gradient_alpha, gradient, gradient_alpha, 0)
        
        # Apply alpha blending
        mask = alpha_channel > 0
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        return np.where(mask_3ch, image, bg_with_gradient)
