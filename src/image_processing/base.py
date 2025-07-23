
from pathlib import Path
import cv2
import numpy as np
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
