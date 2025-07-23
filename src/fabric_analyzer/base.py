
# Advanced African Fabric-Aware Recoloring System
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import color, feature, filters, morphology, segmentation
from PIL import Image


class EnhancedAfricanFabricAnalyzer:
    """Enhanced analyzer for African fabric patterns, embroidery, and texture details"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_fabric_structure(self, image, mask):
        """Comprehensive fabric structure analysis for superior recoloring"""
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Apply mask to focus analysis on fabric area
        masked_gray = cv2.bitwise_and(gray, mask)
        
        results = {}
        
        # 1. Advanced embroidery detection
        results['embroidery_features'] = self._detect_embroidery_advanced(image, mask)
        
        # 2. Pattern and texture analysis
        results['texture_analysis'] = self._analyze_texture_patterns(masked_gray)
        
        # 3. Fabric fold and wrinkle detection
        results['fold_analysis'] = self._detect_folds_and_wrinkles(masked_gray)
        
        # 4. Color variation mapping
        results['color_variation'] = self._analyze_color_variations(image, mask)
        
        # 5. Fabric region segmentation
        results['fabric_regions'] = self._segment_fabric_regions_advanced(image, mask)
        
        # 6. Edge and seam detection
        results['edge_features'] = self._detect_edges_and_seams(masked_gray)
        
        return results
    
    def _detect_embroidery_advanced(self, image, mask):
        """Advanced embroidery and decorative pattern detection"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection for embroidery
        edges_fine = cv2.Canny(gray, 30, 100)  # Fine details
        edges_coarse = cv2.Canny(gray, 80, 200)  # Major outlines
        
        # Combine edge information
        embroidery_edges = cv2.addWeighted(edges_fine, 0.7, edges_coarse, 0.3, 0)
        embroidery_edges = cv2.bitwise_and(embroidery_edges, mask)
        
        # Detect embroidery regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        embroidery_regions = cv2.morphologyEx(embroidery_edges, cv2.MORPH_CLOSE, kernel)
        embroidery_regions = cv2.morphologyEx(embroidery_regions, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # Find high-contrast decorative areas
        mean_val = np.mean(gray[mask > 0])
        std_val = np.std(gray[mask > 0])
        decorative_areas = ((np.abs(gray - mean_val) > std_val * 1.5) & (mask > 0)).astype(np.uint8) * 255
        
        return {
            'edges': embroidery_edges,
            'regions': embroidery_regions,
            'decorative_areas': decorative_areas,
            'fine_details': edges_fine & mask,
            'major_outlines': edges_coarse & mask
        }
    
    def _analyze_texture_patterns(self, masked_gray):
        """Analyze fabric texture patterns using multiple methods"""
        
        # Local Binary Pattern for texture
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(masked_gray, P=24, R=3, method='uniform')
        
        # Gabor filters for pattern detection
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            real, _ = filters.gabor(masked_gray, frequency=0.1, theta=np.deg2rad(theta))
            gabor_responses.append(real)
        
        gabor_combined = np.sum(np.abs(gabor_responses), axis=0)
        
        # Structure tensor for orientation analysis
        Axx, Axy, Ayy = feature.structure_tensor(masked_gray, sigma=1)
        orientation = np.arctan2(2*Axy, Axx - Ayy) / 2
        
        return {
            'lbp': lbp,
            'gabor_response': gabor_combined,
            'orientation': orientation,
            'coherence': feature.structure_tensor_eigenvalues([Axx, Axy, Ayy])[0]
        }
    
    def _detect_folds_and_wrinkles(self, masked_gray):
        """Enhanced fold and wrinkle detection"""
        
        # Ridge detection using eigenvalues of Hessian
        from skimage.feature import hessian_matrix, hessian_matrix_eigvals
        
        H_elems = hessian_matrix(masked_gray, sigma=2,use_gaussian_derivatives=False)
        eigvals = hessian_matrix_eigvals(H_elems)
        
        # Ridges correspond to negative eigenvalues
        ridges = eigvals[0] < -0.01
        
        # Valley detection (positive eigenvalues)
        valleys = eigvals[1] > 0.01
        
        # Combine for fold detection
        folds = ridges | valleys
        
        # Directional filtering for linear features
        kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        kernel_v = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        
        folds_h = cv2.filter2D(masked_gray, -1, kernel_h)
        folds_v = cv2.filter2D(masked_gray, -1, kernel_v)
        folds_directional = np.sqrt(folds_h**2 + folds_v**2)
        
        return {
            'ridges': ridges.astype(np.uint8) * 255,
            'valleys': valleys.astype(np.uint8) * 255,
            'folds_combined': folds.astype(np.uint8) * 255,
            'directional_folds': folds_directional
        }
    
    def _analyze_color_variations(self, image, mask):
        """Analyze color variations across the fabric"""
        
        # Convert to LAB for perceptual color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate local color statistics
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return {'mean_color': np.array([0, 0, 0]), 'color_variance': 0}
        
        mean_color = np.mean(masked_pixels, axis=0)
        color_variance = np.var(masked_pixels, axis=0)
        
        # Create brightness map
        brightness = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        brightness_map = brightness * (mask / 255.0)
        
        # Identify lighting variations
        mean_brightness = np.mean(brightness[mask > 0])
        std_brightness = np.std(brightness[mask > 0])
        
        # Classify regions by brightness
        highlight_regions = ((brightness > mean_brightness + std_brightness) & (mask > 0)).astype(np.uint8) * 255
        shadow_regions = ((brightness < mean_brightness - std_brightness) & (mask > 0)).astype(np.uint8) * 255
        midtone_regions = ((np.abs(brightness - mean_brightness) <= std_brightness) & (mask > 0)).astype(np.uint8) * 255
        
        return {
            'mean_color': mean_color,
            'color_variance': color_variance,
            'brightness_map': brightness_map,
            'highlight_regions': highlight_regions,
            'shadow_regions': shadow_regions,
            'midtone_regions': midtone_regions
        }
    
    def _segment_fabric_regions_advanced(self, image, mask):
        """Advanced fabric region segmentation"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Statistical analysis of the fabric
        masked_area = image[mask > 0]
        if len(masked_area) == 0:
            return self._get_empty_regions(mask)
        
        mean_brightness = np.mean(gray[mask > 0])
        std_brightness = np.std(gray[mask > 0])
        
        # Enhanced region classification
        # Main fabric (primary material)
        main_fabric = ((gray >= mean_brightness - 0.5 * std_brightness) & 
                      (gray <= mean_brightness + 0.5 * std_brightness) & 
                      (mask > 0))
        
        # Embroidery/decorative (high contrast areas)
        embroidery = ((np.abs(gray - mean_brightness) > 1.2 * std_brightness) & (mask > 0))
        
        # Seams and edges (linear features)
        seams = self._detect_seams_and_folds_advanced(gray) & (mask > 0)
        
        # Shadow areas (darker regions)
        shadows = ((gray < mean_brightness - 0.8 * std_brightness) & (mask > 0) & ~embroidery)
        
        # Highlight areas (brighter regions)
        highlights = ((gray > mean_brightness + 0.8 * std_brightness) & (mask > 0) & ~embroidery)
        
        return {
            'main_fabric': main_fabric.astype(np.uint8) * 255,
            'embroidery': embroidery.astype(np.uint8) * 255,
            'seams': seams.astype(np.uint8) * 255,
            'shadows': shadows.astype(np.uint8) * 255,
            'highlights': highlights.astype(np.uint8) * 255
        }
    
    def _detect_seams_and_folds_advanced(self, gray_image):
        """Advanced seam and fold line detection"""
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray_image, 50, 150)
        edges2 = cv2.Canny(gray_image, 30, 100)
        
        # Combine edges
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        # Hough line detection for seams
        lines = cv2.HoughLinesP(edges_combined, 1, np.pi/180, threshold=40, 
                               minLineLength=25, maxLineGap=8)
        
        # Create seam mask
        seam_mask = np.zeros_like(gray_image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(seam_mask, (x1, y1), (x2, y2), 255, 2)
        
        return seam_mask > 0
    
    def _detect_edges_and_seams(self, masked_gray):
        """Comprehensive edge and seam detection"""
        
        # Multi-method edge detection
        edges_canny = cv2.Canny(masked_gray, 50, 150)
        
        # Sobel edges
        sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx**2 + sobely**2)
        
        # Laplacian edges
        laplacian = cv2.Laplacian(masked_gray, cv2.CV_64F)
        
        return {
            'canny_edges': edges_canny,
            'sobel_edges': (sobel_combined / sobel_combined.max() * 255).astype(np.uint8),
            'laplacian_edges': (np.abs(laplacian) / np.abs(laplacian).max() * 255).astype(np.uint8)
        }
    
    def _get_empty_regions(self, mask):
        """Return empty regions when no masked area is found"""
        empty = np.zeros_like(mask)
        return {
            'main_fabric': empty,
            'embroidery': empty,
            'seams': empty,
            'shadows': empty,
            'highlights': empty
        }



class BaseAfricanClothingStyleRecolorer:
    def __init__(self, image, mask):
        self.image = image.astype(np.float32)
        self.mask = mask
        self.fabric_analyzer = EnhancedAfricanFabricAnalyzer()

    def _detect_embroidery_advanced(self, image, mask):
        """
        REVISED: Advanced embroidery detection using texture and ridge analysis,
        which is more effective for low-contrast (e.g., white-on-white) patterns.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Use ridge detection to find fine, thread-like structures
        from skimage.feature import hessian_matrix, hessian_matrix_eigvals
        # A sigma of ~1.5 is good for typical thread widths in high-res images
        H_elems = hessian_matrix(gray, sigma=1.5, mode='reflect', cval=0)
        eigvals = hessian_matrix_eigvals(H_elems)
        # Ridges have a strong negative response in the Hessian eigenvalues
        ridges = (eigvals[0] < -0.01) & (mask > 0)
        ridges = ridges.astype(np.uint8) * 255

        # 2. Use Canny edge detection for additional detail
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_fine = cv2.bitwise_and(edges_fine, mask)

        # 3. Combine the texture features, giving more weight to the specific ridge detection
        combined_features = cv2.addWeighted(ridges, 1.0, edges_fine, 0.5, 0)

        # 4. Use morphological operations to connect broken lines and form solid regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Close small gaps in the detected threads
        embroidery_regions = cv2.morphologyEx(combined_features, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Thicken the lines slightly to ensure a solid mask
        embroidery_regions = cv2.dilate(embroidery_regions, kernel, iterations=1)
        
        return {
            'edges': edges_fine,
            'regions': embroidery_regions, # Pass the refined regions for backward compatibility
            'decorative_areas': embroidery_regions, # CRITICAL: Use our new robust mask
            'fine_details': edges_fine,
            'major_outlines': cv2.Canny(gray, 80, 200) & mask
        }
    
    def _advanced_fabric_aware_method(self, target_rgb, fabric_analysis):
        """
        FIXED: Advanced method that preserves target color while maintaining fabric awareness
        """
        target_rgb = np.array(target_rgb, dtype=np.uint8)
        active_mask = self.mask > 0
        if not np.any(active_mask):
            return self.image.astype(np.uint8)

        # --- 1. Base Setup ---
        input_image_uint8 = np.clip(self.image, 0, 255).astype(np.uint8)
        lab_image = cv2.cvtColor(input_image_uint8, cv2.COLOR_RGB2LAB)
        
        # Convert target color to LAB
        target_lab = cv2.cvtColor(np.array([[target_rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0]
        target_l, target_a, target_b = target_lab.astype(np.float32)

        # --- 2. FIX: Preserve Target Color While Maintaining Fabric Detail ---
        # Get original luminance channel for preserving shading/folds
        original_l_channel = lab_image[:, :, 0].astype(np.float32)
        
        # Calculate the average brightness of the original fabric within the mask
        avg_fabric_l = np.mean(original_l_channel[active_mask])
        
        # Create a "shading map" that represents the folds and highlights
        # relative to the average brightness
        shading_map = original_l_channel - avg_fabric_l
        
        # Apply this shading map to the TARGET color's lightness value
        # This preserves the original folds but on the new color
        new_l_channel = np.clip(target_l + shading_map, 0, 100)

        # --- 3. FIX: Handle Highlights Properly ---
        # Find extremely bright pixels and make them bright versions of target color
        highlight_mask = (original_l_channel > 97) & active_mask
        new_l_channel[highlight_mask] = np.clip(target_l + 15, 0, 100)

        # --- 4. Apply Complete Target Color (L, A, B channels) ---
        recolored_lab = lab_image.copy()
        
        # CRITICAL FIX: Apply ALL three channels with the new luminance
        recolored_lab[active_mask, 0] = new_l_channel[active_mask].astype(np.uint8)
        recolored_lab[active_mask, 1] = target_a  # Target color's A channel
        recolored_lab[active_mask, 2] = target_b  # Target color's B channel

        # Convert back to RGB
        result = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2RGB)

        # --- 5. Preserve Embroidery Details ---
        embroidery_mask_bool = (fabric_analysis['embroidery_features']['decorative_areas'] > 0) & active_mask
        embroidery_mask_bool = morphology.remove_small_objects(embroidery_mask_bool, min_size=50, connectivity=2)

        if np.any(embroidery_mask_bool):
            original_embroidery_color = input_image_uint8[embroidery_mask_bool].astype(np.float32)
            new_base_color = result[embroidery_mask_bool].astype(np.float32)
            
            blend_factor = 0.6
            blended_color = (original_embroidery_color * blend_factor) + (new_base_color * (1 - blend_factor))
            result[embroidery_mask_bool] = np.clip(blended_color, 0, 255).astype(np.uint8)

        # --- 6. FIX: Handle White Lines (Edge Artifacts) ---
        # Dilate the mask slightly to ensure complete coverage
        from scipy import ndimage
        dilated_mask = ndimage.binary_dilation(active_mask, structure=np.ones((3,3)))
        
        # Apply result with the dilated mask to prevent white lines
        final_result = input_image_uint8.copy()
        final_result[dilated_mask] = result[dilated_mask]
        
        # --- 7. ADDITIONAL FIX: Color Consistency Check ---
        # Ensure the final result maintains target color consistency
        # by checking if the average color in non-embroidery areas matches target
        non_embroidery_mask = active_mask & ~embroidery_mask_bool if np.any(embroidery_mask_bool) else active_mask
        
        if np.any(non_embroidery_mask):
            # Get average color in recolored area
            avg_result_color = np.mean(final_result[non_embroidery_mask], axis=0)
            target_color_float = target_rgb.astype(np.float32)
            
            # If there's significant color drift, apply a subtle correction
            color_difference = np.linalg.norm(avg_result_color - target_color_float)
            if color_difference > 20:  # Threshold for acceptable color drift
                # Apply a gentle color correction to maintain target color
                correction_factor = 0.8  # How much to push toward target color
                corrected_color = (final_result[non_embroidery_mask].astype(np.float32) * (1 - correction_factor) + 
                                target_color_float * correction_factor)
                final_result[non_embroidery_mask] = np.clip(corrected_color, 0, 255).astype(np.uint8)

        return final_result
    
    def _fabric_aware_method(self, target_rgb):
        """FIXED: Method that understands fabric properties with proper color handling"""
        
        # FIXED: Ensure consistent data types
        image_uint8 = np.clip(self.image, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        
        # Identify different fabric regions
        main_fabric = (gray > 180) & (self.mask > 0)
        shadow_areas = (gray > 100) & (gray <= 180) & (self.mask > 0)
        detail_areas = (gray <= 100) & (self.mask > 0)
        
        result = image_uint8.astype(np.float32)  # Work in float32 for precision
        target_rgb = np.array(target_rgb, dtype=np.float32)
        
        # Process each region differently
        if np.any(main_fabric):
            brightness_factor = gray[main_fabric].astype(np.float32) / 200.0
            brightness_factor = np.clip(brightness_factor, 0.7, 1.1)
            
            for i in range(3):
                result[main_fabric, i] = target_rgb[i] * brightness_factor
        
        if np.any(shadow_areas):
            shadow_brightness = gray[shadow_areas].astype(np.float32) / 255.0
            shadow_factor = shadow_brightness * 0.6 + 0.2
            
            for i in range(3):
                result[shadow_areas, i] = target_rgb[i] * shadow_factor
        
        if np.any(detail_areas):
            original_color = image_uint8[detail_areas].astype(np.float32)
            detail_brightness = gray[detail_areas].astype(np.float32) / 255.0
            
            blend_ratio = 0.4 + detail_brightness * 0.3
            
            for i in range(3):
                blended = target_rgb[i] * (1 - blend_ratio) + original_color[:, i] * blend_ratio
                result[detail_areas, i] = blended
        
        # FIXED: Ensure proper clipping and data type conversion
        return np.clip(result, 0, 255).astype(np.uint8)
