# Advanced African Fabric-Aware Recoloring System
import matplotlib.pyplot as plt
import numpy as np
import cv2,json
from pathlib import Path
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Polygon as MPLPolygon

from src.fabric_mask_generator.base import BaseAfricanFabricMaskGenerator
from src.image_processing.base import ImageProcessor
from src.fabric_analyzer.base import BaseAfricanClothingStyleRecolorer
from src.polygon_drawer.base import BaseAdvancedPolygonDrawer


class EnhancedPolygonDrawer(BaseAdvancedPolygonDrawer):
    """
    An enhanced drawer for creating multiple, disconnected polygon segments.
    """
    def __init__(self, image, point_size=6, line_width=2):
        self.segments = []
        self.current_segment_points = []
        self.current_segment_visuals = {'circles': [], 'lines': [], 'patch': None}
        self.segment_colors = ['red', 'cyan', 'lime', 'orange', 'magenta', 'yellow']
        self.current_segment_index = 0
        # Call parent __init__ AFTER setting up child-specific properties
        super().__init__(image, point_size, line_width)

    def setup_control_buttons(self):
        """Override to add the 'New Segment' button."""
        super().setup_control_buttons()
        ax_new_segment = plt.axes([0.54, 0.02, 0.12, 0.05])
        self.btn_new_segment = Button(ax_new_segment, 'New Segment')
        self.btn_new_segment.on_clicked(self.start_new_segment)
    
    def get_status_text(self):
        """Override to show segment-specific status."""
        segment_info = f"Segment {self.current_segment_index + 1}"
        mode_text = "DRAWING" if self.drawing_mode else "NAVIGATION"
        
        if self.closed:
            return f"✅ SEGMENT CLOSED | {segment_info} | Click 'New Segment' or 'Finish'"
        
        points_drawn = len(self.current_segment_points)
        if points_drawn == 0:
            return f"🎯 STARTING {segment_info} | Left-click to add first point"
        
        status = f"📍 {mode_text} MODE | {points_drawn} points on {segment_info}"
        if points_drawn < 3:
            status += f" | Need {3-points_drawn} more"
        else:
            status += " | Right-click to close segment"
        return status
    
    def add_point(self, x, y):
        """Override to handle points for the current segment."""
        if self.closed:
            print("⚠️ Current segment is closed. Start a new one to continue drawing.")
            return
            
        new_point = (int(x), int(y))
        self.current_segment_points.append(new_point)
        color = self.segment_colors[self.current_segment_index % len(self.segment_colors)]
        
        circle = Circle(new_point, self.point_size, color=color, zorder=5)
        self.ax.add_patch(circle)
        self.current_segment_visuals['circles'].append(circle)
        
        if len(self.current_segment_points) > 1:
            prev_point = self.current_segment_points[-2]
            line, = self.ax.plot([prev_point[0], new_point[0]], 
                               [prev_point[1], new_point[1]], 
                               color=color, linewidth=self.line_width, zorder=4)
            self.current_segment_visuals['lines'].append(line)
        
        self.update_status()
        self.fig.canvas.draw_idle()

    def update_preview_line(self):
        """Override to use current segment points."""
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        
        if not self.mouse_pos or not self.current_segment_points: return

        last_point = self.current_segment_points[-1]
        color = self.segment_colors[self.current_segment_index % len(self.segment_colors)]
        self.preview_line, = self.ax.plot([last_point[0], self.mouse_pos[0]], 
                                         [last_point[1], self.mouse_pos[1]], 
                                         f'{color}--', linewidth=1)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Override to check current segment length."""
        if not event.inaxes == self.ax or not self.drawing_mode or event.xdata is None:
            return
        if event.button == 1:
            self.add_point(event.xdata, event.ydata)
        elif event.button == 3 and len(self.current_segment_points) > 2:
            self.close_polygon()

    def close_polygon(self):
        """Override to close the current segment and store it."""
        if len(self.current_segment_points) < 3 or self.closed: return
        
        color = self.segment_colors[self.current_segment_index % len(self.segment_colors)]
        
        # Add closing line
        first, last = self.current_segment_points[0], self.current_segment_points[-1]
        line, = self.ax.plot([last[0], first[0]], [last[1], first[1]], 
                           color=color, linewidth=self.line_width)
        self.current_segment_visuals['lines'].append(line)
        
        # Create filled polygon patch
        patch = MPLPolygon(np.array(self.current_segment_points), alpha=0.3, 
                                  facecolor=color, edgecolor=color)
        self.ax.add_patch(patch)
        self.current_segment_visuals['patch'] = patch

        # Save the completed segment
        self.segments.append({
            'points': self.current_segment_points,
            'visuals': self.current_segment_visuals
        })
        
        self.closed = True
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
            
        self.update_status()
        self.fig.canvas.draw_idle()
        print(f"🎉 SEGMENT {self.current_segment_index + 1} COMPLETED!")
    def finish_polygon(self, event=None):
        """
        Correctly checks for completed segments or a valid current segment before finishing.
        """
        has_completed_segments = len(self.segments) > 0
        current_segment_is_drawable = len(self.current_segment_points) >= 3

        if has_completed_segments or current_segment_is_drawable:
            # If the user is drawing a valid segment but hasn't closed it, close it for them.
            if current_segment_is_drawable and not self.closed:
                self.close_polygon()
            plt.close(self.fig)
            print("✅ Polygon drawing finished.")
        else:
            print("⚠️ Need at least one segment with 3+ points to finish.")

    def start_new_segment(self, event=None):
        """Starts a new segment if the current one is finished."""
        if len(self.current_segment_points) > 0 and not self.closed:
            print("⚠️ Please close the current segment (right-click) before starting a new one.")
            return
            
        # Reset for the new segment
        self.current_segment_points = []
        self.current_segment_visuals = {'circles': [], 'lines': [], 'patch': None}
        self.closed = False
        if len(self.segments) > 0: # Only increment index if we've completed at least one
            self.current_segment_index = len(self.segments)

        self.update_status()
        self.fig.canvas.draw_idle()
        print(f"🆕 Started new segment {self.current_segment_index + 1}")

    def undo_point(self, event=None):
        """Override to undo points from the current segment."""
        if len(self.current_segment_points) == 0 or self.closed: return
            
        self.current_segment_points.pop()
        
        if self.current_segment_visuals['circles']:
            self.current_segment_visuals['circles'][-1].remove()
            self.current_segment_visuals['circles'].pop()
            
        if self.current_segment_visuals['lines']:
            self.current_segment_visuals['lines'][-1].remove()
            self.current_segment_visuals['lines'].pop()
            
        self.update_status()
        self.fig.canvas.draw_idle()

    def clear_all(self, event=None):
        """Override to clear all segments and the current drawing."""
        # Clear completed segments
        for seg in self.segments:
            for circle in seg['visuals']['circles']: circle.remove()
            for line in seg['visuals']['lines']: line.remove()
            if seg['visuals']['patch']: seg['visuals']['patch'].remove()
        self.segments = []

        # Clear current drawing
        for circle in self.current_segment_visuals['circles']: circle.remove()
        for line in self.current_segment_visuals['lines']: line.remove()
        if self.preview_line: self.preview_line.remove()
        
        # Reset state
        self.current_segment_points = []
        self.current_segment_visuals = {'circles': [], 'lines': [], 'patch': None}
        self.preview_line = None
        self.closed = False
        self.current_segment_index = 0
        
        self.update_status()
        self.fig.canvas.draw_idle()
        print("🗑️ All segments and points have been cleared.")

    def get_all_points(self):
        """Returns a list containing point lists for each segment."""
        all_segments_points = [seg['points'] for seg in self.segments]
        # Also include the currently active (but unclosed) segment if it exists
        if self.current_segment_points and not self.closed:
            all_segments_points.append(self.current_segment_points)
        return all_segments_points

    def draw(self):
        """Main method to start drawing and return points from all segments."""
        print("\n🚀 Starting Multi-Segment Polygon Drawer...")
        print("   'f' to finish | 'u' to undo | 'c' to clear | 'esc' to cancel")
        plt.show()
        return self.get_all_points()
    

class AfricanFabricMaskGenerator(BaseAfricanFabricMaskGenerator):
      def generate_advanced_mask(self, segments_list, dilate_iterations=1,
                             edge_preserve_strength=0.8, feather_size=3):
        """
        Generate mask by correctly filling each polygon segment.
        """
        mask = np.zeros(self.image_shape[:2], dtype=np.uint8)

        if not segments_list:
            print("Warning: No polygon data received to generate mask.")
            return mask

        # Iterate through each segment (which is a list of points)
        for polygon_points in segments_list:
            if polygon_points and len(polygon_points) >= 3:
                # Convert the list of points to a NumPy array for cv2.fillPoly
                poly_np = np.array(polygon_points, dtype=np.int32)
                # cv2.fillPoly expects a list of polygons, so we wrap our single polygon in a list
                cv2.fillPoly(mask, [poly_np], 255)

        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

        if feather_size > 0:
            mask_float = mask.astype(np.float32) / 255.0
            mask_feathered = cv2.bilateralFilter(mask, -1, feather_size*10, feather_size*2)
            mask_blend = (mask_float * edge_preserve_strength +
                         (mask_feathered.astype(np.float32) / 255.0) * (1 - edge_preserve_strength))
            mask = np.clip(mask_blend * 255, 0, 255).astype(np.uint8)

        return mask


class AfricanClothingStyleRecolorer(BaseAfricanClothingStyleRecolorer):
    
 
    def _advanced_fabric_aware_method(self, target_rgb, fabric_analysis=None):
        """
        FIXED: Precisely recolors fabric using LAB color space, preserving realistic shading.
        This method blends the lightness (L) of the target color with the original
        fabric's lightness to maintain texture while accurately achieving the
        target color's overall brightness. The color information (a, b) is
        taken directly from the target color for high fidelity.
        """
        active_mask = self.mask > 0
        if not np.any(active_mask):
            return self.image.astype(np.uint8)

        # 1. Convert source image and target color to LAB space
        # Ensure input image is uint8 for color conversion
        input_image_uint8 = np.clip(self.image, 0, 255).astype(np.uint8)
        lab_image = cv2.cvtColor(input_image_uint8, cv2.COLOR_RGB2LAB)

        # Convert the single target_rgb color to its LAB representation
        target_lab = cv2.cvtColor(np.array([[target_rgb]], dtype=np.uint8), cv2.COLOR_RGB2LAB)[0][0]
        target_l, target_a, target_b = target_lab

        # 2. Blend Lightness (L channel)
        # This is the key step to fix the color shift. We preserve the original
        # shading's *variation* but shift its *overall level* to match the target.
        original_l_channel = lab_image[:, :, 0].astype(np.float32)
        
        # Alpha controls the blend. 0.0 = all original lightness (causes color shift).
        # 1.0 = all target lightness (flattens texture). 0.4-0.6 is a good balance.
        alpha = 0.65
        if target_rgb[1] > 0:
            alpha = 0.55
        # Perform blending in float precision to avoid clipping/rounding errors
        blended_l = (alpha * float(target_l) + (1 - alpha) * original_l_channel)
        
        # Clip to the valid LAB L-channel range [0, 255] and convert back to uint8
        blended_l_uint8 = np.clip(blended_l, 0, 255).astype(np.uint8)

        # 3. Apply the new color to the masked area
        recolored_lab = lab_image.copy()
        recolored_lab[active_mask, 0] = blended_l_uint8[active_mask] # Use blended lightness
        recolored_lab[active_mask, 1] = target_a                     # Use precise target color's 'a' channel
        recolored_lab[active_mask, 2] = target_b                     # Use precise target color's 'b' channel

        # 4. Convert back to RGB for the final result
        result_rgb = cv2.cvtColor(recolored_lab, cv2.COLOR_LAB2LRGB)

        return np.clip(result_rgb, 0, 255).astype(np.uint8)



def advanced_african_fabric_pipeline(image_path, target_colors=None, load_existing_mask=True,
                                   use_advanced_analyzer=False, save_images=True, 
                                   trim_images=True):
    """Complete advanced pipeline for African fabric recoloring"""
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        alpha_channel = None
        rgb_img = img[..., ::-1]  # fallback BGR to RGB
    
    img = rgb_img
    
    print("=== ADVANCED AFRICAN FABRIC RECOLORING SYSTEM ===")
    
    # Initialize mask generator
    mask_gen = AfricanFabricMaskGenerator(img.shape)
    mask = None
    
    # Try to load existing mask
    if load_existing_mask:
        mask, metadata = mask_gen.load_mask(image_path)
    
    # Create new mask if none exists
    if mask is None:
        print("1. Draw polygon around the agbada/fabric...")
        drawer = EnhancedPolygonDrawer(img)
        all_polygons = drawer.draw()
        
        if not all_polygons:
            print("Error: Need at least 3 points for polygon")
            return None
        
        # Generate advanced mask
        mask = mask_gen.generate_advanced_mask(
            all_polygons, 
            dilate_iterations=1, 
            edge_preserve_strength=0.8, 
            feather_size=3
        )
        
        # Save mask for future use
        mask_gen.save_mask_with_custom_image(mask, image_path, all_polygons)
    
    # Initialize recolorer
    recolorer = AfricanClothingStyleRecolorer(img, mask)
    fabric_analysis = None
    
    # Analyze fabric structure if advanced analyzer is enabled
    if use_advanced_analyzer:
        print("2. Analyzing fabric structure with advanced analyzer...")
        fabric_analysis = recolorer.fabric_analyzer.analyze_fabric_structure(img, mask)
        print("   - Detected embroidery features")
        print("   - Analyzed texture patterns")
        print("   - Mapped fold structures")
        print("   - Segmented fabric regions")
    else:
        print("2. Using standard fabric analysis...")
        # Create minimal analysis for compatibility
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fabric_analysis = {
            'fabric_regions': recolorer.fabric_analyzer._get_empty_regions(mask),
            'color_variation': {'brightness_map': gray},
            'embroidery_features': {'edges': np.zeros_like(gray)},
            'fold_analysis': {'folds_combined': np.zeros_like(gray)}
        }
    
    print("3. Generating recolored versions...")
    
    # Create results
    results = {}
    results_trimmed = {}
    
    # Generate recolored images
    for color_name, rgb in target_colors:
        if use_advanced_analyzer and fabric_analysis:
            # Use advanced method with fabric analysis
            result = recolorer._advanced_fabric_aware_method(rgb, fabric_analysis)
        else:
            # Use standard fabric-aware method
            result = recolorer._fabric_aware_method(rgb)
        
        results[color_name] = result
        
        # Apply trimming if requested
        if trim_images:
            trimmed = ImageProcessor.trim_recolored_image(img, result, mask, trim_strength=0.95)
            if alpha_channel is not None:
                trimmed = ImageProcessor.fill_transparent_background(trimmed, alpha_channel)
            results_trimmed[f"{color_name} (Trimmed)"] = trimmed
    
    # Fixed visualization with correct dimensions
    print("4. Displaying results...")
    
    # Calculate grid dimensions
    total_images = 2 + len(target_colors)  # original + mask + recolored versions
    if trim_images:
        total_images += len(target_colors)  # add trimmed versions
    
    cols = min(4, total_images)
    rows = (total_images + cols - 1) // cols
    
    plt.figure(figsize=(20, 5 * rows))
    
    # Show original
    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title("Original Agbada", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Show mask visualization
    plt.subplot(rows, cols, 2)
    # mask_overlay = img.copy()
    # mask_overlay[mask > 128] = mask_overlay[mask > 128] * 0.7 + np.array([255, 0, 0]) * 0.3
    mask_overlay = img.copy().astype(np.uint8)  # Ensure uint8
    red_overlay = np.array([255, 0, 0], dtype=np.uint8)
    
    # Create overlay with proper broadcasting
    mask_bool = mask > 128
    mask_overlay[mask_bool] = (mask_overlay[mask_bool].astype(np.float32) * 0.7 + 
                              red_overlay.astype(np.float32) * 0.3).astype(np.uint8)
    
    plt.imshow(mask_overlay.astype(np.uint8))
    plt.title("Mask Overlay", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Show recolored versions
    plot_idx = 3
    for color_name, result in results.items():
        if plot_idx <= rows * cols:
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(result)
            plt.title(f"{color_name}", fontsize=11, fontweight='bold')
            plt.axis('off')
            plot_idx += 1
    
    # Show trimmed versions if enabled
    if trim_images:
        for color_name, result in results_trimmed.items():
            if plot_idx <= rows * cols:
                plt.subplot(rows, cols, plot_idx)
                plt.imshow(result)
                plt.title(f"{color_name}", fontsize=11, fontweight='bold')
                plt.axis('off')
                plot_idx += 1
    
    plt.tight_layout()
    plt.get_current_fig_manager().toolbar_visible = True  # Ensure toolbar is visible
    plt.show()
    
    # Show fabric analysis if advanced analyzer is used
    if use_advanced_analyzer and fabric_analysis:
        print("5. Fabric analysis visualization...")
        plt.figure(figsize=(16, 8))
        
        plt.subplot(2, 4, 1)
        plt.imshow(fabric_analysis['fabric_regions']['main_fabric'], cmap='gray')
        plt.title("Main Fabric")
        plt.axis('off')
        
        plt.subplot(2, 4, 2)  
        plt.imshow(fabric_analysis['fabric_regions']['embroidery'], cmap='gray')
        plt.title("Embroidery/Details")
        plt.axis('off')
        
        plt.subplot(2, 4, 3)
        plt.imshow(fabric_analysis['fabric_regions']['seams'], cmap='gray')
        plt.title("Seams/Folds")
        plt.axis('off')
        
        plt.subplot(2, 4, 4)
        plt.imshow(fabric_analysis['embroidery_features']['edges'], cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')
        
        plt.subplot(2, 4, 5)
        plt.imshow(fabric_analysis['color_variation']['brightness_map'], cmap='gray')
        plt.title("Brightness Map")
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        plt.imshow(fabric_analysis['fold_analysis']['folds_combined'], cmap='gray')
        plt.title("Fold Analysis")
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        plt.imshow(fabric_analysis['fabric_regions']['shadows'], cmap='gray')
        plt.title("Shadow Regions")
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        plt.imshow(fabric_analysis['fabric_regions']['highlights'], cmap='gray')
        plt.title("Highlight Regions")
        plt.axis('off')
        
        plt.tight_layout()
        plt.get_current_fig_manager().toolbar_visible = True  # Ensure toolbar is visible
        plt.show()
    
    # Save images in high resolution
    if save_images:
        print("6. Saving high-resolution images...")
        base_name = Path(image_path).stem
        
        # Prepare all images for saving
        all_images = {
            "original": img,
            "mask_overlay": mask_overlay.astype(np.uint8)
        }
        all_images.update(results)
        
        if trim_images:
            all_images.update(results_trimmed)
        
        # Save images
        saved_files = ImageProcessor.save_images_high_res(
            all_images, 
            base_name, 
            output_dir="fabric_recoloring_output",
            save_ultra_high_res=True,
        )
        
        print(f"Saved {len(saved_files)} image files")
    
    print("=== PROCESSING COMPLETE ===")
    
    return {
        'original_image': img,
        'mask': mask,
        'recolored_results': results,
        'trimmed_results': results_trimmed if trim_images else None,
        'fabric_analysis': fabric_analysis,
        'saved_files': saved_files if save_images else None
    }

