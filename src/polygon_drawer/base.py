import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle, Polygon as MPLPolygon
import numpy as np
from abc import ABC

class BaseAdvancedPolygonDrawer(ABC):
    """
    A class to draw a single, closed polygon on an image using matplotlib.
    Supports zooming, panning, undoing, and clearing.
    """
    def __init__(self, image, point_size=8, line_width=2):
        self.image = image
        self.points = []
        self.point_size = point_size
        self.line_width = line_width
        self.closed = False
        self.drawing_mode = True
        
        # Create figure with toolbar for zoom/pan
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.subplots_adjust(bottom=0.15)  # Make room for buttons
        
        # Display image
        self.ax.imshow(self.image)
        self.ax.set_title("Polygon Drawer\nLEFT CLICK: Add point | RIGHT CLICK: Close polygon", 
                         fontsize=12, fontweight='bold')
        
        # Store visual elements
        self.point_circles = []
        self.line_segments = []
        self.preview_line = None
        self.polygon_patch = None
        
        # Mouse tracking for preview
        self.mouse_pos = None
        
        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add control buttons
        self.setup_control_buttons()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=11, 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        self.update_status()
        
        print("🎯 ADVANCED POLYGON DRAWER READY")

    def setup_control_buttons(self):
        """Create and set up control buttons."""
        ax_undo = plt.axes([0.1, 0.02, 0.08, 0.05])
        ax_clear = plt.axes([0.2, 0.02, 0.08, 0.05])
        ax_finish = plt.axes([0.3, 0.02, 0.08, 0.05])
        ax_toggle = plt.axes([0.4, 0.02, 0.12, 0.05])
        
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_finish = Button(ax_finish, 'Finish')
        self.btn_toggle = Button(ax_toggle, 'DRAWING Mode')
        
        self.btn_undo.on_clicked(self.undo_point)
        self.btn_clear.on_clicked(self.clear_all)
        self.btn_finish.on_clicked(self.finish_polygon)
        self.btn_toggle.on_clicked(self.toggle_drawing_mode)
        self.btn_toggle.color = 'lightgreen'

    def get_status_text(self):
        """Get the current status text."""
        if self.closed:
            return f"✅ POLYGON COMPLETED | {len(self.points)} points"
        points_drawn = len(self.points)
        if points_drawn == 0:
            return "🎯 START DRAWING | Left-click to add first point"
        
        mode_text = "DRAWING" if self.drawing_mode else "NAVIGATION"
        status = f"📍 {mode_text} MODE | {points_drawn} points"
        if points_drawn < 3:
            status += f" | Need {3-points_drawn} more to close"
        else:
            status += " | Right-click to close"
        return status

    def on_motion(self, event):
        if not event.inaxes == self.ax or not self.drawing_mode or self.closed or len(self.points) == 0:
            return
        self.mouse_pos = (event.xdata, event.ydata)
        if self.mouse_pos[0] is not None:
            self.update_preview_line()

    def update_preview_line(self):
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
        
        last_point = self.points[-1]
        self.preview_line, = self.ax.plot([last_point[0], self.mouse_pos[0]], 
                                         [last_point[1], self.mouse_pos[1]], 
                                         'r--', linewidth=1)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if not event.inaxes == self.ax or not self.drawing_mode or event.xdata is None:
            return
        if event.button == 1:
            self.add_point(event.xdata, event.ydata)
        elif event.button == 3 and len(self.points) > 2:
            self.close_polygon()

    def add_point(self, x, y):
        if self.closed: return
        new_point = (int(x), int(y))
        self.points.append(new_point)
        
        circle = Circle(new_point, self.point_size, color='red', zorder=5)
        self.ax.add_patch(circle)
        self.point_circles.append(circle)
        
        if len(self.points) > 1:
            prev_point = self.points[-2]
            line, = self.ax.plot([prev_point[0], new_point[0]], 
                               [prev_point[1], new_point[1]], 'r-', lw=self.line_width)
            self.line_segments.append(line)
        
        self.update_status()
        self.fig.canvas.draw_idle()

    def close_polygon(self):
        if len(self.points) < 3 or self.closed: return
        
        line, = self.ax.plot([self.points[-1][0], self.points[0][0]], 
                           [self.points[-1][1], self.points[0][1]], 'g-', lw=self.line_width)
        self.line_segments.append(line)
        
        self.polygon_patch = MPLPolygon(np.array(self.points), alpha=0.2, facecolor='green', edgecolor='green')
        self.ax.add_patch(self.polygon_patch)
        
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
            
        self.closed = True
        self.update_status()
        self.fig.canvas.draw_idle()
        print(f"🎉 POLYGON COMPLETED with {len(self.points)} points!")

    def undo_point(self, event=None):
        if len(self.points) == 0 or self.closed: return
            
        self.points.pop()
        self.point_circles[-1].remove()
        self.point_circles.pop()
            
        if len(self.line_segments) > 0:
            self.line_segments[-1].remove()
            self.line_segments.pop()
            
        if self.preview_line:
            self.preview_line.remove()
            self.preview_line = None
            
        self.update_status()
        self.fig.canvas.draw_idle()

    def clear_all(self, event=None):
        for artist in self.point_circles + self.line_segments:
            artist.remove()
        if self.polygon_patch:
            self.polygon_patch.remove()
        if self.preview_line:
            self.preview_line.remove()
            
        self.points, self.point_circles, self.line_segments = [], [], []
        self.preview_line, self.polygon_patch = None, None
        self.closed = False
        
        self.update_status()
        self.fig.canvas.draw_idle()
        print("🗑️ All points cleared!")

    def finish_polygon(self, event=None):
        if len(self.points) >= 3:
            if not self.closed: self.close_polygon()
            plt.close(self.fig)
        else:
            print("⚠️ Need at least 3 points to finish.")

    def toggle_drawing_mode(self, event=None):
        self.drawing_mode = not self.drawing_mode
        mode_text = "DRAWING" if self.drawing_mode else "NAVIGATION"
        color = "lightgreen" if self.drawing_mode else "orange"
        self.btn_toggle.label.set_text(f"{mode_text} Mode")
        self.btn_toggle.color = color
        self.update_status()
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'u': self.undo_point()
        elif event.key == 'c': self.clear_all()
        elif event.key == 'f': self.finish_polygon()
        elif event.key == 't': self.toggle_drawing_mode()
        elif event.key == 'escape': plt.close(self.fig)

    def update_status(self):
        self.status_text.set_text(self.get_status_text())

    def draw(self):
        """Main method to start drawing and return points."""
        print("\n🚀 Starting Polygon Drawer... \n    'f' to finish | 'u' to undo | 'c' to clear | 'esc' to cancel")
        plt.show()
        return self.points




