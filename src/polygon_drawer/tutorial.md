# Multi-Segment Annotation Tool Tutorial

## 1. Overview

This tool provides an interactive window to draw multiple, disconnected, closed polygons (called "segments") over an image. It's designed for tasks like creating multiple masks for different objects or areas of interest in a single session.

The main features include drawing with mouse clicks, support for multiple independent shapes, color-coding for each shape, and a full suite of controls including zoom, pan, undo, and clear.

---

## 2. Core Features

-   **Multi-Segment Drawing**: Draw several independent polygons on one image.
-   **Interactive UI**: Simple and intuitive interface with buttons and status updates.
-   **Navigation Support**: Use the standard `matplotlib` toolbar to Zoom and Pan the image without interrupting your drawing.
-   **Color-Coded Segments**: Each new polygon segment is automatically assigned a different color for easy identification.
-   **Full Control**: Undo points, clear the entire canvas, or finish your work using on-screen buttons or keyboard shortcuts.
-   **Status Bar**: Get real-time feedback on your progress, including the number of points and current mode.

---

## 3. Getting Started: Example Code

To use the tool, you need to have `matplotlib` and `numpy` installed.

```bash
pip install matplotlib numpy pillow
```

Then, you can run the tool with your own image like this:

```python
import numpy as np
from PIL import Image
# Make sure the EnhancedPolygonDrawer class code is in the same file or imported
# from polygon_drawer import EnhancedPolygonDrawer 

# 1. Load your image
try:
    image_path = 'path/to/your/image.jpg' # <-- CHANGE THIS
    image = np.array(Image.open(image_path))
except FileNotFoundError:
    print("Image not found. Using a black dummy image for demonstration.")
    image = np.zeros((600, 800, 3), dtype=np.uint8)


# 2. Create an instance of the drawer
# This will open the interactive window
drawer = EnhancedPolygonDrawer(image)

# 3. Start the drawing process
# The code will pause here until you close the window or click "Finish"
all_drawn_polygons = drawer.draw()

# 4. Process the results
if all_drawn_polygons:
    print("\n✅ Drawing finished. The following segments were created:")
    for i, points in enumerate(all_drawn_polygons):
        print(f"  - Segment {i+1}: {len(points)} vertices")
else:
    print("\n❌ Window was closed before any polygons were completed.")

```

---

## 4. Step-by-Step Guide

### The User Interface

When you run the script, a window will pop up displaying your image.
-   **Main Canvas**: Your image where you will draw.
-   **Toolbar**: The standard Matplotlib navigation toolbar at the top/bottom of the window for Zoom and Pan.
-   **Control Buttons**: A set of buttons at the bottom for controlling the drawing process (`Undo`, `Clear All`, `Finish`, etc.).
-   **Status Bar**: A text box in the bottom-left corner providing instructions and status updates.

### Drawing Your First Segment

1.  **Add Points**: **Left-click** on the image to place the vertices of your polygon.
2.  **Close the Segment**: Once you have at least 3 points, **Right-click** anywhere on the image. This will automatically connect the last point to the first, closing the shape and filling it with a semi-transparent color.

### Adding More Segments

1.  After you have closed a segment, it is considered complete.
2.  To start drawing a new, separate polygon, click the **New Segment** button.
3.  The status bar will update, and you can begin left-clicking again to place points for the new segment.
4.  This new segment will have a different color. Repeat this process for as many segments as you need.

### Navigating the Image

-   **Zoom/Pan Toolbar**: Use the magnifying glass and cross-arrow icons in the toolbar to zoom in on details or pan across the image. Drawing will automatically resume when you are done.
-   **Toggle Mode**: Alternatively, click the **Toggle Mode** button (or press `T`). This switches to "NAVIGATION" mode, preventing accidental point placement while you use the toolbar. Click it again to return to "DRAWING" mode.

---

## 5. Controls Reference

| Control | Mouse Action | Keyboard Shortcut | On-Screen Button | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Add Point** | `Left Click` | - | - | Adds a vertex to the current polygon segment. |
| **Close Segment** | `Right Click` | - | - | Closes the current shape (requires 3+ points). |
| **Undo Point** | - | `U` | `Undo` | Removes the last point added to the current segment. |
| **Clear All** | - | `C` | `Clear All` | Deletes all completed segments and current progress. |
| **Start New Segment** | - | - | `New Segment` | Begins a new, disconnected polygon. |
| **Finish & Exit** | - | `F` | `Finish` | Closes the current segment (if valid) and closes the window. |
| **Toggle Mode** | - | `T` | `Toggle Mode` | Switches between "Drawing" and "Navigation" modes. |
| **Cancel & Exit** | - | `Esc` | (Close Window) | Exits the program immediately without saving progress. |

---

## 6. Understanding the Output

The `drawer.draw()` method returns a **list of lists**.

-   The outer list represents all the segments you drew.
-   Each inner list contains the `(x, y)` coordinates (as tuples) for the vertices of one segment.

**Example Output:** `[ [(x1, y1), (x2, y2), ...], [(x_a, y_a), (x_b, y_b), ...] ]`

-   `[(x1, y1), (x2, y2), ...]` are the points for the first polygon.
-   `[(x_a, y_a), (x_b, y_b), ...]` are the points for the second polygon.

If you close the window before finishing any segments, the method returns an empty list `[]`.