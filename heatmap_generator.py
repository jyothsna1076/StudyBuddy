import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.accumulation_map = np.zeros((height, width), dtype=np.float32)

    def add_struggle_point(self, x, y, mode="cursor", intensity=1.0,scroll_y=0, view_h=850):
        """
        mode "cursor": Highlights the entire horizontal line/row.
        mode "gaze": Highlights the entire page.
        """
        if mode == "cursor":
            # Highlight the entire width at that specific Y (and a small padding for the line height)
            y_start = max(0, y - 10)
            y_end = min(self.height, y + 10)
            self.accumulation_map[y_start:y_end, :] += intensity
        
        elif mode == "gaze":
            # Add intensity to every single pixel on the page
            y_start = scroll_y
            y_end = min(self.height, scroll_y + view_h)
            self.accumulation_map[y_start:y_end, :] += intensity # Lower intensity since it hits everything

    def get_heatmap_overlay(self):
        """Normalizes the accumulated points and converts to a heatmap color scheme."""
        max_val = np.max(self.accumulation_map)
        if max_val < 0.1: # If there's barely any struggle, return a blank mask
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # NORMALIZE: This keeps the scale consistent
        # Instead of dividing by max_val (which causes the rainbow shift), 
        # we divide by a fixed 'threshold' to keep colors stable.
        normalized_map = np.clip((self.accumulation_map / 10.0) * 255, 0, 255).astype(np.uint8)
        
        # APPLY COLORMAP
        heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        
        # CLEANUP: Remove the "blue/green" (low struggle) background
        # This prevents the rainbow effect and only shows the hot (red/yellow) parts
        mask = normalized_map < 50  # Adjust this number to hide more/less of the 'cool' colors
        heatmap[mask] = [0, 0, 0]
        
        return heatmap