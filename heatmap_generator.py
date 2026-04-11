import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.accumulation_map = np.zeros((height, width), dtype=np.float32)

    def add_struggle_point(self, x, y, mode="cursor", intensity=1.0):
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
            self.accumulation_map += (intensity * 0.05) # Lower intensity since it hits everything

    def get_heatmap_overlay(self):
        max_val = np.max(self.accumulation_map)
        if max_val == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # Standard normalization and colormap logic
        normalized_map = np.clip((self.accumulation_map / max_val) * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        heatmap[normalized_map == 0] = [0, 0, 0]
        return heatmap