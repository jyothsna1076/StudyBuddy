"""
HeatmapGenerator Module
Handles the accumulation of 'struggle' intensity data over time and renders it 
as a color-coded overlay using OpenCV matrix operations.
"""

import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Initialize a 2D matrix of zeros matching the document dimensions.
        # np.float32 is used instead of uint8 to allow continuous accumulation 
        # over a long session without integer overflow.
        self.accumulation_map = np.zeros((height, width), dtype=np.float32)

    def add_struggle_point(self, x, y, mode="cursor", intensity=1.0, scroll_y=0, view_h=850):
        """
        Injects heat/intensity into the accumulation matrix based on the user's focus mode.
        
        mode "cursor": Highlights a specific horizontal line/row representing reading a sentence.
        mode "gaze": Highlights the entire visible viewport.
        """
        if mode == "cursor":
            # Highlight the entire width of the page at the specific Y coordinate.
            # Padding (+/- 10 pixels) represents an approximate line height.
            y_start = max(0, y - 10)
            y_end = min(self.height, y + 10)
            
            # Vectorized addition across the entire horizontal slice of the matrix
            self.accumulation_map[y_start:y_end, :] += intensity
        
        elif mode == "gaze":
            # Add intensity to the currently visible section of the document.
            # This is used when eye-tracking indicates overall page-level confusion.
            y_start = scroll_y
            y_end = min(self.height, scroll_y + view_h)
            
            # Apply intensity across the viewport area
            self.accumulation_map[y_start:y_end, :] += intensity 

    def get_heatmap_overlay(self):
        """
        Normalizes the accumulated float matrix, applies a scientific colormap, 
        and masks out low-intensity values to create a clean overlay.
        """
        max_val = np.max(self.accumulation_map)
        
        # Optimization: If the document has barely any accumulated struggle,
        # return a blank, zeroed-out image immediately to save processing power.
        if max_val < 0.1: 
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        # --- NORMALIZATION STRATEGY ---
        # We divide by a fixed constant (10.0) rather than dividing by 'max_val'.
        # Dividing by max_val causes a "rainbow shift" where the hottest point is 
        # ALWAYS red, even if the user only struggled for 1 second. 
        # Using a fixed threshold ensures color scales consistently (e.g., 10 ticks = red).
        normalized_map = np.clip((self.accumulation_map / 10.0) * 255, 0, 255).astype(np.uint8)
        
        # --- APPLY COLORMAP ---
        # COLORMAP_JET transitions from Blue (Low) -> Green (Medium) -> Red (High)
        heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        
        # --- ALPHA MASKING / CLEANUP ---
        # JET maps '0' values to dark blue. We want areas with low struggle to be transparent 
        # so the user can read the underlying text. 
        # We create a boolean mask where intensity is below a strict threshold.
        mask = normalized_map < 50  
        
        # Apply black [0, 0, 0] to the masked areas. In the main.py rendering loop,
        # cv2.addWeighted will treat this black as empty space, allowing text to show through.
        heatmap[mask] = [0, 0, 0]
        
        return heatmap