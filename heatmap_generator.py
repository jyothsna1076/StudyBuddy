import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # A blank matrix to accumulate "struggle" gaze points
        self.accumulation_map = np.zeros((height, width), dtype=np.float32)

    def add_struggle_point(self, x, y, intensity=1.0, radius=50):
        """Adds a localized Gaussian blob to the accumulation map."""
        # Create a meshgrid to generate a circle
        Y, X = np.ogrid[:self.height, :self.width]
        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
        
        # Add intensity within the radius
        mask = dist_from_center <= radius
        self.accumulation_map[mask] += intensity

    def get_heatmap_overlay(self):
        """Normalizes the accumulated points and converts to a heatmap color scheme."""
        # Normalize between 0 and 255
        max_val = np.max(self.accumulation_map)
        if max_val == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        normalized_map = (self.accumulation_map / max_val) * 255
        normalized_map = normalized_map.astype(np.uint8)
        
        # Apply the Jet colormap (Blue = low struggle, Red = high struggle)
        heatmap = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
        
        # Black out areas with absolutely zero data to make them transparent later
        heatmap[normalized_map == 0] = [0, 0, 0]
        
        return heatmap