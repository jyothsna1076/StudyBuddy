"""
MouseTracker Module
Acts as a hardware input bridge, utilizing OpenCV's HighGUI framework to capture 
physical mouse movements within the application window.
"""

import cv2

class MouseTracker:
    def __init__(self, window_name):
        # Initialize default spatial coordinates
        self.x = 0
        self.y = 0
        self.window_name = window_name
        
        # --- EVENT LISTENER REGISTRATION ---
        # The window must be explicitly named and created in memory before attaching a callback.
        cv2.namedWindow(self.window_name)
        
        # Bind the internal handler function to the OpenCV window.
        # This establishes an event-driven connection to the operating system's hardware interrupts.
        cv2.setMouseCallback(self.window_name, self._mouse_event_handler)

    def _mouse_event_handler(self, event, x, y, flags, param):
        """
        Internal callback function strictly structured to match OpenCV's C++ backend requirements.
        This function is executed asynchronously whenever the OS detects mouse activity in the window.
        
        Parameters:
        - event: The specific type of mouse action (click, scroll, move, etc.).
        - x, y: The Cartesian coordinates of the mouse relative to the window's top-left corner.
        - flags, param: Additional states (like holding CTRL/SHIFT) not utilized in this implementation.
        """
        # Filter the event stream to exclusively capture spatial translation (movement)
        if event == cv2.EVENT_MOUSEMOVE:
            # Update the class instance variables with the latest coordinates
            self.x = x
            self.y = y

    def get_position(self):
        """
        Provides a non-blocking retrieval mechanism for the main processing loop.
        Returns the most recently recorded coordinates as a tuple.
        """
        return self.x, self.y