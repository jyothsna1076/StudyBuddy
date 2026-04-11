import cv2

class MouseTracker:
    def __init__(self, window_name):
        self.x = 0
        self.y = 0
        self.window_name = window_name
        # Register the callback immediately upon initialization
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event_handler)

    def _mouse_event_handler(self, event, x, y, flags, param):
        """Internal callback to update coordinates."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y

    def get_position(self):
        """Returns the current mouse coordinates."""
        return self.x, self.y