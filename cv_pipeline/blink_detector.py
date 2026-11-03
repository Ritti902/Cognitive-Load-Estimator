import numpy as np
  
def calculate_ear(landmarks, eye_indices):
    '''
    EAR = Eye Aspect Ratio
    eye_indices = list of 6 landmark numbers for one eye
    Returns a float — high means open, low means closed
    '''
      # Get (x, y) coordinates for each of the 6 eye landmarks
    p = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
  
      # Calculate vertical distances (how tall the eye is)
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
  
      # Calculate horizontal distance (how wide the eye is)
    h = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    if h <= 1e-8:
        return 0.0
  
      # EAR formula
    ear = (v1 + v2) / (2.0 * h)
    return ear
  
  
class BlinkCounter:
    def __init__(self):
        self.blink_count = 0
        self.ear_below_threshold = False
        self.EAR_THRESHOLD = 0.25   # below this = eye closed
  
    def update(self, left_ear, right_ear):
        '''Call this every frame. Returns blink_count so far.'''
        avg_ear = (left_ear + right_ear) / 2.0
  
        if avg_ear < self.EAR_THRESHOLD:
            self.ear_below_threshold = True   # eye is closing
        else:
            if self.ear_below_threshold:      # eye just reopened
                self.blink_count += 1         # that was one blink!
            self.ear_below_threshold = False
  
        return self.blink_count
  
    def get_blinks_per_minute(self, elapsed_seconds):
        '''Convert total blinks to per-minute rate'''
        if elapsed_seconds == 0:
            return 0
        return (self.blink_count / elapsed_seconds) * 60
