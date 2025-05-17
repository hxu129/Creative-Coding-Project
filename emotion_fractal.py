import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Valence-Arousal map for emotions (Valence: -1 to 1, Arousal: -1 to 1)
# These are approximate values and can be tuned.
EMOTION_VA_MAP = {
    'happy':    (0.8, 0.6),   # High Valence, Moderate-High Arousal
    'sad':      (-0.7, -0.4),  # Low Valence, Low-Moderate Arousal
    'angry':    (-0.6, 0.7),   # Low Valence, High Arousal
    'fear':     (-0.5, 0.8),   # Low Valence, High Arousal
    'surprise': (0.4, 0.7),   # Neutral-High Valence, High Arousal
    'disgust':  (-0.8, 0.3),   # Low Valence, Moderate Arousal
    'neutral':  (0.0, 0.0)    # Neutral Valence, Low Arousal
}

# Factors for V/A influence on Julia C parameter
# These can be tuned to adjust sensitivity
AROUSAL_TO_CREAL_FACTOR = 0.1  # How much arousal affects the real part of C
VALENCE_TO_CIMAG_FACTOR = 0.1  # How much valence affects the imaginary part of C
INITIAL_VIEW_RANGE = 3.0 # Initial x and y range (e.g., -1.5 to 1.5 if range is 3.0)

class EmotionFractalGenerator:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        self.max_iter = 100 # Default, will be changed by arousal
        
        # Define base parameters for each emotion (Julia set 'c' parameter)
        self.emotion_parameters = {
            'happy': {'c': -0.4 + 0.6j},    # Douady's Rabbit - organic, lively
            'sad': {'c': -0.75 + 0.11j},    # San Marco Dragon - complex, detailed
            'angry': {'c': 0.25 + 0.52j},   # Douady's Elephant - dense, intense
            'fear': {'c': -0.1 + 0.8j},     # Feather-like - delicate, intricate
            'surprise': {'c': 0.3 + 0.5j},  # Spiral pattern - dynamic
            'disgust': {'c': -0.8 - 0.2j},  # Complex structure - detailed
            'neutral': {'c': 0j}            # Dendrite - balanced, calm
        }
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize emotion variables
        self.last_emotion_time = 0
        self.emotion_detection_interval = 2.0  # Detect emotions every 2 seconds
        self.current_emotion = 'neutral' # Start with neutral
        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_colormap_name = 'magma' # Default colormap
        self.current_julia_c = 0j # Initialize current C for Julia set

        # Zoom parameters
        self.zoom_factor_per_second = 0.80 # Zoom in to 80% of the view per second
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.view_xmin = -INITIAL_VIEW_RANGE / 2
        self.view_xmax = INITIAL_VIEW_RANGE / 2
        self.view_ymin = -INITIAL_VIEW_RANGE / 2
        self.view_ymax = INITIAL_VIEW_RANGE / 2
        self.last_frame_time = time.time()
        
        # Initialize with a default fractal state
        self.update_fractal_parameters() # Calculate initial C, max_iter, colormap
        self.current_fractal = self.julia(
            c=self.current_julia_c, 
            max_iter=self.max_iter, 
            xmin=self.view_xmin, xmax=self.view_xmax, 
            ymin=self.view_ymin, ymax=self.view_ymax
        )
        
    def get_emotion_va(self, emotion):
        """Get valence and arousal for the given emotion."""
        return EMOTION_VA_MAP.get(emotion, EMOTION_VA_MAP['neutral'])

    def get_fractal_max_iter(self, arousal):
        """Map arousal (-1 to 1) to max_iter (e.g., 30 to 150)."""
        # Normalize arousal from -1 to 1 to 0 to 1
        normalized_arousal = (arousal + 1) / 2
        min_iter = 30
        max_iter_range = 120 # Max iterations will be min_iter + max_iter_range
        return int(min_iter + normalized_arousal * max_iter_range)

    def get_fractal_colormap(self, valence):
        """Map valence (-1 to 1) to a Matplotlib colormap name."""
        if valence < -0.5:
            return 'Blues_r'  # Very Negative Valence -> cool, dark blues
        elif valence < -0.1:
            return 'coolwarm' # Negative Valence -> transitioning from blue to red via white
        elif valence <= 0.1:
            return 'viridis'  # Neutral Valence -> balanced green/purple
        elif valence <= 0.5:
            return 'plasma'   # Positive Valence -> warm oranges, yellows
        else:
            return 'YlOrRd'   # Very Positive Valence -> hot yellows, oranges, reds
        
    def update_fractal_parameters(self):
        """Updates all fractal-related parameters based on current emotion, V/A."""
        # Get Valence and Arousal for the current emotion
        self.current_valence, self.current_arousal = self.get_emotion_va(self.current_emotion)
        
        # Determine base c from discrete emotion
        base_c_params = self.emotion_parameters.get(self.current_emotion, self.emotion_parameters['neutral'])
        base_c = base_c_params['c']
        # If the emotion has a suggested zoom center, use it. Otherwise, C can be a good center for some Julias.
        self.zoom_center_x = base_c_params.get('zoom_center_x', base_c.real)
        self.zoom_center_y = base_c_params.get('zoom_center_y', base_c.imag)

        # Modulate base_c with V/A
        effective_c_real = base_c.real + self.current_arousal * AROUSAL_TO_CREAL_FACTOR
        effective_c_imag = base_c.imag + self.current_valence * VALENCE_TO_CIMAG_FACTOR
        self.current_julia_c = complex(effective_c_real, effective_c_imag)
        
        # Get max_iter from arousal
        self.max_iter = self.get_fractal_max_iter(self.current_arousal)
        
        # Get colormap from valence
        self.current_colormap_name = self.get_fractal_colormap(self.current_valence)

        print(f"Params Updated - Emotion: {self.current_emotion}, V: {self.current_valence:.2f}, A: {self.current_arousal:.2f}, C: {self.current_julia_c}, Iter: {self.max_iter}, CMap: {self.current_colormap_name}")
        
    def julia(self, c, max_iter, xmin, xmax, ymin, ymax):
        """Generate Julia set fractal."""
        # print(f"Julia called with c={c}, max_iter={max_iter}, xmin={xmin:.3f}, xmax={xmax:.3f}, ymin={ymin:.3f}, ymax={ymax:.3f}") # Debug zoom
        x_pixels = np.linspace(xmin, xmax, self.width)
        y_pixels = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x_pixels, y_pixels)
        Z = X + 1j * Y
        
        divtime = max_iter + np.zeros(Z.shape, dtype=int)
        
        for i in range(max_iter):
            Z = Z**2 + c
            diverge = Z*np.conj(Z) > 2**2
            div_now = diverge & (divtime == max_iter)
            divtime[div_now] = i
            Z[diverge] = 2
            
        # Normalize the values for better visualization
        divtime = divtime / max_iter
        return divtime
        
    def create_emotion_based_fractal(self):
        """Create and display emotion-based fractal."""
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Two subplots: 1 for camera, 1 for fractal
        fig.canvas.manager.set_window_title('Emotion-Driven Fractals')
        
        # Initialize plots
        camera_plot = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        ax1.set_title('Camera Feed & Emotion Detection')
        ax1.axis('off')

        self.fractal_plot = ax2.imshow(self.current_fractal, cmap=self.current_colormap_name)
        ax2.set_title(f"Fractal: {self.current_emotion.capitalize()} (V:{self.current_valence:.2f}, A:{self.current_arousal:.2f}) Iter:{self.max_iter}")
        ax2.axis('off')
        
        def update(frame_count):
            current_time_anim = time.time()
            delta_time = current_time_anim - self.last_frame_time
            self.last_frame_time = current_time_anim

            ret, frame = self.cap.read()
            if not ret:
                return [camera_plot, self.fractal_plot]
                
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Process emotions if enough time has passed
            current_time = time.time()
            if len(faces) > 0 and (current_time - self.last_emotion_time) > self.emotion_detection_interval:
                try:
                    # Get the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Analyze emotions
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    # Get top emotion
                    emotions = sorted(result[0]['emotion'].items(), key=lambda x: x[1], reverse=True)
                    new_emotion = emotions[0][0]
                    
                    if self.current_emotion != new_emotion:
                        self.current_emotion = new_emotion
                        # Reset view to initial full range when emotion changes, to avoid over-zooming into unrelated areas
                        self.view_xmin = -INITIAL_VIEW_RANGE / 2
                        self.view_xmax = INITIAL_VIEW_RANGE / 2
                        self.view_ymin = -INITIAL_VIEW_RANGE / 2
                        self.view_ymax = INITIAL_VIEW_RANGE / 2
                        self.update_fractal_parameters() # This updates V/A, C, max_iter, colormap, zoom_center
                        # self.last_emotion_time = current_time # Moved this to after potential long calculation
                    
                    # Update V/A even if dominant emotion hasn't changed, for subtle C shifts
                    # If we want C to change even without dominant emotion change, call update_fractal_parameters() here too
                    # For now, V/A for C modulation uses the V/A from the last dominant emotion change for stability
                    # but max_iter and colormap also get updated inside update_fractal_parameters()

                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display emotion information on camera feed
                    for i, (emotion_name, confidence) in enumerate(emotions[:3]): # emotion_name for clarity
                        text = f"#{i+1}: {emotion_name.capitalize()} ({confidence:.1f}%)"
                        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # Display V/A on camera feed (using the latest V/A from current_emotion for display consistency)
                    display_valence, display_arousal = self.get_emotion_va(self.current_emotion) 
                    va_text = f"V: {display_valence:.2f}, A: {display_arousal:.2f}"
                    cv2.putText(frame, va_text, (10, 30 + 3*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.last_emotion_time = current_time # Update emotion time after all processing for this cycle

                except Exception as e:
                    print(f"Error in emotion detection/fractal update: {e}")
            
            # Continuous Zoom Logic
            current_range_x = self.view_xmax - self.view_xmin
            current_range_y = self.view_ymax - self.view_ymin
            
            # Calculate zoom amount based on delta_time
            # zoom_factor_this_frame = 1 - (1 - self.zoom_factor_per_second) * delta_time
            # A more stable way for time-based zoom factor for iterative multiplication:
            zoom_multiplier = self.zoom_factor_per_second ** delta_time

            new_range_x = current_range_x * zoom_multiplier
            new_range_y = current_range_y * zoom_multiplier
            
            # Adjust xmin, xmax, ymin, ymax to zoom towards zoom_center
            self.view_xmin = self.zoom_center_x - (self.zoom_center_x - self.view_xmin) * zoom_multiplier
            self.view_xmax = self.zoom_center_x + (self.view_xmax - self.zoom_center_x) * zoom_multiplier
            self.view_ymin = self.zoom_center_y - (self.zoom_center_y - self.view_ymin) * zoom_multiplier
            self.view_ymax = self.zoom_center_y + (self.view_ymax - self.zoom_center_y) * zoom_multiplier

            # Regenerate fractal every frame due to zoom and potential continuous C changes
            # (If C is only changing on discrete emotion change, this is mainly for zoom)
            self.current_fractal = self.julia(
                c=self.current_julia_c, 
                max_iter=self.max_iter, 
                xmin=self.view_xmin, xmax=self.view_xmax, 
                ymin=self.view_ymin, ymax=self.view_ymax
            )
            self.fractal_plot.set_array(self.current_fractal)
            self.fractal_plot.set_cmap(self.current_colormap_name) # Update colormap if it changed

            camera_plot.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax2.set_title(f"Fractal: {self.current_emotion.capitalize()} (V:{self.current_valence:.2f}, A:{self.current_arousal:.2f}) Iter:{self.max_iter}")

            return [camera_plot, self.fractal_plot]
        
        # Create animation
        anim = FuncAnimation(fig, update, interval=50, blit=True) # blit=True is important for performance
        plt.tight_layout() # Adjust layout
        plt.show()
        
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    generator = EmotionFractalGenerator()
    generator.create_emotion_based_fractal()

if __name__ == "__main__":
    main() 