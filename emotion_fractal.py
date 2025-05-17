import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.animation import FuncAnimation
import time
import math
import colorsys # Added for new colormap generation

# Valence-Arousal map for emotions (Valence: -1 to 1, Arousal: -1 to 1)
EMOTION_VA_MAP = {
    'happy':    (0.8, 0.6),
    'sad':      (-0.7, -0.4),
    'angry':    (-0.6, 0.7),
    'fear':     (-0.5, 0.8),
    'surprise': (0.4, 0.7),
    'disgust':  (-0.8, 0.3),
    'neutral':  (0.0, 0.0)
}

INITIAL_VIEW_RANGE = 2.5

# --- MODIFICATION: Focus C value range for more connected sets ---
# Updated C value range for fractal diversity
C_REAL_MIN = -1.2
C_REAL_MAX = 1.2
C_IMAG_MIN = -1.2
C_IMAG_MAX = 1.2

# New constants for guiding target C values towards more intricate regions
# Aiming for 'seahorse valley' like regions
TARGET_C_REAL_CENTER = -0.75 # Changed from -0.1
TARGET_C_REAL_SPAN = 0.5  # c_real will range from -0.75 +/- 0.25 => -1.0 to -0.5
TARGET_C_IMAG_CENTER = 0.1  # Changed from 0.65
TARGET_C_IMAG_SPAN = 0.5  # c_imag will range from 0.1 +/- 0.25 => -0.15 to 0.35

# --- MODIFICATION: Further reduce zoom center variation range ---
ZOOM_CENTER_X_MIN = -0.3
ZOOM_CENTER_X_MAX = 0.3
ZOOM_CENTER_Y_MIN = -0.3
ZOOM_CENTER_Y_MAX = 0.3

MIN_VIEW_WIDTH = 0.0001 # Maximum zoom limit

# --- VISUAL OPTIMIZATION: Easing function ---
def ease_in_out_quad(t):
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2

class EmotionFractalGenerator:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.last_emotion_detection_time = 0
        # Changed to random 2-5 second interval
        self.min_emotion_detection_interval = 2.0 
        self.max_emotion_detection_interval = 5.0
        # self.emotion_detection_interval = 10.0 # Removed fixed interval
        self.next_emotion_detection_time = 0.0

        self.detected_emotion_category = 'neutral'

        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.target_valence = 0.0
        self.target_arousal = 0.0
        self.previous_valence = 0.0
        self.previous_arousal = 0.0
        self.va_transition_progress = 1.0
        self.va_transition_duration = 4.5

        # Initialize C values to the new target center for intricate fractals
        initial_c = complex(TARGET_C_REAL_CENTER, TARGET_C_IMAG_CENTER) # Updated to new center
        self.current_julia_c = initial_c 
        self.target_julia_c = initial_c 
        self.previous_julia_c = initial_c
        self.c_transition_progress = 1.0
        self.c_transition_duration = 5.0

        # --- MODIFICATION: Parameters for Stuck State Detection & Nudge ---
        self.stuck_state_timer = 0.0
        self.stuck_detection_threshold = 0.002
        self.stuck_duration_trigger = 3.0    
        self.c_nudge_magnitude = 0.1        # MODIFIED: Increased from 0.05

        self.max_iter = 150
        self.current_colormap_object = self._generate_artistic_colormap(self.current_valence, self.current_arousal)

        # Disable continuous zoom by setting factor to 1.0
        self.zoom_factor_per_second = 1.0
        
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.target_zoom_center_x = 0.0
        self.target_zoom_center_y = 0.0
        self.previous_zoom_center_x = 0.0
        self.previous_zoom_center_y = 0.0
        self.zc_transition_progress = 1.0
        self.zc_transition_duration = 4.0

        self.c_complexity_random_factor = 0.08  # For adding randomness to target C values
        self.max_iter_complexity_random_span = 40 # For adding randomness to max_iter (+/- half of this, e.g., +/- 20)

        self.view_xmin = self.zoom_center_x - INITIAL_VIEW_RANGE / 2
        self.view_xmax = self.zoom_center_x + INITIAL_VIEW_RANGE / 2
        self.view_ymin = self.zoom_center_y - INITIAL_VIEW_RANGE / 2
        self.view_ymax = self.zoom_center_y + INITIAL_VIEW_RANGE / 2
        self.last_frame_time = time.time()
        
        self._update_fractal_parameters_from_va()
        self.current_fractal_data = self.julia_smooth_color(
            c=self.current_julia_c, max_iter=self.max_iter,
            xmin=self.view_xmin, xmax=self.view_xmax,
            ymin=self.view_ymin, ymax=self.view_ymax
        )

        # For emotion response optimization
        self.emotion_change_threshold = 0.15
        self.last_emotion_vector = (0.0, 0.0) # Initial V,A

        self.rotation_angle = 0.0 # Initialize rotation angle

    def _get_va_from_discrete_emotion(self, emotion_category):
        return EMOTION_VA_MAP.get(emotion_category, EMOTION_VA_MAP['neutral'])

    def _map_value(self, value, in_min, in_max, out_min, out_max):
        if (in_max - in_min) == 0: return out_min
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _map_arousal_to_c_real(self, arousal):
        return TARGET_C_REAL_CENTER + self._map_value(arousal, -1.0, 1.0, -TARGET_C_REAL_SPAN / 2, TARGET_C_REAL_SPAN / 2)

    def _map_valence_to_c_imag(self, valence):
        return TARGET_C_IMAG_CENTER + self._map_value(valence, -1.0, 1.0, -TARGET_C_IMAG_SPAN / 2, TARGET_C_IMAG_SPAN / 2)

    def _map_arousal_to_zoom_center_x(self, arousal):
        return self._map_value(arousal, -1.0, 1.0, ZOOM_CENTER_X_MIN, ZOOM_CENTER_X_MAX)

    def _map_valence_to_zoom_center_y(self, valence):
        return self._map_value(valence, -1.0, 1.0, ZOOM_CENTER_Y_MIN, ZOOM_CENTER_Y_MAX)
        
    def _get_fractal_max_iter(self, arousal):
        normalized_arousal = (arousal + 1) / 2 # 0 to 1
        min_i = 100  # Increased base iterations
        max_i_range = 250 # Increased max additional iterations
        
        base_iter = int(min_i + normalized_arousal * max_i_range)
        
        # Add random offset for complexity variation
        # self.max_iter_complexity_random_span (e.g., 40) will result in an offset like +/- 20
        random_iter_offset = int((np.random.rand() - 0.5) * self.max_iter_complexity_random_span)
        
        calculated_iter = base_iter + random_iter_offset
        return max(50, calculated_iter) # Ensure max_iter doesn't go too low (e.g., min 50 iterations)

    def _generate_artistic_colormap(self, valence, arousal=0.0):
        """Generates a colormap with dynamic hue based on valence and arousal."""
        # Using HSL color space for more natural gradients
        base_hue_angle = (valence + 1) / 2 * 360  # Hue mapped to 0-360 degrees
        
        # MODIFIED: Increased random hue offset range from +/- 5/360 to +/- 15 degrees.
        # The original random offset was already divided by 360. Now applying a degree-based offset.
        base_hue_angle += (np.random.rand() * 30 - 15) # Random offset between -15 and +15 degrees
        
        time_hue_angle = (time.time() * 10) % 360
        base_hue_angle += time_hue_angle

        base_hue_angle = base_hue_angle % 360 # Ensure final angle is within 0-360 before normalization
        
        # Define color for the inside of the set (dark, related to hue, low saturation)
        inside_color_h = base_hue_angle / 360.0 # Normalized hue for HSL conversion
        inside_color_l = 0.4  # Very dark lightness
        inside_color_s = 0.7   # Low saturation for a muted color
        inside_rgb = colorsys.hls_to_rgb(inside_color_h, inside_color_l, inside_color_s)

        colors = [inside_rgb] # Start with the calculated dark (but not black) color for inside the set
        
        num_gradient_colors = 5 # Number of colors in the gradient part

        # Parameters for the main gradient colors
        gradient_base_saturation = 0.7 + np.clip(arousal, -1, 1) * 0.3
        gradient_base_lightness = 0.3 + np.clip(valence, -1, 1) * 0.2

        for i in range(num_gradient_colors):
            # Use the final (potentially randomized and time-shifted) base_hue_angle for the start of the gradient hues
            angle = (base_hue_angle + i * 15) % 360  # Each step increases hue by 15 degrees
            h = angle / 360.0 # Normalized hue for HSL conversion
            
            # Adjust lightness for gradient effect, ensuring it starts above the inside_color_l
            l = (gradient_base_lightness * (0.7 + (i / (num_gradient_colors - 1)) * 0.6)) if num_gradient_colors > 1 else gradient_base_lightness
            l = np.clip(l, 0.15, 0.9) # Ensure lightness is in a good range (0.15 is brighter than inside_color_l=0.05)
            
            # Adjust saturation for gradient effect
            s = np.clip(gradient_base_saturation * (0.8 + (i / (num_gradient_colors - 1)) * 0.2), 0.3, 1.0) if num_gradient_colors > 1 else np.clip(gradient_base_saturation, 0.3, 1.0)

            rgb = colorsys.hls_to_rgb(h, l, s)
            colors.append(rgb)
        
        cmap_name = f"dynamic_hue_v{valence:.2f}_a{arousal:.2f}"
        if len(colors) > 1: # This should always be true as colors starts with inside_rgb
            # Nodes: first color (inside_rgb) at 0.0, rest of gradient from 0.1 to 1.0
            nodes = [0.0] + list(np.linspace(0.1, 1.0, len(colors)-1))
            
            min_len = min(len(nodes), len(colors))
            nodes = nodes[:min_len]
            actual_colors_for_lscm = colors[:min_len]

            if not actual_colors_for_lscm: return plt.cm.magma # Fallback
            
            color_def_list = []
            for i_node in range(len(nodes)):
                color_def_list.append( (nodes[i_node], actual_colors_for_lscm[i_node]) )
            
            try:
                return LinearSegmentedColormap.from_list(cmap_name, color_def_list, N=256)
            except Exception as e:
                print(f"Error generating colormap: {e}")
                return plt.cm.magma # Fallback
        else:
            # This case should ideally not be reached if colors always has at least inside_rgb
            return plt.cm.magma # Fallback

    def _update_fractal_parameters_from_va(self):
        """
        Calculates/updates fractal parameters based on current V/A.
        target_julia_c is NO LONGER set here. It's set once when emotion changes.
        max_iter and colormap ARE updated here based on current V/A.
        """
        self.max_iter = self._get_fractal_max_iter(self.current_arousal)
        self.current_colormap_object = self._generate_artistic_colormap(self.current_valence, self.current_arousal)
        self.rotation_angle = np.radians(self.current_arousal * 45) # Use arousal to control rotation

    def julia_smooth_color(self, c, max_iter, xmin, xmax, ymin, ymax): # Removed escape_radius default
        # Dynamic escape radius based on arousal
        dynamic_escape_radius = 2.0 + (self.current_arousal * 0.5) # Access self.current_arousal
        escape_radius_sq = dynamic_escape_radius**2
        log_escape_radius = math.log(dynamic_escape_radius) # Use dynamic_escape_radius

        x_pixels = np.linspace(xmin, xmax, self.width)
        y_pixels = np.linspace(ymin, ymax, self.height)
        X, Y = np.meshgrid(x_pixels, y_pixels)
        Z = X + 1j * Y

        # Apply complex rotation
        Z_rotated = Z * np.exp(1j * self.rotation_angle)
        Z = Z_rotated
        
        fractal_image = np.zeros(Z.shape, dtype=float)
        iterations = np.zeros(Z.shape, dtype=int)
        # Z_at_escape = np.copy(Z) # This might not be needed with active_pixels logic

        # Dynamic iteration termination
        active_pixels = np.ones_like(Z, dtype=bool)
        Z_at_escape_for_smoothing = np.zeros_like(Z, dtype=complex) # For storing Z at escape for smoothing

        for i in range(max_iter):
            if not np.any(active_pixels): # Optimized break condition
                break
            
            Z_active = Z[active_pixels]
            Z_active_new = Z_active**2 + c
            Z[active_pixels] = Z_active_new
            
            iterations[active_pixels] += 1
            
            escaped_now = np.abs(Z_active_new) >= dynamic_escape_radius # Check against dynamic_escape_radius
            
            # Store Z value at escape for smoothing only for those that escaped in this iteration
            Z_at_escape_for_smoothing[active_pixels][escaped_now] = Z_active_new[escaped_now]

            # Update active_pixels: remove those that escaped in this iteration
            active_pixels[active_pixels] = ~escaped_now


        # Corrected mask for all escaped pixels over the iterations
        escaped_mask = iterations < max_iter # Pixels that escaped before max_iter
        # Ensure we only try to smooth pixels that actually escaped
        # and have valid Z_at_escape_for_smoothing values (which are non-zero complex numbers)
        valid_for_smoothing_mask = escaped_mask & (np.abs(Z_at_escape_for_smoothing) > 1e-9)


        if np.any(valid_for_smoothing_mask):
            abs_Z_at_escape = np.abs(Z_at_escape_for_smoothing[valid_for_smoothing_mask])
            # Clamp to avoid log(0) or log(negative) if any Z_at_escape is too small, though abs() handles negative.
            # Minimum value for log argument should be > 0. Using a small epsilon if abs_Z_at_escape can be exactly 0 or 1.
            # np.log(log(abs_Z)) requires abs_Z > 1 for log(abs_Z) > 0.
            # And abs_Z must be > 0 for the outer log.
            abs_Z_at_escape = np.maximum(abs_Z_at_escape, 1.0 + 1e-9) # ensure log(abs_Z_at_escape) > 0 for log(log(...))
            
            # The smoothing formula: iter + 1 - log(log|Z_n|)/log(2)
            # Ensure that log(|Z_n|) itself is > 0, hence |Z_n| > 1.
            # If |Z_n| is very close to escape_radius, log(log|Z_n|) can be tricky.
            # A common approach is to use log_escape_radius directly.
            # smooth_val = iterations + 1 - log(log|Z_n|/log(2)) / log_escape_radius_base_2 OR
            # smooth_val = iterations + 1 - log(log|Z_n|)/log(2)
            # The formula used previously was: iterations[escaped_mask] + 1.0 - np.log(np.log(abs_Z_at_escape)) / math.log(2.0)
            # This requires abs_Z_at_escape > 1 for inner log, and then inner log > 0 for outer log.
            
            # Let's try a slightly different but common smoothing:
            # nu = log( log( |Z| ) / log(escape_radius) ) / log(2)
            # value = iterations - nu
            # This keeps values near iteration counts.
            # Here, log_escape_radius = math.log(dynamic_escape_radius)
            
            # For safety, ensure arguments to log are positive.
            log_abs_Z = np.log(abs_Z_at_escape) # abs_Z_at_escape is already >= 1.0 + 1e-9
            smooth_factor = np.log(log_abs_Z / log_escape_radius) / math.log(2.0) # Make sure log_escape_radius is not 0
            
            fractal_image[valid_for_smoothing_mask] = iterations[valid_for_smoothing_mask] - smooth_factor
        
        # Pixels that did not escape are "inside" the set.
        inside_set_mask = (iterations == max_iter)
        fractal_image[inside_set_mask] = 0.0 # Typically, inside is mapped to 0 or a specific value

        # Normalize the smoothed values for coloring (only for escaped points)
        # This normalization should occur on the part of fractal_image that corresponds to escaped_mask.
        if np.any(escaped_mask): # Use original escaped_mask for normalization range
            # Consider only values from 'valid_for_smoothing_mask' for min/max calculation
            # or more generally, all 'escaped_mask' points that got a value.
            # The values in 'fractal_image[escaped_mask]' can be iteration counts or smoothed iteration counts.
            
            # Let's refine normalization:
            # We want to normalize values that are not 0 (inside set)
            # and are on the 'escaped_mask'
            drawable_mask = escaped_mask & (fractal_image != 0.0) # Pixels that escaped and have some non-zero value
            
            if np.any(drawable_mask):
                min_val_escaped = np.min(fractal_image[drawable_mask])
                max_val_escaped = np.max(fractal_image[drawable_mask])
                if max_val_escaped > min_val_escaped:
                    fractal_image[drawable_mask] = (fractal_image[drawable_mask] - min_val_escaped) / (max_val_escaped - min_val_escaped)
                elif max_val_escaped == min_val_escaped and min_val_escaped != 0 : # all escaped to same value
                     fractal_image[drawable_mask] = 0.5 # map to a mid-range color
                # else: (if min_val_escaped is 0, or drawable_mask is empty, they remain 0 or unchanged)
            
            # Ensure points that only just escaped (low iteration count) but didn't get smoothed
            # (e.g. if valid_for_smoothing_mask was false for them) are still colored.
            # The previous normalization was simpler:
            # min_val_escaped = np.min(fractal_image[escaped_mask])
            # max_val_escaped = np.max(fractal_image[escaped_mask])
            # if max_val_escaped > min_val_escaped:
            #     fractal_image[escaped_mask] = (fractal_image[escaped_mask] - min_val_escaped) / (max_val_escaped - min_val_escaped)
            # else: fractal_image[escaped_mask] = 0.0 (if all same value)
            # This simpler one might be more robust if smoothing logic is tricky. Let's revert to that for now.

            min_val_escaped = np.min(fractal_image[escaped_mask])
            max_val_escaped = np.max(fractal_image[escaped_mask])
            if max_val_escaped > min_val_escaped:
                fractal_image[escaped_mask] = (fractal_image[escaped_mask] - min_val_escaped) / (max_val_escaped - min_val_escaped)
            else: # All escaped points got the same value or only one escaped point
                fractal_image[escaped_mask] = 0.5 # Map to a mid-range color if not distinguishable


        fractal_image[inside_set_mask] = 0.0 # Re-ensure inside is 0 after normalization
        fractal_image = np.clip(fractal_image, 0.0, 1.0)
        return fractal_image

    def create_emotion_based_fractal(self):
        plt.style.use('default') # Explicitly set a default light style
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1, 2]})
        ret, frame_cv = self.cap.read()
        cam_h, cam_w = (480, 640) if not ret else frame_cv.shape[:2]
        camera_plot = ax1.imshow(np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
        ax1.set_title('Camera Feed & Emotion Analysis')
        ax1.axis('off')
        
        self.fractal_plot = ax2.imshow(self.current_fractal_data, cmap=self.current_colormap_object, vmin=0, vmax=1)
        ax2.set_title(f"Fractal (V:{self.current_valence:.2f} A:{self.current_arousal:.2f} C:{self.current_julia_c.real:.3f}{self.current_julia_c.imag:+.3f}j Iter:{self.max_iter} ZC:({self.zoom_center_x:.1f},{self.zoom_center_y:.1f}))")
        ax2.axis('off')
        
        def update_animation(frame_count):
            animation_loop_time = time.time()
            delta_time = animation_loop_time - self.last_frame_time
            if delta_time == 0: delta_time = 1/60.0
            self.last_frame_time = animation_loop_time

            ret, frame_cv_raw = self.cap.read()
            if not ret: return [camera_plot, self.fractal_plot]
            
            frame_cv_display = cv2.flip(frame_cv_raw, 1)
            gray = cv2.cvtColor(frame_cv_display, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120,120))
            current_emotion_for_display = self.detected_emotion_category
            
            if len(faces) > 0 and animation_loop_time >= self.next_emotion_detection_time:
                try:
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    fx, fy, fw, fh = largest_face
                    face_roi = frame_cv_display[fy:fy+fh, fx:fx+fw]
                    if face_roi.size > 0:
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                        if result and isinstance(result, list) and len(result) > 0:
                             emotions_data = result[0]['emotion']
                             dominant_emotion = max(emotions_data, key=emotions_data.get)
                             
                             # Emotion response optimization
                             current_v_detected, current_a_detected = self._get_va_from_discrete_emotion(dominant_emotion)
                             delta_v = abs(current_v_detected - self.last_emotion_vector[0])
                             delta_a = abs(current_a_detected - self.last_emotion_vector[1])

                             # Only update targets if emotion category changes OR V/A change significantly
                             significant_va_change = (delta_v > self.emotion_change_threshold or delta_a > self.emotion_change_threshold)
                             
                             if self.detected_emotion_category != dominant_emotion or significant_va_change:
                                 if significant_va_change:
                                     print(f"Significant V/A change detected: V_delta={delta_v:.2f}, A_delta={delta_a:.2f}. Current emotion: {dominant_emotion}")
                                     # self.next_emotion_detection_time = animation_loop_time # Immediate re-check can be too fast, let normal cycle handle it or slightly reduce next interval
                                     # For now, let's rely on the standard emotion detection interval update below,
                                     # but ensure the new VA targets are set.
                                 
                                 self.detected_emotion_category = dominant_emotion # Update category
                                 current_emotion_for_display = dominant_emotion
                                 
                                 self.previous_valence = self.current_valence
                                 self.previous_arousal = self.current_arousal
                                 # Use the just detected V/A as the new target
                                 self.target_valence = current_v_detected
                                 self.target_arousal = current_a_detected
                                 self.va_transition_progress = 0.0
                                 
                                 # Add randomness to target C value for complexity
                                 random_c_offset_real = (np.random.rand() - 0.5) * self.c_complexity_random_factor
                                 random_c_offset_imag = (np.random.rand() - 0.5) * self.c_complexity_random_factor

                                 # Base target C from V/A
                                 base_target_c_real = self._map_arousal_to_c_real(self.target_arousal)
                                 base_target_c_imag = self._map_valence_to_c_imag(self.target_valence)
                                 
                                 # Apply random offset
                                 final_target_c_real = base_target_c_real + random_c_offset_real
                                 final_target_c_imag = base_target_c_imag + random_c_offset_imag
                                 
                                 # Ensure the randomized C value stays within the global min/max bounds
                                 final_target_c_real = np.clip(final_target_c_real, C_REAL_MIN, C_REAL_MAX)
                                 final_target_c_imag = np.clip(final_target_c_imag, C_IMAG_MIN, C_IMAG_MAX)

                                 self.target_julia_c = complex(final_target_c_real, final_target_c_imag)
                                 
                                 self.target_zoom_center_x = self._map_arousal_to_zoom_center_x(self.target_arousal)
                                 self.target_zoom_center_y = self._map_valence_to_zoom_center_y(self.target_valence)
                                 
                                 self.last_emotion_vector = (current_v_detected, current_a_detected) # Update last emotion vector

                except Exception as e: print(f"Emotion detection error: {e}")
                finally:
                    # Update next_emotion_detection_time to random 2-5s interval
                    self.next_emotion_detection_time = animation_loop_time + np.random.uniform(self.min_emotion_detection_interval, self.max_emotion_detection_interval)
            
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                cv2.rectangle(frame_cv_display, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                cv2.putText(frame_cv_display, f"Emotion: {current_emotion_for_display.capitalize()}", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            va_text_display = f"V: {self.current_valence:.2f}, A: {self.current_arousal:.2f}"
            cv2.putText(frame_cv_display, va_text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            camera_plot.set_array(cv2.cvtColor(frame_cv_display, cv2.COLOR_BGR2RGB))

            if self.va_transition_progress < 1.0:
                self.va_transition_progress += delta_time / self.va_transition_duration
                self.va_transition_progress = min(self.va_transition_progress, 1.0)
                eased_progress_va = ease_in_out_quad(self.va_transition_progress)
                self.current_valence = self.previous_valence + (self.target_valence - self.previous_valence) * eased_progress_va
                self.current_arousal = self.previous_arousal + (self.target_arousal - self.previous_arousal) * eased_progress_va
            else:
                self.current_valence = self.target_valence
                self.current_arousal = self.target_arousal

            self._update_fractal_parameters_from_va()

            is_c_target_different = abs(self.current_julia_c.real - self.target_julia_c.real) > 1e-5 or \
                                    abs(self.current_julia_c.imag - self.target_julia_c.imag) > 1e-5
            if is_c_target_different and self.c_transition_progress >= 1.0:
                self.previous_julia_c = self.current_julia_c
                self.c_transition_progress = 0.0
            if self.c_transition_progress < 1.0:
                self.c_transition_progress += delta_time / self.c_transition_duration
                self.c_transition_progress = min(self.c_transition_progress, 1.0)
                eased_progress_c = ease_in_out_quad(self.c_transition_progress)
                real_part = self.previous_julia_c.real + (self.target_julia_c.real - self.previous_julia_c.real) * eased_progress_c
                imag_part = self.previous_julia_c.imag + (self.target_julia_c.imag - self.previous_julia_c.imag) * eased_progress_c
                self.current_julia_c = complex(real_part, imag_part)
            elif not is_c_target_different : 
                 self.current_julia_c = self.target_julia_c # Snap if already at target

            is_zc_target_different = abs(self.zoom_center_x - self.target_zoom_center_x) > 1e-5 or \
                                     abs(self.zoom_center_y - self.target_zoom_center_y) > 1e-5

            if is_zc_target_different and self.zc_transition_progress >= 1.0: # New target and previous transition done
                self.previous_zoom_center_x = self.zoom_center_x
                self.previous_zoom_center_y = self.zoom_center_y
                self.zc_transition_progress = 0.0 # Start new transition
            
            if self.zc_transition_progress < 1.0:
                self.zc_transition_progress += delta_time / self.zc_transition_duration
                self.zc_transition_progress = min(self.zc_transition_progress, 1.0)
                eased_progress_zc = ease_in_out_quad(self.zc_transition_progress)
                self.zoom_center_x = self.previous_zoom_center_x + (self.target_zoom_center_x - self.previous_zoom_center_x) * eased_progress_zc
                self.zoom_center_y = self.previous_zoom_center_y + (self.target_zoom_center_y - self.previous_zoom_center_y) * eased_progress_zc
            elif not is_zc_target_different:
                self.zoom_center_x = self.target_zoom_center_x # Snap if already at target
                self.zoom_center_y = self.target_zoom_center_y

            zoom_multiplier = self.zoom_factor_per_second ** delta_time
            current_width_view = self.view_xmax - self.view_xmin
            current_height_view = self.view_ymax - self.view_ymin

            new_width = current_width_view * zoom_multiplier
            new_height = current_height_view * zoom_multiplier

            if new_width < MIN_VIEW_WIDTH:
                new_width = MIN_VIEW_WIDTH
            if new_height < MIN_VIEW_WIDTH:
                new_height = MIN_VIEW_WIDTH

            self.view_xmin = self.zoom_center_x - new_width / 2
            self.view_xmax = self.zoom_center_x + new_width / 2
            self.view_ymin = self.zoom_center_y - new_height / 2
            self.view_ymax = self.zoom_center_y + new_height / 2
            
            self.current_fractal_data = self.julia_smooth_color(
                c=self.current_julia_c, max_iter=self.max_iter,
                xmin=self.view_xmin, xmax=self.view_xmax,
                ymin=self.view_ymin, ymax=self.view_ymax
            )
            self.fractal_plot.set_array(self.current_fractal_data)
            self.fractal_plot.set_cmap(self.current_colormap_object)
            self.fractal_plot.set_clim(vmin=0, vmax=1) 
            
            # --- MODIFICATION: Stuck State Detection & Nudge --- (after fractal_data is calculated)
            if self.current_fractal_data is not None and self.current_fractal_data.size > 0:
                current_variance = np.var(self.current_fractal_data)
                if current_variance < self.stuck_detection_threshold: # Use updated threshold
                    self.stuck_state_timer += delta_time
                else:
                    self.stuck_state_timer = 0.0

                if self.stuck_state_timer > self.stuck_duration_trigger: # Use updated trigger
                    print(f"Stuck state detected (variance: {current_variance:.4f})! Multi-dimensional perturbation.")
                    
                    perturbation_type = np.random.choice(['c', 'zoom', 'both'])
                    print(f"Perturbation type: {perturbation_type}")

                    if perturbation_type in ['c', 'both']:
                        if np.random.rand() < 0.5:
                            print("Nudging C value towards target dendritic region center.")
                            rand_offset_real = np.random.normal(0, TARGET_C_REAL_SPAN / 4) # Smaller random offset around target center
                            rand_offset_imag = np.random.normal(0, TARGET_C_IMAG_SPAN / 4)
                            new_c_real = TARGET_C_REAL_CENTER + rand_offset_real
                            new_c_imag = TARGET_C_IMAG_CENTER + rand_offset_imag
                        else:
                            print("Nudging C value based on zoom center with smaller randomness.")
                            new_c_real = self.zoom_center_x + np.random.normal(0, 0.1) # Reduced random scale
                            new_c_imag = self.zoom_center_y + np.random.normal(0, 0.1) # Reduced random scale
                        
                        self.target_julia_c = complex(
                            np.clip(new_c_real, C_REAL_MIN, C_REAL_MAX), # Still clip within overall bounds
                            np.clip(new_c_imag, C_IMAG_MIN, C_IMAG_MAX)
                        )
                        self.c_transition_progress = 0.0 # Start C transition

                    if perturbation_type in ['zoom', 'both']:
                        # Generate spiral zoom path or random new center
                        print("Nudging zoom center and temporarily adjusting zoom speed.")
                        theta = np.random.uniform(0, 2 * np.pi)
                        # Keep zoom center target within a reasonable range, e.g., [-0.5, 0.5]
                        self.target_zoom_center_x = np.clip(0.5 * np.cos(theta), ZOOM_CENTER_X_MIN, ZOOM_CENTER_X_MAX)
                        self.target_zoom_center_y = np.clip(0.5 * np.sin(theta), ZOOM_CENTER_Y_MIN, ZOOM_CENTER_Y_MAX)
                        
                        # Make previous zoom center the current one to avoid jump before transition
                        self.previous_zoom_center_x = self.zoom_center_x
                        self.previous_zoom_center_y = self.zoom_center_y
                        self.zc_transition_progress = 0.0 # Start zoom center transition
                        
                        # Temporarily adjust zoom factor per second if needed, or rely on existing zoom.
                        # The prompt suggests self.zoom_factor_per_second = 0.97
                        # This should be a temporary effect, so perhaps store original and restore.
                        # For simplicity now, let's just set it.
                        # self.zoom_factor_per_second = 0.97 # This might be too fast or disruptive.
                        # Consider a smaller, temporary adjustment if any, or rely on c change.

                    # Reset stuck timer and ensure transitions start
                    self.stuck_state_timer = 0.0
                    # c_transition_progress and zc_transition_progress are set above if type matches.
                    
                    # Also, force V/A to re-evaluate based on current emotion, maybe it got stuck on neutral too long
                    # This could help if the system is visually stuck AND emotionally neutral.
                    # current_v_detected, current_a_detected = self._get_va_from_discrete_emotion(self.detected_emotion_category)
                    # self.target_valence = current_v_detected
                    # self.target_arousal = current_a_detected
                    # self.va_transition_progress = 0.0 # Force V/A transition
                    print(f"Nudged. New target C: {self.target_julia_c}, new target ZC: ({self.target_zoom_center_x:.2f}, {self.target_zoom_center_y:.2f})")
            else:
                self.stuck_state_timer = 0.0

            ax2.set_title(f"Fractal V:{self.current_valence:.2f} A:{self.current_arousal:.2f} C:{self.current_julia_c.real:.3f}{self.current_julia_c.imag:+.3f}j Iter:{self.max_iter} ZC:({self.zoom_center_x:.2f},{self.zoom_center_y:.2f}) Zoom:{1/new_width:.2f}x Angle:{np.degrees(self.rotation_angle):.1f}")
            
            return [camera_plot, self.fractal_plot]
        
        anim = FuncAnimation(fig, update_animation, interval=40, blit=True) 
        plt.tight_layout(pad=2.0)
        plt.show()
        
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        generator = EmotionFractalGenerator(width=512, height=512) 
        generator.create_emotion_based_fractal()
    except Exception as e:
        print(f"An error occurred: {e}")
        if 'generator' in locals() and hasattr(generator, 'cap') and hasattr(generator.cap, 'isOpened') and generator.cap.isOpened():
            generator.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()