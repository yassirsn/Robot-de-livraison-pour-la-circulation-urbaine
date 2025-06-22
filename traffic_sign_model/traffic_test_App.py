import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import PIL.Image
import PIL.ImageTk
import numpy as np
import threading
import time

class TrafficSignTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detection Tester")
        self.root.geometry("1000x700")
        
        # Variables
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.model = None  # Your AI model will go here
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Traffic Sign Detection Tester", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image upload section
        ttk.Label(control_frame, text="Image Testing:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.upload_btn = ttk.Button(control_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Camera section
        ttk.Label(control_frame, text="Camera Testing:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(20, 5))
        
        self.camera_btn = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Model info section
        ttk.Label(control_frame, text="Model Info:", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        self.model_status = ttk.Label(control_frame, text="Model: Not loaded", foreground="red")
        self.model_status.grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.load_model_btn = ttk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_model_btn.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Results section
        ttk.Label(control_frame, text="Detection Results:", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        
        self.results_text = tk.Text(control_frame, height=8, width=30, wrap=tk.WORD)
        self.results_text.grid(row=8, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=8, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure control frame grid
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(8, weight=1)
        
        # Display panel
        display_frame = ttk.LabelFrame(main_frame, text="Display", padding="10")
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.image_label = ttk.Label(display_frame, text="No image loaded", anchor="center")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure display frame grid
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
    def load_model(self):
        """Load your trained traffic sign detection model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras model files", "*.h5"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                self.log_result("Loading model...")
                
                # Import required libraries
                import tensorflow as tf
                from tensorflow.keras.models import load_model
                
                # Load the trained model
                self.model = load_model(model_path)
                
                # Load class names (you'll need to provide the path to your labels.csv or class names)
                self.load_class_names()
                
                # Set model parameters (adjust these to match your training settings)
                self.IMG_HEIGHT = 32  # Adjust to your model's input size
                self.IMG_WIDTH = 32   # Adjust to your model's input size
                self.CHANNELS = 1     # 1 for grayscale, 3 for RGB
                
                self.model_status.config(text="Model: Loaded ✓", foreground="green")
                self.log_result(f"Model loaded successfully from: {model_path}")
                self.log_result(f"Model input shape: {self.model.input_shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.log_result(f"Error loading model: {str(e)}")
    
    def load_class_names(self):
        """Load class names for traffic signs"""
        # Option 1: Load from CSV file (like your labels.csv)
        csv_path = filedialog.askopenfilename(
            title="Select Labels CSV File (Optional)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if csv_path:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                # Assuming your CSV has columns like 'ClassId' and 'Name'
                self.class_names = df.sort_values('ClassId')['Name'].tolist()
                self.log_result(f"Loaded {len(self.class_names)} class names from CSV")
            except Exception as e:
                self.log_result(f"Error loading CSV: {str(e)}")
                self.use_default_class_names()
        else:
            self.use_default_class_names()
    
    def use_default_class_names(self):
        """Use default German traffic sign class names"""
        self.class_names = [
            'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
            'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
            'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
            'No passing', 'No passing for vehicles over 3.5 metric tons',
            'Right-of-way at the next intersection', 'Priority road', 'Yield',
            'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
            'No entry', 'General caution', 'Dangerous curve to the left',
            'Dangerous curve to the right', 'Double curve', 'Bumpy road',
            'Slippery road', 'Road narrows on the right', 'Road work',
            'Traffic signals', 'Pedestrians', 'Children crossing',
            'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
            'End of all speed and passing limits', 'Turn right ahead',
            'Turn left ahead', 'Ahead only', 'Go straight or right',
            'Go straight or left', 'Keep right', 'Keep left',
            'Roundabout mandatory', 'End of no passing',
            'End of no passing by vehicles over 3.5 metric tons'
        ]
        self.log_result(f"Using default class names ({len(self.class_names)} classes)")
    
    def preprocess_single_image(self, img_array):
        """Preprocess image for model prediction (matches your training preprocessing)"""
        target_height = getattr(self, 'IMG_HEIGHT', 32)
        target_width = getattr(self, 'IMG_WIDTH', 32)
        channels = getattr(self, 'CHANNELS', 1)
        
        # Resize image
        img_resized = cv2.resize(img_array, (target_width, target_height))
        
        # Convert to grayscale if model expects grayscale
        if channels == 1:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            img_equalized = cv2.equalizeHist(img_gray)
            img_normalized = img_equalized / 255.0
            # Reshape for model input: (1, height, width, 1)
            img_reshaped = img_normalized.reshape(1, target_height, target_width, 1)
        else:
            # For RGB models
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb / 255.0
            # Reshape for model input: (1, height, width, 3)
            img_reshaped = img_normalized.reshape(1, target_height, target_width, 3)
        
        return img_reshaped
    
    def get_class_name(self, class_id_val):
        """Get class name from class ID"""
        if hasattr(self, 'class_names') and self.class_names and 0 <= class_id_val < len(self.class_names):
            return self.class_names[class_id_val]
        else:
            return f"Class ID: {class_id_val}"
        
    def upload_image(self):
        """Handle image upload and processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load and display image
                image = cv2.imread(file_path)
                if image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                self.current_frame = image.copy()
                self.display_image(image)
                self.log_result(f"Image loaded: {file_path}")
                
                # Process with AI model
                self.process_image(image)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")
    
    def toggle_camera(self):
        """Start or stop camera feed"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_running = True
            self.camera_btn.config(text="Stop Camera")
            self.log_result("Camera started")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.camera_btn.config(text="Start Camera")
        self.log_result("Camera stopped")
    
    def camera_loop(self):
        """Main camera loop"""
        while self.camera_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_image(frame)
                    
                    # Process with AI model every few frames to avoid lag
                    if hasattr(self, 'frame_count'):
                        self.frame_count += 1
                    else:
                        self.frame_count = 0
                    
                    if self.frame_count % 10 == 0:  # Process every 10th frame
                        self.process_image(frame)
                
            time.sleep(0.03)  # ~30 FPS
    
    def process_image(self, image):
        """Process image with your AI model"""
        if self.model is None:
            self.log_result("Please load a model first")
            return
        
        try:
            # TODO: Replace this with your actual model inference code
            # Example:
            # predictions = self.model.predict(preprocess_image(image))
            # results = postprocess_predictions(predictions)
            
            # Placeholder detection (replace with your actual code)
            detections = self.dummy_detection(image)
            
            # Draw detections on image
            annotated_image = self.draw_detections(image, detections)
            self.display_image(annotated_image)
            
            # Log results
            if detections:
                for detection in detections:
                    self.log_result(f"Detected: {detection['class']} ({detection['confidence']:.2f})")
            else:
                self.log_result("No traffic signs detected")
                
        except Exception as e:
            self.log_result(f"Error in detection: {str(e)}")
    
    def dummy_detection(self, image):
        """Real detection function using your trained model"""
        if self.model is None:
            return []
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_single_image(image)
            
            # Make prediction
            predictions_probs = self.model.predict(processed_img, verbose=0)
            predicted_class_id = np.argmax(predictions_probs, axis=1)[0]
            confidence = np.max(predictions_probs)
            
            # Get class name
            predicted_class_name = self.get_class_name(predicted_class_id)
            
            # Only return detection if confidence is above threshold
            confidence_threshold = 0.5  # Adjust as needed
            if confidence > confidence_threshold:
                # For traffic sign detection, we'll create a centered bounding box
                # since your model does classification, not object detection
                h, w = image.shape[:2]
                box_size = min(h, w) // 2
                x = (w - box_size) // 2
                y = (h - box_size) // 2
                
                detection = {
                    'class': predicted_class_name,
                    'confidence': float(confidence),
                    'bbox': [x, y, box_size, box_size],
                    'class_id': int(predicted_class_id)
                }
                
                return [detection]
            else:
                return []
                
        except Exception as e:
            self.log_result(f"Error in model prediction: {str(e)}")
            return []
    
    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        annotated = image.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Resize image to fit display
        display_size = (640, 480)
        h, w = image.shape[:2]
        
        # Calculate scaling to maintain aspect ratio
        scale = min(display_size[0]/w, display_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = PIL.Image.fromarray(rgb_image)
        photo = PIL.ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep a reference
    
    def log_result(self, message):
        """Add message to results log"""
        timestamp = time.strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    try:
        root = tk.Tk()
        app = TrafficSignTester(root)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        print("Starting Traffic Sign Detection Tester...")
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()