# Autonomous Delivery Robot System

An autonomous delivery robot system featuring custom-built CNNs for real-time visual perception and an interactive web interface for mission planning and control. This project was developed for the AI module at Abdelmalek Essaadi University.

---

##  Key Features

- **Real-Time Visual Perception:** Two custom CNN models built from scratch in TensorFlow/Keras for:
  - **Traffic Light Recognition:** Classifies red, yellow, and green lights.
  - **Traffic Sign Recognition:** Identifies 43 unique traffic signs (GTSRB dataset).
- **Interactive Control Dashboard:** A full-stack web interface for:
  - **Route Planning:** Interactive start/end point selection on a map.
  - **Route Optimization:** Real-time shortest path calculation using the Geoapify API.
  - **Robot Monitoring:** Live status, position, and ETA tracking.
- **End-to-End System Design:** A complete "Perceive -> Decide -> Act" architecture designed for autonomous navigation in an urban environment.

---

## Tech Stack

#### 🤖 AI & Computer Vision
- Python
- TensorFlow & Keras
- OpenCV
- Scikit-learn
- NumPy

#### 🌐 Web Interface & Services
- HTML5 & CSS3
- JavaScript (ES6+)
- Leaflet.js (Interactive Maps)
- Geoapify API (Routing, Geocoding)



## Usage

- To train the models, run the Jupyter notebooks or Python scripts inside the `traffic_light_model/` and `traffic_sign_model/` directories.
- To use the control panel, open the `index.html` file and interact with the map and controls.

