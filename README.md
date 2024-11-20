# Sensor_Module

The Sensor Module is a perception component of the DART-LLM (Dependency-Aware Multi-Robot Task Decomposition and Execution using Large Language Models) system. It provides object detection and recognition capabilities using Vision-Language Models (VLMs) and selective search algorithms. The module processes aerial imagery and updates the object map database with detected items, their locations, and shapes.

## Features

- **VLM-Based Object Detection**
  - Zero-shot object detection using OpenAI's CLIP model
  - Selective Search algorithm for region proposals
  - Processes bird's-eye view images
  - Extracts object shapes (e.g., triangle, rectangle, circle) from image patches
  - Updates the object map database with detected objects

- **ROS2 Integration**
  - Subscribes to keyword topics to specify target objects
  - Publishes detection results and images to ROS2 topics
  - Separate node for updating the object database based on detection results

## Installation

1. **Clone the Repository into Your ROS2 Workspace:**
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/yourusername/Sensor_Module.git
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install torch torchvision
   pip install selectivesearch opencv-python
   pip install git+https://github.com/openai/CLIP.git
   pip install pillow
   pip install rclpy
   pip install cv_bridge
   pip install numpy
   pip install rclpy
   ```

3. **Build the Package:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select sensor_module
   ```

4. **Source the Workspace:**
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

## Usage

### Basic Operation

Launch the sensor module:
```bash
/bin/python3 /root/share/ros2_ws/src/Sensor_Module/sensor_vlm_zero_shot_detection.py
```
```bash
/bin/python3 /root/share/ros2_ws/src/Sensor_Module/sensor_object_database_updater.py
```

### Input/Output Format

**Zero-Shot Detection Node:**

- **Input:** Receives target keywords from the `keywords_topic` (e.g., `["excavator", "dump truck"]`).
- **Output:**
  - Detection results published to `/vlm_detection_results` as a JSON string.
  - Detection images published to `area_image_topic` as ROS `sensor_msgs/Image` messages.

**Detection Results Format:**
```json
[
    {
        "bounding_box": [x1, y1, x2, y2],
        "confidence": 0.85,
        "shape": "rectangle",
        "object_name": "dump truck"
    },
    {
        "bounding_box": [x1, y1, x2, y2],
        "confidence": 0.90,
        "shape": "circle",
        "object_name": "excavator"
    }
]
```

**Object Database Updater Node:**

- **Input:** Subscribes to `/vlm_detection_results` to receive detection data.
- **Output:** Updates the `object_database.json` file with object information.

**Object Database Entry Format:**
```json
{
    "object_name": "excavator",
    "position": {
        "x": 150.0,
        "y": 75.0
    },
    "shape": "rectangle"
}
```

### ROS2 Topics

- **Zero-Shot Detection Node**

  - **Subscribes to:**
    - `keywords_topic` (`std_msgs/msg/String`): Receives target object keywords.
    
  - **Publishes to:**
    - `/vlm_detection_results` (`std_msgs/msg/String`): Publishes detection results in JSON format.
    - `area_image_topic` (`sensor_msgs/msg/Image`): Publishes images with visualized detections.

- **Object Database Updater Node**

  - **Subscribes to:**
    - `/vlm_detection_results` (`std_msgs/msg/String`): Receives detection results to update the object database.

## Configuration

- **Image Path:**
  - The image used for detection is specified in the `sensor_vlm_zero_shot_detection.py` script. By default, it is set to:
    ```
    Top_view_data/Kyushu_University_Field/Kyushu_University_Field.jpg
    ```
    - Update the `self.image_path` variable in the script if you wish to use a different image.

- **Object Database Path:**
  - The `object_database.json` file is saved in:
    ```
    ../../src/breakdown_function_handler/object_database/object_database.json
    ```
    - Ensure that this path exists and the script has write permissions.

- **Detection Threshold:**
  - The confidence threshold for detections is set in the `detect_objects` method:
    ```python
    if max_score > 0.2:
        # Process detection
    ```
    - Adjust the threshold value as needed for your application.

## Dependencies

- **ROS2 Humble or Later**
- **Python 3.8+**
- **PyTorch**
- **OpenAI CLIP**
- **OpenCV**
- **Selective Search**
- **Pillow**
- **NumPy**
- **rclpy**
- **cv_bridge**

## Notes

- **Hardware Requirements:**
  - The CLIP model can utilize GPU acceleration if available. Ensure that your system has a compatible GPU and the appropriate drivers installed.

- **Object Detection Limitations:**
  - The zero-shot detection relies on the CLIP model's ability to recognize objects based on textual descriptions. Detection accuracy may vary depending on the object and the quality of the input image.

- **Shape Extraction:**
  - The module includes functionality to extract basic shapes (triangle, rectangle, circle) from detected object patches using contour analysis.