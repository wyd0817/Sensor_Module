import os
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ObjectDatabaseUpdater(Node):
    def __init__(self):
        super().__init__('object_database_updater')

        # Path to the JSON database
        self.json_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'src', 'breakdown_function_handler', 'object_database', 'object_database.json'
        ))

        # Subscribe to the '/vlm_detection_results' topic to receive detection data
        self.subscription = self.create_subscription(
            String,
            '/vlm_detection_results',
            self.detection_callback,
            10
        )
        self.get_logger().info("ObjectDatabaseUpdater initialized and listening to /vlm_detection_results.")

    def detection_callback(self, msg):
        """
        Process received detection data and update the JSON file.
        """
        self.get_logger().info('Received detection results message')
        try:
            detection_data = json.loads(msg.data)
            if not isinstance(detection_data, list):
                self.get_logger().error("Received data is not a list.")
                return

            for detection in detection_data:
                # Check if data is valid
                if "bounding_box" not in detection or "shape" not in detection or "object_name" not in detection:
                    self.get_logger().error(f"Invalid detection format: {detection}")
                    continue

                # Extract information
                bounding_box = detection["bounding_box"]
                shape = detection["shape"]
                object_name = detection["object_name"]

                # Calculate position
                position = {
                    "x": (bounding_box[0] + bounding_box[2]) / 2,
                    "y": (bounding_box[1] + bounding_box[3]) / 2
                }

                # Save to JSON file
                self.save_position_and_shape_to_json(object_name, position, shape)

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to decode JSON data: {e}")
        except Exception as e:
            self.get_logger().error(f"An error occurred: {e}")

    def save_position_and_shape_to_json(self, object_name, position, shape):
        """
        Save target information to the JSON file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

        if os.path.exists(self.json_path) and os.path.getsize(self.json_path) > 0:
            with open(self.json_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = []

        object_data = {
            "object_name": object_name,
            "position": position,
            "shape": shape
        }

        # Check if object with the same name exists; update instead of inserting duplicate
        updated = False
        for item in data:
            if item["object_name"] == object_name:
                item.update(object_data)
                updated = True
                break
        if not updated:
            data.append(object_data)

        with open(self.json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        self.get_logger().info(f"Saved target position and shape of '{object_name}' to {self.json_path}")

def main(args=None):
    rclpy.init(args=args)
    updater = ObjectDatabaseUpdater()
    rclpy.spin(updater)
    updater.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
