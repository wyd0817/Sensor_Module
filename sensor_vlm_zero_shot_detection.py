import clip
import numpy as np
import selectivesearch
import torch
from PIL import Image as PILImage
from PIL import ImageDraw
from collections import OrderedDict
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import json
import cv2

class ZeroShotDetectionNode(Node):
    def __init__(self):
        super().__init__('zero_shot_detection_node')
        # Set the path to the image
        self.task_pictures = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Top_view_data'))
        self.image_path = os.path.join(self.task_pictures, 'Kyushu_University_Field', 'Kyushu_University_Field.jpg')

        # Subscribe to the 'keywords_topic' to receive target keywords
        self.subscription = self.create_subscription(
            String,
            'keywords_topic',
            self.listener_callback,
            10)

        # Create publishers to publish detection results and images
        self.results_publisher = self.create_publisher(String, '/vlm_detection_results', 10)
        self.image_publisher = self.create_publisher(Image, 'area_image_topic', 10)
        self.br = CvBridge()

        # Generate bounding boxes using selective search
        self.bounding_boxes = self.generate_bounding_boxes(
            self.image_path,
            resize=None,
            topk=50,
            scale=200,
            sigma=0.8,
            min_size=50)

        # Load the CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.device = device

    def load_image(self, img_path, resize=None, return_pil=False):
        # Load and preprocess the image
        image = PILImage.open(img_path).convert("RGB")
        if resize is not None:
            image = image.resize((resize, resize))
        if return_pil:
            return image
        image = np.asarray(image).astype(np.float32) / 255.
        return image

    def generate_bounding_boxes(self, img_path, resize=None, topk=50, scale=200, sigma=0.8, min_size=50):
        # Generate candidate bounding boxes using selective search
        img = self.load_image(img_path, resize=resize, return_pil=True)
        img_search = self.load_image(img_path, resize=resize)
        _, regions = selectivesearch.selective_search(
            img_search, scale=scale, sigma=sigma, min_size=min_size)
        candidates = OrderedDict()
        for i, r in enumerate(regions):
            if r['rect'] in candidates or r['size'] < 1000:
                continue
            x, y, w, h = r['rect']
            if w / h > 1.5 or h / w > 1.5:
                continue
            candidates[i] = r['rect']
        print('Finish generating bounding boxes!')
        return list(candidates.values())[:topk]

    def detect_objects(self, img, targets):
        batch = []
        img_patches = []
        dets = []

        # Extract image patches based on bounding boxes
        for box in self.bounding_boxes:
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            patch = (img[y1:y2, x1:x2] * 255.).astype(np.uint8)
            img_patches.append(patch)

            patch_pil = PILImage.fromarray(patch).convert("RGB")
            patch_tensor = self.preprocess(patch_pil).unsqueeze(0)
            batch.append(patch_tensor)

        # Combine all image patches into a batch
        batch = torch.cat(batch, dim=0).to(self.device)

        # Convert target labels to tokenized text
        text_input = clip.tokenize(targets).to(self.device)

        # Calculate image and text embeddings
        with torch.no_grad():
            patch_embs = self.model.encode_image(batch).float()
            text_embs = self.model.encode_text(text_input).float()

            # Normalize embeddings
            patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # Calculate similarity scores
        scores = patch_embs @ text_embs.t()

        for target_idx, target in enumerate(targets):
            target_boxes = []
            target_scores = []
            target_dets = []

            # Process each bounding box
            for i, box in enumerate(self.bounding_boxes):
                x, y, w, h = box
                x1, y1, x2, y2 = x, y, x + w, y + h
                max_score = scores[i, target_idx].item()
                if max_score > 0.25:
                    img_patch = img_patches[i]
                    shape = self.extract_shape_from_patch(img_patch)

                    target_boxes.append([x1, y1, x2, y2])
                    target_scores.append(max_score)

                    target_dets.append({
                        "bounding_box": [x1, y1, x2, y2],
                        "confidence": max_score,
                        "shape": shape,
                        "object_name": target
                    })

            # Apply Non-Maximum Suppression (NMS)
            if target_boxes:
                keep_indices = self.nms(target_boxes, target_scores, threshold=0.5)
                for idx in keep_indices:
                    dets.append(target_dets[idx])

        return dets

    def extract_shape_from_patch(self, img_patch):
        # Convert image to grayscale
        gray = cv2.cvtColor(img_patch, cv2.COLOR_RGB2GRAY)

        # Ensure image is uint8 type
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply threshold
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect shape based on contours
        shape = self.detect_shape(contours)

        return shape

    def detect_shape(self, contours):
        shape = "unknown"
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
            elif len(approx) > 4:
                shape = "circle"
        return shape

    def vis_detections(self, img, detections):
        # Convert image to PIL Image if necessary
        if isinstance(img, np.ndarray):
            img = (img * 255).astype(np.uint8)
            img = PILImage.fromarray(img)

        # Ensure the image supports transparency by converting to 'RGBA' mode
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Identify the detection with the highest confidence score
        if detections:
            max_confidence = max(det['confidence'] for det in detections)
        else:
            max_confidence = None

        # Create a drawing context with 'RGBA' mode to support transparency
        draw = ImageDraw.Draw(img, 'RGBA')

        for det in detections:
            x1, y1, x2, y2 = det['bounding_box']
            confidence = det['confidence']
            object_name = det['object_name']
            shape = det.get('shape', 'unknown')

            if confidence == max_confidence:
                # For the highest confidence detection, use semi-transparent red fill
                fill_color = (255, 0, 0, 100)  # Semi-transparent red (alpha=100)
                outline_color = (255, 0, 0, 255)  # Solid red outline
            else:
                # Other detections remain unchanged
                fill_color = None  # No fill
                outline_color = (255, 0, 0, 255)  # Solid red outline

            # Draw the bounding box with the specified fill and outline colors
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=2)

            # Draw the label above the bounding box
            label = f"{object_name} ({shape}): {confidence:.2f}"
            draw.text((x1, y1 - 10), label, fill=(255, 0, 0, 255))

        return img

    def listener_callback(self, msg):
        self.get_logger().info(f'Received keywords: "{msg.data}"')
        keywords = msg.data.strip('[]').replace('"', '').split(', ')
        img = self.load_image(self.image_path)
        detections = self.detect_objects(img, keywords)

        if len(detections) > 0:
            # Visualize detection results
            detection_image = self.vis_detections(img, detections)

            # Save detection results image to file as PNG
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_image_path = os.path.join(current_dir, 'detection_results.png')
            detection_image.save(output_image_path)
            self.get_logger().info(f'Detection results image saved at {output_image_path}')

            # Save detection results to JSON file
            output_json_path = os.path.join(current_dir, 'detection_results.json')
            with open(output_json_path, 'w') as f:
                json.dump(detections, f, indent=4)
            self.get_logger().info(f'Detection results saved at {output_json_path}')

            # Publish detection results to '/vlm_detection_results' topic
            try:
                results_msg = String()
                results_msg.data = json.dumps(detections)
                self.results_publisher.publish(results_msg)
                self.get_logger().info('Published detection results to /vlm_detection_results')
            except Exception as e:
                self.get_logger().error(f"Failed to publish detection results: {e}")

            # Publish detection image to 'area_image_topic'
            try:
                # Convert detection_image to 'RGB' mode if it's in 'RGBA' mode
                if detection_image.mode == 'RGBA':
                    detection_image_rgb = detection_image.convert('RGB')
                else:
                    detection_image_rgb = detection_image

                # Convert the PIL Image to a NumPy array
                detection_image_np = np.array(detection_image_rgb)

                # Convert the NumPy array to a ROS image message
                ros_image = self.br.cv2_to_imgmsg(detection_image_np, "rgb8")
                self.image_publisher.publish(ros_image)
                self.get_logger().info('Published detection image to area_image_topic')
            except Exception as e:
                self.get_logger().error(f"Failed to publish detection image: {e}")

        else:
            self.get_logger().info('No detections match the given keywords.')

    def nms(self, boxes, scores, threshold):
        # Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
        boxes = np.array(boxes)
        scores = np.array(scores)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

def main(args=None):
    rclpy.init(args=args)
    zero_shot_detection_node = ZeroShotDetectionNode()
    rclpy.spin(zero_shot_detection_node)
    zero_shot_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
