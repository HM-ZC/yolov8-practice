import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from ultralytics import YOLO

class YOLOv8SegmentationNode(Node):
    def __init__(self):
        super().__init__('yolov8_segmentation_node')
        
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            sys.exit(1)

        self.bridge = CvBridge()
        self.model = YOLO('/home/hmzc/ros2_ws/src/best.pt')

        self.K = np.array([[469.8769, 0, 334.8598], [0, 469.8360, 240.2752], [0, 0, 1]])
        self.D = np.array([-0.0555, 0.0907, 0, 0])

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.process_image(frame)

    def process_image(self, cv_image):

        h, w = cv_image.shape[:2]
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (w, h), None)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        undistorted_image = cv2.remap(cv_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        results = self.model(undistorted_image)
        if results is None:
            return


        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(undistorted_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f'{box.conf[0]:.2f}'
                    cv2.putText(undistorted_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if not hasattr(result, 'masks') or result.masks is None:
                continue
            for mask in result.masks.xy:
                mask_np = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_np, [np.array(mask).astype(np.int32)], 255)
                self.draw_segmentation_mask(undistorted_image, mask_np)
                self.apply_logo(undistorted_image, mask_np)


        cv2.imshow('Detection', undistorted_image)
        cv2.waitKey(1)

    def apply_logo(self, image, mask):
        logo = cv2.imread('/home/hmzc/ros2_ws/src/yolov8_detector/084525b05264ae90e7b94a556d0f10bc_720.jpg')
        if logo is None:
            return
        logo_height, logo_width = logo.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return

        contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4:
            return

        contour_center = np.mean(approx, axis=0)
        pts2 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
        pts2 = (pts2 - contour_center) * 0.5 + contour_center
        pts2 = np.float32(pts2)

        pts1 = np.float32([[0, 0], [logo_width, 0], [logo_width, logo_height], [0, logo_height]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_logo = cv2.warpPerspective(logo, M, (image.shape[1], image.shape[0]))

        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(pts2)], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(image, mask_inv)
        logo_fg = cv2.bitwise_and(transformed_logo, mask)
        combined = cv2.add(img_bg, logo_fg)

        np.copyto(image, combined)

    def draw_segmentation_mask(self, image, mask):
        color_mask = np.zeros_like(image)
        color_mask[mask == 255] = [0, 255, 0]

        image[:] = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    yolov8_segmentation_node = YOLOv8SegmentationNode()
    rclpy.spin(yolov8_segmentation_node)
    yolov8_segmentation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
