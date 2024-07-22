from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[{
                'video_device': '/dev/video2',
                'framerate': 30.0,
                'pixel_format': 'yuyv',
                'image_width': 640,
                'image_height': 480,
            }]
        ),
        Node(
            package='yolov8_detector',
            executable='yolov8_node',
            name='yolov8_detector',
            output='screen'
        )
    ])
