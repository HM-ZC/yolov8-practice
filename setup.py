from setuptools import setup
import os
from glob import glob

package_name = 'yolov8_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HM_ZC',
    maintainer_email='hmzc0327@gmail.com',
    description='YOLOv8 Detector for ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov8_node = yolov8_detector.yolov8_node:main',
        ],
    },
)
