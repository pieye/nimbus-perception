# Nimbus-Detection

## Installation
To use the nimbus-detection the following software must be installed on your Rasperry Pi:
  - ROS perception (tested with noetic)
  - OpenCV 4.2
  - tensorflow lite
All three must be build from source for ARM.

But we offer a Raspberry Pi image which already includes everything including the nimbus software.

## Usage
To start Nimbus-Detection use the provided launch file:
```
roslaunch nimbus_detection nimbus_detection.launch
```
This will also start the nimbus_3d_driver in order to capture the pointcloud.
