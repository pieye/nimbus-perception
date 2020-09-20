cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o3 -std=c++11 -pthread")

# set the project name
project(nimbus_segmentation)

find_package(OpenCV 4.2 EXACT REQUIRED)
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg cv_bridge image_transport pcl_conversions pcl_ros) 
find_package(Threads REQUIRED)

## Declare a catkin package
catkin_package(
   #INCLUDE_DIRS include
   #${PROJECT_NAME}_node
   CATKIN_DEPENDS geometry_msgs pcl_ros roscpp sensor_msgs std_msgs tf2
#  DEPENDS system_lib
)

include_directories(
  /home/pi/tensorflow
#  /usr/local/include/opencv4
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

# add the executable
add_executable(nimbus_segmentation nimbus_segmentation.cpp)

target_link_libraries( nimbus_segmentation ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} /home/pi/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a /home/pi/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a /lib/arm-linux-gnueabihf/libdl.so.2)