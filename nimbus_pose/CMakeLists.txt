cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o3 -std=c++14 -pthread")

# set the project name
project(nimbus_pose)

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
  $ENV{HOME}/tensorflow
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

# add the executable
add_executable(nimbus_pose nimbus_pose.cpp)

#target_link_libraries( nimbus_pose ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a /lib/arm-linux-gnueabihf/libdl.so.2)
target_link_libraries( nimbus_pose ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a /lib/aarch64-linux-gnu/libdl.so.2)
#target_link_libraries( nimbus_pose ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a $ENV{HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build/libflatbuffers.a)