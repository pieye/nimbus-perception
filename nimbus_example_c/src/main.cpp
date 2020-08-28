#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl_ros/point_cloud.h"

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub;

void cloud_cb (const PointCloud::ConstPtr& cloud)
{
  // Do something with the data (cloud) <-- this example just counts the points
  ROS_INFO_STREAM("Published: " << cloud->points.size() << " Points.");

  //publish the data.
  pub.publish(cloud);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "nimbus_example_c");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/nimbus/pointcloud", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("output", 1);

  // Spin
  ros::spin();
}