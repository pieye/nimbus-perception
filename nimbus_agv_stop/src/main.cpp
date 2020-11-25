#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl_ros/point_cloud.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <chrono> 


typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub_roi;
ros::Publisher pub_voxel;
ros::Publisher pub_outlier;
ros::Publisher pub_ground;

void cloud_cb (const PointCloud::ConstPtr& cloud)
{
    auto start = std::chrono::high_resolution_clock::now();

    //Filter ROI with area inside 3d box
    pcl::CropBox<pcl::PointXYZI> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(-2, 0.6, -0.1, 1.0));
    boxFilter.setMax(Eigen::Vector4f(2, 5, 2, 1.0));
    boxFilter.setInputCloud(cloud);
    PointCloud::Ptr cloud_filtered(new PointCloud);
    boxFilter.filter(*cloud_filtered);

    //Voxelize the point cloud to speed up computation
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud (cloud_filtered);
    sor.setLeafSize (0.02, 0.02, 0.02);
    PointCloud::Ptr cloud_filtered_voxel(new PointCloud);
    sor.filter (*cloud_filtered_voxel);

    //Remove outlier
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;
    outrem.setInputCloud(cloud_filtered_voxel);
    outrem.setRadiusSearch(0.1);
    outrem.setMinNeighborsInRadius(10);
    outrem.setKeepOrganized(true);
    PointCloud::Ptr cloud_filtered_outlier(new PointCloud);
    outrem.filter(*cloud_filtered_outlier);

    //Remove Ground with RANSAC
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    //Set up parameters for our segmentation/ extraction scheme
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE ); //only want points perpendicular to a given axis
    seg.setMaxIterations(500); // this is key (default is 50 and that sucks)
    seg.setNumberOfThreads(4);
    //seg.setSamplesMaxDist()
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1); // keep points within 0.1 m of the plane
    //because we want a specific plane (X-Z Plane) (In camera coordinates the ground plane is perpendicular to the y axis)
    Eigen::Vector3f axis = Eigen::Vector3f(0.0,1.0,0.0); //y axis
    seg.setAxis(axis);
    seg.setEpsAngle(30.0f * (3.149/180.0f) ); // plane can be within 30 degrees of X-Z plane
    seg.setInputCloud(cloud_filtered_outlier);
    seg.segment(*inliers, *coefficients);

    //extract the outlier
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud_filtered_outlier);
    extract.setIndices(inliers);
    extract.setNegative(false);
    PointCloud::Ptr cloud_filtered_ground(new PointCloud);
    extract.filter(*cloud_filtered_ground);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
  
    ROS_INFO_STREAM("inliers: " << inliers->indices.size());
    ROS_INFO_STREAM("Removed Points: " << cloud->points.size()-cloud_filtered_ground->points.size() << " Points. In time: " << elapsed.count());

    //publish the data.
    pub_roi.publish(cloud_filtered);
    pub_voxel.publish(cloud_filtered_voxel);
    pub_outlier.publish(cloud_filtered_outlier);
    pub_ground.publish(cloud_filtered_ground);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "nimbus_agv_emergency");
  ros::NodeHandle nh;

  // Create a ROS publisher for the output point cloud
  pub_roi      = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("roi", 1);
  pub_voxel    = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("voxelized", 1);
  pub_outlier  = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("outlier", 1);
  pub_ground   = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("ground", 1);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/nimbus/pointcloud", 1, cloud_cb);

  // Spin
  ros::spin();
}