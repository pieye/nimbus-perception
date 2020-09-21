//ROS
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

//OpenCV
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/ocl.hpp>


//Tensorflow
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"

//PCl
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl_ros/point_cloud.h"

#include <sstream>
#include <cmath>
#include <chrono> 

#define IMG_WIDTH 352
#define IMG_HEIGHT 286

using namespace cv;
using namespace std;
using namespace tflite;

const size_t width = 300;
const size_t height = 300;

dnn::Net net;
std::vector<std::string> Names;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub;
ros::Publisher pub_img;
ros::Publisher vis_pub;

static bool readCOCOLabels(std::string fileName){
	//Open the File
	std::ifstream in(fileName.c_str());
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Names.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}


//dection boxes keypoints in 3D space
//
void detect_in_cloud(Mat &src, const PointCloud::ConstPtr& cloud){
    float min_confidence = 0.25;
    ros::param::get("/nimbus_pose/min_confidence", min_confidence);
    int window_size;
    ros::param::get("/nimbus_pose/window_size", window_size);

    Mat blobimg = dnn::blobFromImage(src, 1.0, Size(300, 300), 0.0, true);
	net.setInput(blobimg);
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for(int i=0; i<detectionMat.rows; i++){
		float detect_confidence = detectionMat.at<float>(i, 2);

		if(detect_confidence > min_confidence){
			size_t det_index = (size_t)detectionMat.at<float>(i, 1);
			float x1 = detectionMat.at<float>(i, 3)*src.cols;
			float y1 = detectionMat.at<float>(i, 4)*src.rows;
			float x2 = detectionMat.at<float>(i, 5)*src.cols;
			float y2 = detectionMat.at<float>(i, 6)*src.rows;
			Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
			rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
			putText(src, format("%s", Names[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);
		}
	}
}



//calc 1-sigma (68%) depth and take min & max for box size
//--> use x & y box size of image



//    //RVIZ Visualisation - 3D Sceleton + 2d image
//    visualization_msgs::MarkerArray ma; 
//    //Iterate over all keypoints
//    for(i=0;i<17;i++){
//        //Don't draw nose and ears (they're very unaccurate)
//        if(i != NOSE && i != RIGHT_EAR && i != LEFT_EAR){
//            visualization_msgs::Marker marker;
//            marker.header.frame_id = "nimbus";
//            marker.header.stamp = ros::Time();
//            marker.ns = "nimbus_pose";
//            marker.id = i;
//            marker.type = visualization_msgs::Marker::SPHERE;
//            marker.action = visualization_msgs::Marker::ADD;
//            marker.color.a = 0.0;
//            marker.pose.position.x = 0;
//            marker.pose.position.y = 0;
//            marker.pose.position.z = 0;
//            marker.pose.orientation.x = 0.0;
//            marker.pose.orientation.y = 0.0;
//            marker.pose.orientation.z = 0.0;
//            marker.pose.orientation.w = 1.0;
//    
//            if(confidence[i]>min_confidence){
//                circle(src,location_2d[i],4,Scalar(50, 175, 215),FILLED);
//                if(location_2d[i].y >= 0 && location_2d[i].x >= 0 && location_2d[i].y <= IMG_HEIGHT && location_2d[i].x <= IMG_WIDTH 
//                    && cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x > -100 && cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x < 100){
//                        marker.color.a = 1.0;
//                        float temp_depth = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].z;
//                        for(int ii=-window_size/2;ii<window_size/2;ii++){
//                            for(int jj=-window_size/2;jj<window_size/2;jj++){
//                                std::cout << cloud->points[location_2d[i].x+ii + IMG_WIDTH*location_2d[i].y+jj].z << std::endl;
//                                if(cloud->points[location_2d[i].x+ii + IMG_WIDTH*location_2d[i].y+jj].z < temp_depth)
//                                    temp_depth = cloud->points[location_2d[i].x+ii + IMG_WIDTH*location_2d[i].y+jj].z;
//                            }
//                        }
//                        
//                        ///////////////////////////////////////////////////////////////////////////////////////////
//                        //                               Here is the 3D Pose data!                               //
//                        marker.pose.position.x = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x;
//                        marker.pose.position.y = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].y;
//                        marker.pose.position.z = temp_depth;
//                        ///////////////////////////////////////////////////////////////////////////////////////////
//                }
//            }
//            marker.color.r = 0.196;
//            marker.color.g = 0.686;
//            marker.color.b = 0.843;
//            marker.scale.x = marker.scale.y = marker.scale.z = 0.15;
//            ma.markers.push_back(marker);
//        }
//
//        
//    }
//    vis_pub.publish(ma);
//}

//Callback when new pointcloud arrives
//
void cloud_cb(const PointCloud::ConstPtr& cloud){
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat grHistogram(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, Scalar(0, 0, 0));

    for(int j = 0; j < (IMG_HEIGHT); j++){
        for(int i = 0; i < (IMG_WIDTH); i++){
          uint8_t intensity = std::min(std::max((log(cloud->points[i + IMG_WIDTH*j].intensity)*30), 0.0f), 255.0f);
          grHistogram.at<Vec3b>(j, i)[0] = intensity;
          grHistogram.at<Vec3b>(j, i)[1] = intensity;
          grHistogram.at<Vec3b>(j, i)[2] = intensity;
        }
    }

    detect_in_cloud(grHistogram, cloud);

    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;
    header.seq = 1;
    header.stamp = ros::Time::now(); 
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, grHistogram);
    sensor_msgs::Image img_msg; 
    img_bridge.toImageMsg(img_msg);
    
    pub_img.publish(img_msg); 
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    ROS_INFO_STREAM("nimbus-pose runtime: " << elapsed.count() << " seconds.");
}


int main(int argc,char ** argv){
    // Initialize ROS
    ros::init(argc, argv, "nimbus_detection");
    ros::NodeHandle nh;
    pub_img = nh.advertise<sensor_msgs::Image>("nimbus_detection", 1);
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>( "visualization_marker_array", 0 );

    //Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/nimbus/pointcloud", 1, cloud_cb);

    //tf lite model
    string model_path;
    ros::param::get("/nimbus_pose/model_path", model_path);

    //MobileNetV1
    //net = dnn::readNetFromTensorflow("model_path.c_str()");
    //MobileNetV2
    net = dnn::readNetFromTensorflow("/home/pi/catkin_ws/src/nimbus-perception/nimbus_detection/frozen_inference_graph_V2.pb","/home/pi/catkin_ws/src/nimbus-perception/nimbus_detection/ssd_mobilenet_v2_coco_2018_03_29.pbtxt");

    if (net.empty()){
        cout << "init the model net error";
        exit(-1);
    }

	// Get the names
	bool result = readCOCOLabels(model_path.c_str() + "COCO_labels.txt");
	if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

    while(1){
        ros::spinOnce();
    }
}