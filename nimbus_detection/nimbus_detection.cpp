//ROS
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include <ros/package.h>

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

//size of network input
const size_t width = 300;
const size_t height = 300;

using namespace cv;
using namespace std;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

int marker_counter = 0;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub;
ros::Publisher pub_img;
ros::Publisher vis_pub;

//delete old marker which are not used anymore
//
visualization_msgs::Marker DeleteMarker(){
    visualization_msgs::Marker deleteMarker;
    if(marker_counter >= 0){
        deleteMarker.header.frame_id = "nimbus";
        deleteMarker.header.stamp = ros::Time();
        deleteMarker.ns = "nimbus_detection";
        deleteMarker.id = marker_counter;
        deleteMarker.type = visualization_msgs::Marker::CUBE;
        deleteMarker.action = visualization_msgs::Marker::DELETE;
        deleteMarker.color.a = 0.0;
        marker_counter--;
    }
    return deleteMarker;
}

//Create RVIZ text marker with label class
//
visualization_msgs::Marker addText(float x, float y, float z, float depth, int id, string Name){
    visualization_msgs::Marker marker;
    marker.header.frame_id = "nimbus";
    marker.header.stamp = ros::Time();
    marker.ns = "nimbus_detection";
    marker.id = id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = 1.0;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z-depth/2;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.text = Name;
    marker.scale.z = 0.2;
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    return marker;
}

//Calculate mean and standard deviation of bounding box 
//and use x-sigma confidence bounds to approximate cluster size
//return rviz CUBE marker
visualization_msgs::Marker Calculate3DBoundingBox(int id, float y1, float x1, float y2, float x2, PointCloud cloud){
    //restrict bounding boxes on the image plane (no negative pixel locations)
    if(x1 < 0)
        x1 = 1;
    if(y1 < 0)
        y1 = 1;
    if(x2 >= IMG_WIDTH-1)
        x2 = IMG_HEIGHT-1;
    if(y2 >= IMG_HEIGHT-1)
        y2 = IMG_HEIGHT-1;

    float box_size;
    ros::param::get("/nimbus_detection/box_size", box_size);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "nimbus";
    marker.header.stamp = ros::Time();
    marker.ns = "nimbus_detection";
    marker.id = id;
    marker.type = visualization_msgs::Marker::CUBE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = 0.7;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    //Max amount of pixels
    std::vector<float> x_distribution;
    std::vector<float> y_distribution;
    std::vector<float> z_distribution;
    float max = (y2-y1)*(x2-x1);

    //only use every second point to speed up the computation
    for(int i = x1; i < (x2); i=i+2){
        for(int j = y1; j < (y2); j=j+2){
            if(isnan(cloud.points[i + IMG_WIDTH*j].x) || isnan(cloud.points[i + IMG_WIDTH*j].y) || isnan(cloud.points[i + IMG_WIDTH*j].z)){
                max--;
            }
            else{

                x_distribution.push_back(cloud.points[i + IMG_WIDTH*j].x);
                y_distribution.push_back(cloud.points[i + IMG_WIDTH*j].y);
                z_distribution.push_back(cloud.points[i + IMG_WIDTH*j].z);
            }
        }
    }

    //sort x, y and z distributions in order to find the median + peak
    std::sort(x_distribution.begin(), x_distribution.end());
    std::sort(y_distribution.begin(), y_distribution.end());
    std::sort(z_distribution.begin(), z_distribution.end());

    //bounding box center is median of pointcloud distribution
    marker.pose.position.x = x_distribution[z_distribution.size()/2];
    marker.pose.position.y = y_distribution[z_distribution.size()/2];
    marker.pose.position.z = z_distribution[z_distribution.size()/2]; 

    //pieye colors
    marker.color.r = 0.196;
    marker.color.g = 0.686;
    marker.color.b = 0.843;


    //x and y size threshold
    float x_threshold = box_size;
    float y_threshold = box_size;
    float z_threshold = box_size/2;
    marker.scale.x = x_distribution[x_distribution.size()*(0.5+x_threshold/2)] - x_distribution[x_distribution.size()*(0.5-x_threshold/2)];
    marker.scale.y = y_distribution[y_distribution.size()*(0.5+y_threshold/2)] - y_distribution[y_distribution.size()*(0.5-y_threshold/2)];
    marker.scale.z = z_distribution[z_distribution.size()*(0.5+z_threshold/2)] - z_distribution[z_distribution.size()*(0.5-z_threshold/2)];

    return marker;
}

//import label list for visualisation
static bool readCOCOLabels(std::string fileName){
	std::ifstream in(fileName.c_str());
	if(!in.is_open()) return false;

	std::string str;
	while (std::getline(in, str))
	{
		if(str.size()>0) Labels.push_back(str);
	}
	in.close();
	return true;
}


//dection boxes in 3D space
//
void detect_in_cloud(Mat &src, PointCloud cloud){
    float min_confidence = 0.55;
    ros::param::get("/nimbus_detection/min_confidence", min_confidence);
    
    Mat image;
    int cam_width  = src.cols;
    int cam_height = src.rows;

    //copy image to input as input tensor
    cv::resize(src, image, Size(width,height));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    //tensorflow CPU inference setting
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      

    //run the model
    interpreter->Invoke();      

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes   = interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores    = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections      = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //counter for amount of visualisation (rviz) markers in this call
    int temp_marker_count = 0;
    visualization_msgs::MarkerArray ma; 
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > min_confidence){
            int  det_index = (int)detection_classes[i]+1;
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
            putText(src, format("%s", Labels[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);

            visualization_msgs::Marker marker = Calculate3DBoundingBox(temp_marker_count, y1, x1, y2, x2, cloud);
            ma.markers.push_back(marker);
            temp_marker_count++;
            ma.markers.push_back(addText(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z, marker.scale.z, temp_marker_count, Labels[det_index]));
            temp_marker_count++;
        }
    }

    //delete old markes
    while(marker_counter > temp_marker_count){
        ma.markers.push_back(DeleteMarker());
    }
    ma.markers.push_back(DeleteMarker());
    marker_counter = temp_marker_count;
   
    vis_pub.publish(ma);
}

//Callback when new pointcloud arrives
//
void cloud_cb(PointCloud cloud){
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat grHistogram(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, Scalar(0, 0, 0));

    //create cv2 image from pointcloud
    for(int j = 0; j < (IMG_HEIGHT); j++){
        for(int i = 0; i < (IMG_WIDTH); i++){
          uint8_t intensity = std::min(std::max((log(cloud.points[i + IMG_WIDTH*j].intensity)*30), 0.0f), 255.0f);
          grHistogram.at<Vec3b>(j, i)[0] = intensity;
          grHistogram.at<Vec3b>(j, i)[1] = intensity;
          grHistogram.at<Vec3b>(j, i)[2] = intensity;
        }
    }

    //run object detection and visualisation
    detect_in_cloud(grHistogram, cloud);

    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;
    header.seq = 1;
    header.stamp = ros::Time::now(); 
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, grHistogram);
    sensor_msgs::Image img_msg; 
    img_bridge.toImageMsg(img_msg);
    
    //publish debug image
    pub_img.publish(img_msg); 

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    ROS_INFO_STREAM("nimbus-detection runtime: " << elapsed.count() << " seconds.");
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
    string model_file;
    string labels_file;
    ros::param::get("/nimbus_detection/model_file", model_file);
    ros::param::get("/nimbus_detection/labels_file", labels_file);

    std::string path = ros::package::getPath("nimbus_detection");

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile((path + "/" + model_file).c_str());

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

	// Get the names
	bool result = readCOCOLabels(path + "/" + labels_file);
    if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

    while(1){
        ros::spinOnce();
    }
}