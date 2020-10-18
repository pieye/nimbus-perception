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
    float sigma;
    ros::param::get("/nimbus_detection/sigma", sigma);

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

    float sum_x = 0;
    float sum_y = 0;
    float sum_z = 0;

    float var_x = 0;
    float var_y = 0;
    float var_z = 0;

    float sd_x = 0;
    float sd_y = 0;
    float sd_z = 0;

    float min_height = std::numeric_limits<float>::max();
    float max_height = std::numeric_limits<float>::min();
    float min_width  = std::numeric_limits<float>::max();
    float max_width  = std::numeric_limits<float>::min();

    //Max amount of pixels
    float max = (y2-y1)*(x2-x1);

    for(int i = x1; i < (x2); i++){
        for(int j = y1; j < (y2); j++){
            if(isnan(cloud.points[i + IMG_WIDTH*j].x) || isnan(cloud.points[i + IMG_WIDTH*j].y) || isnan(cloud.points[i + IMG_WIDTH*j].z)){
                max--;
            }
            else{
                sum_x += cloud.points[i + IMG_WIDTH*j].x;
                sum_y += cloud.points[i + IMG_WIDTH*j].y;
                sum_z += cloud.points[i + IMG_WIDTH*j].z;

                //get min and max height+width of box to limit estimated size
                if(min_height > cloud.points[i + IMG_WIDTH*j].y)
                    min_height = cloud.points[i + IMG_WIDTH*j].y;

                if(max_height < cloud.points[i + IMG_WIDTH*j].y)
                    max_height = cloud.points[i + IMG_WIDTH*j].y;
                
                if(min_width > cloud.points[i + IMG_WIDTH*j].x)
                    min_width = cloud.points[i + IMG_WIDTH*j].x;

                if(max_width < cloud.points[i + IMG_WIDTH*j].x)
                    max_width = cloud.points[i + IMG_WIDTH*j].x;
            }
        }
    }
    //calc mean
    float mean_x = sum_x/max;
    float mean_y = sum_y/max;
    float mean_z = sum_z/max;

    //Max amount of pixels
    max = (y2-y1)*(x2-x1);

    //Calculate Variance of pointcloud distribution in x,y,z direction
    for(int i = x1; i < (x2); i++){
        for(int j = y1; j < (y2); j++){
            if(isnan(cloud.points[i + IMG_WIDTH*j].x) || isnan(cloud.points[i + IMG_WIDTH*j].y) || isnan(cloud.points[i + IMG_WIDTH*j].z)){
                max--;
            }
            else{
                var_x += (cloud.points[i + IMG_WIDTH*j].x - mean_x) * (cloud.points[i + IMG_WIDTH*j].x - mean_x);
                var_y += (cloud.points[i + IMG_WIDTH*j].y - mean_y) * (cloud.points[i + IMG_WIDTH*j].y - mean_y);
                var_z += (cloud.points[i + IMG_WIDTH*j].z - mean_z) * (cloud.points[i + IMG_WIDTH*j].z - mean_z);
            }
        }
    }

    //calc variance
    var_x /= abs(sum_x);
    var_y /= abs(sum_y);
    var_z /= abs(sum_z);
   
    if(isnan(sum_x) || isnan(sum_y) || isnan(sum_z)){
            marker.color.a = 0.0;
            std::cout << "NAN!" << std::endl;
    }
    else{
        //bounding box center is mean of pointcloud distribution
        marker.pose.position.x = mean_x;
        marker.pose.position.y = mean_y;
        marker.pose.position.z = mean_z;
    }
    //pieye colors
    marker.color.r = 0.196;
    marker.color.g = 0.686;
    marker.color.b = 0.843;

    //calc standard deviation and use it for the bounding box size
    //also check estimated x-sigma size against actual max in bounding box to avoid too large boxes
    if(sqrt(var_x)*sigma > max_width-min_width)
        marker.scale.x = max_width-min_width;
    else
        marker.scale.x = sqrt(var_x)*sigma;

    if(sqrt(var_y)*sigma > max_height-min_height)
        marker.scale.y = max_height-min_height;
    else
        marker.scale.y = sqrt(var_y)*sigma;

    marker.scale.z = sqrt(var_z)*sigma;


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