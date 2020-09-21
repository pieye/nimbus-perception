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

const size_t width = 300;
const size_t height = 300;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;

int marker_counter = 0;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub;
ros::Publisher pub_img;
ros::Publisher vis_pub;

visualization_msgs::Marker DeleteMarker(){
    visualization_msgs::Marker deleteMarker;
    if(marker_counter > 0){
        deleteMarker.header.frame_id = "nimbus";
        deleteMarker.header.stamp = ros::Time();
        deleteMarker.ns = "nimbus_detection";
        deleteMarker.id = marker_counter;
        deleteMarker.ns = "points";
        deleteMarker.type = visualization_msgs::Marker::CUBE;
        deleteMarker.action = visualization_msgs::Marker::DELETE;
        deleteMarker.color.a = 0.0;
        marker_counter--;
    }
    return deleteMarker;
}

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

    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 0.2;

    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    return marker;
}

//Calculate mean and standard deviation of bounding box 
//and use 1-sigma confidence bounds to approximate cluster size
visualization_msgs::Marker Calculate3DBoundingBox(int id, float y1, float x1, float y2, float x2, const PointCloud::ConstPtr& cloud){
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

    float max = (y2-y1)*(x2-x1);
    for(int i = x1; i < (x2); i++){
        for(int j = y1; j < (y2); j++){
            if(isnan(cloud->points[i + IMG_WIDTH*j].x) || isnan(cloud->points[i + IMG_WIDTH*j].y) || isnan(cloud->points[i + IMG_WIDTH*j].z)){
                max--;
            }
            else{
                sum_x += cloud->points[i + IMG_WIDTH*j].x;
                sum_y += cloud->points[i + IMG_WIDTH*j].y;
                sum_z += cloud->points[i + IMG_WIDTH*j].z;
            }
        }
    }

    //calc mean
    float mean_x = sum_x/max;
    float mean_y = sum_y/max;
    float mean_z = sum_z/max;

    max = (y2-y1)*(x2-x1);
    for(int i = x1; i < (x2); i++){
        for(int j = y1; j < (y2); j++){
            if(isnan(cloud->points[i + IMG_WIDTH*j].x) || isnan(cloud->points[i + IMG_WIDTH*j].y) || isnan(cloud->points[i + IMG_WIDTH*j].z)){
                max--;
            }
            else{
                var_x += (cloud->points[i + IMG_WIDTH*j].x - mean_x) * (cloud->points[i + IMG_WIDTH*j].x - mean_x);
                var_y += (cloud->points[i + IMG_WIDTH*j].y - mean_y) * (cloud->points[i + IMG_WIDTH*j].y - mean_y);
                var_z += (cloud->points[i + IMG_WIDTH*j].z - mean_z) * (cloud->points[i + IMG_WIDTH*j].z - mean_z);
            }
        }
    }

    //calc variance
    var_x /= sum_x;
    var_y /= sum_y;
    var_z /= sum_z;

    std::cout << "sum_x" << sum_x <<  "var_x" << var_x << std::endl;

    //variance negative?????
    //problem here --> check math
    //
    //
    //
    //bounding boxes are rotated wrong??? Or variance is wrong?!

   
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
    marker.scale.x = sqrt(var_x)*sigma;
    marker.scale.y = sqrt(var_y)*sigma;
    marker.scale.z = sqrt(var_z)*sigma;

    std::cout << "x " << sqrt(var_x) << "    y " << sqrt(var_y) << "    z " << sqrt(var_z) <<  std::endl;


    return marker;
}

static bool readCOCOLabels(std::string fileName){
	//Open the File
	std::ifstream in(fileName.c_str());
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}


//dection boxes keypoints in 3D space
//
void detect_in_cloud(Mat &src, const PointCloud::ConstPtr& cloud){
    float min_confidence = 0.25;
    ros::param::get("/nimbus_detection/min_confidence", min_confidence);
    
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    //copy image to input as input tensor
    cv::resize(src, image, Size(width,height));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      

    //run the model
    interpreter->Invoke();      

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

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

    //while(marker_counter > temp_marker_count){
    //    ma.markers.push_back(DeleteMarker());
    //}

    marker_counter = temp_marker_count;









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
    
    vis_pub.publish(ma);
}

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

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("/home/pi/catkin_ws/src/nimbus-perception/nimbus_detection/detect.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

	// Get the names
	bool result = readCOCOLabels("/home/pi/catkin_ws/src/nimbus-perception/nimbus_detection/COCO_labels.txt");
	if(!result)
	{
        cout << "loading labels failed";
        exit(-1);
	}

    while(1){
        ros::spinOnce();
    }
}