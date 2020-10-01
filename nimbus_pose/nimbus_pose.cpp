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

using namespace cv;
using namespace std;
using namespace tflite;

int model_width;
int model_height;
int model_channels;

#define IMG_WIDTH 352
#define IMG_HEIGHT 286

#define NOSE            0
#define LEFT_EYE        1
#define RIGHT_EYE       2
#define LEFT_EAR        3
#define RIGHT_EAR       4
#define LEFT_SHOULDER   5    
#define RIGHT_SHOULDER  6   
#define LEFT_ELBOW      7
#define RIGHT_ELBOW     8
#define LEFT_WRIST      9
#define RIGHT_WRIST     10
#define LEFT_HIP        11
#define RIGHT_HIP       12
#define LEFT_KNEE       13
#define RIGHT_KNEE      14
#define LEFT_ANKLE      15
#define RIGHT_ANKLE     16

std::unique_ptr<Interpreter> interpreter;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
ros::Publisher pub;
ros::Publisher pub_img;
ros::Publisher vis_pub;


//resize input image
//
void prepare_image(float* out, Mat &src){
    int i,Len;
    float f;
    uint8_t *in;
    static Mat image;

    // copy image to input as input tensor
    cv::resize(src, image, Size(model_width,model_height),INTER_NEAREST);

    in=image.data;
    Len=image.rows*image.cols*image.channels();
    for(i=0;i<Len;i++){
        f = in[i];
        out[i] = (f - 127.5f) / 127.5f;
    }
}

//detect keypoints in 3D space
//
void detect_in_cloud(Mat &src, const PointCloud::ConstPtr& cloud){
    int i,x,y,j;
    static Point heatmap[17];
    static float confidence[17];
    static Point location_2d[17];
    float min_confidence;
    ros::param::get("/nimbus_pose/min_confidence", min_confidence);
    int window_size;
    ros::param::get("/nimbus_pose/window_size", window_size);

    prepare_image(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    // run model
    interpreter->Invoke();

    const float* heatmapShape = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* offsetShape = interpreter->tensor(interpreter->outputs()[1])->data.f;

    // Finds the (row, col) locations of where the keypoints are most likely to be.
    for(i=0;i<17;i++){
        confidence[i]=heatmapShape[i]; 
        for(y=0;y<9;y++){
            for(x=0;x<9;x++){
                j=17*(9*y+x)+i;
                if(heatmapShape[j]>confidence[i]){
                    confidence[i]=heatmapShape[j]; heatmap[i].x=x; heatmap[i].y=y;
                }
            }
        }
    }

    // Calculating the x and y coordinates of the keypoints with offset adjustment.
    for(i=0;i<17;i++){
        x=heatmap[i].x; y=heatmap[i].y; j=34*(9*y+x)+i;
        location_2d[i].y=(y*src.rows)/8 + offsetShape[j   ];
        location_2d[i].x=(x*src.cols)/8 + offsetShape[j+17];
    }

    //RVIZ Visualisation - 3D Sceleton + 2d image
    visualization_msgs::MarkerArray ma; 
    //Iterate over all keypoints
    for(i=0;i<17;i++){
        //Don't draw nose and ears (they're very unaccurate)
        if(i != NOSE && i != RIGHT_EAR && i != LEFT_EAR){
            visualization_msgs::Marker marker;
            marker.header.frame_id = "nimbus";
            marker.header.stamp = ros::Time();
            marker.ns = "nimbus_pose";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.color.a = 0.0;
            marker.pose.position.x = 0;
            marker.pose.position.y = 0;
            marker.pose.position.z = 0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
    
            if(confidence[i]>min_confidence){
                circle(src,location_2d[i],4,Scalar(50, 175, 215),FILLED);
                if(location_2d[i].y >= 0 && location_2d[i].x >= 0 && location_2d[i].y <= IMG_HEIGHT && location_2d[i].x <= IMG_WIDTH 
                    && cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x > -100 && cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x < 100){
                        marker.color.a = 1.0;
                        float temp_depth = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].z;
                        for(int ii=-window_size/2;ii<window_size/2;ii++){
                            for(int jj=-window_size/2;jj<window_size/2;jj++){
                                if(cloud->points[location_2d[i].x+ii + IMG_WIDTH*location_2d[i].y+jj].z < temp_depth)
                                    temp_depth = cloud->points[location_2d[i].x+ii + IMG_WIDTH*location_2d[i].y+jj].z;
                            }
                        }
                        
                        ///////////////////////////////////////////////////////////////////////////////////////////
                        //                               Here is the 3D Pose data!                               //
                        marker.pose.position.x = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].x;
                        marker.pose.position.y = cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].y;
                        marker.pose.position.z = temp_depth;
                        ///////////////////////////////////////////////////////////////////////////////////////////
                }
            }
            marker.color.r = 0.196;
            marker.color.g = 0.686;
            marker.color.b = 0.843;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.15;
            ma.markers.push_back(marker);
        }

        
    }
    vis_pub.publish(ma);

    //Draw lines on image
    if(confidence[LEFT_SHOULDER]>min_confidence){
        if(confidence[RIGHT_SHOULDER]>min_confidence){
            line(src,location_2d[LEFT_SHOULDER],location_2d[ RIGHT_SHOULDER],Scalar(50, 175, 215),2);
        }
        if(confidence[LEFT_ELBOW]>min_confidence){
            line(src,location_2d[LEFT_SHOULDER],location_2d[ LEFT_ELBOW],Scalar(50, 175, 215),2);
        }
        if(confidence[LEFT_HIP]>min_confidence){
            line(src,location_2d[LEFT_SHOULDER],location_2d[LEFT_HIP],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[RIGHT_SHOULDER]>min_confidence){
        if(confidence[RIGHT_ELBOW]>min_confidence){
            line(src,location_2d[RIGHT_SHOULDER],location_2d[RIGHT_ELBOW],Scalar(50, 175, 215),2);
        }
        if(confidence[RIGHT_HIP]>min_confidence){
            line(src,location_2d[RIGHT_SHOULDER],location_2d[RIGHT_HIP],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[LEFT_ELBOW]>min_confidence){
        if(confidence[LEFT_WRIST]>min_confidence){
            line(src,location_2d[LEFT_ELBOW],location_2d[LEFT_WRIST],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[ RIGHT_ELBOW]>min_confidence){
        if(confidence[RIGHT_WRIST]>min_confidence){
            line(src,location_2d[RIGHT_ELBOW],location_2d[RIGHT_WRIST],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[LEFT_HIP]>min_confidence){
        if(confidence[RIGHT_HIP]>min_confidence){
            line(src,location_2d[LEFT_HIP],location_2d[RIGHT_HIP],Scalar(50, 175, 215),2);
        }
        if(confidence[LEFT_KNEE]>min_confidence){
            line(src,location_2d[LEFT_HIP],location_2d[LEFT_KNEE],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[LEFT_KNEE]>min_confidence){
        if(confidence[LEFT_ANKLE]>min_confidence){
            line(src,location_2d[LEFT_KNEE],location_2d[LEFT_ANKLE],Scalar(50, 175, 215),2);
        }
    }
    if(confidence[RIGHT_KNEE]>min_confidence){
        if(confidence[RIGHT_HIP]>min_confidence){
            line(src,location_2d[RIGHT_KNEE],location_2d[RIGHT_HIP],Scalar(50, 175, 215),2);
        }
        if(confidence[RIGHT_ANKLE]>min_confidence){
            line(src,location_2d[RIGHT_KNEE],location_2d[RIGHT_ANKLE],Scalar(50, 175, 215),2);
        }
    }
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
    ros::init(argc, argv, "nimbus_pose");
    ros::NodeHandle nh;
    pub_img = nh.advertise<sensor_msgs::Image>("nimbus_pose", 1);
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>( "visualization_marker_array", 0 );

    //Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/nimbus/pointcloud", 1, cloud_cb);


    //tf lite model
    std::string path = ros::package::getPath("nimbus_pose");
    string model_file;
    ros::param::get("/nimbus_pose/model_file", model_file);
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile((path + "/" + model_file).c_str());

    //interpreter
    ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      

    //Input from input meta data
    int input = interpreter->inputs()[0];
    model_height   = interpreter->tensor(input)->dims->data[1];
    model_width    = interpreter->tensor(input)->dims->data[2];
    model_channels = interpreter->tensor(input)->dims->data[3];

    while(1){
        ros::spinOnce();
    }
}