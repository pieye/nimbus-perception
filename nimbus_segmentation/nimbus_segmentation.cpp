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

using namespace cv;
using namespace std;
using namespace tflite;

int model_width;
int model_height;
int model_channels;

#define IMG_WIDTH 352
#define IMG_HEIGHT 286

struct RGB {
    unsigned char blue;
    unsigned char green;
    unsigned char red;
};

const RGB Colors[21] = {{127,127,127} ,  // 0 background
                        {  0,  0,255} ,  // 1 aeroplane
                        {  0,255,  0} ,  // 2 bicycle
                        {255,  0,  0} ,  // 3 bird
                        {255,  0,255} ,  // 4 boat
                        {  0,255,255} ,  // 5 bottle
                        {255,255,  0} ,  // 6 bus
                        {  0,  0,127} ,  // 7 car
                        {  0,127,  0} ,  // 8 cat
                        {127,  0,  0} ,  // 9 chair
                        {127,  0,127} ,  //10 cow
                        {  0,127,127} ,  //11 diningtable
                        {127,127,  0} ,  //12 dog
                        {127,127,255} ,  //13 horse
                        {127,255,127} ,  //14 motorbike
                        {255,127,127} ,  //15 person
                        {255,127,255} ,  //16 potted plant
                        {127,255,255} ,  //17 sheep
                        {255,255,127} ,  //18 sofa
                        {  0, 91,127} ,  //19 train
                        { 91,  0,127} }; //20 tv monitor

std::unique_ptr<Interpreter> interpreter;

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
ros::Publisher pub_img;
ros::Publisher pointcloud_pub;

PointCloudRGB::Ptr segmented_cloud(new PointCloudRGB);


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


//segment in 3D space
//
void detect_in_cloud(Mat &src, const PointCloud::ConstPtr& cloud){
    int i,j,k,mi;
    float mx,v;
    float *data;
    RGB *rgb;
    static Mat image;
    static Mat frame(model_width,model_height,CV_8UC3);
    static Mat blend(src.cols   ,src.rows    ,CV_8UC3);

    prepare_image(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    // run model
    interpreter->Invoke();

    const float* heatmapShape = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* offsetShape = interpreter->tensor(interpreter->outputs()[1])->data.f;
    
    // get most likely object per pixel
    data = interpreter->tensor(interpreter->outputs()[0])->data.f;
    rgb = (RGB *)frame.data;

    for(i=0;i<model_height;i++){
        for(j=0;j<model_width;j++){
            for(mi=-1,mx=0.0,k=0;k<21;k++){
                v = data[21*(i*model_width+j)+k];
                if(v>mx){ mi=k; mx=v; }
            }
            rgb[j+i*model_width] = Colors[mi];
            //cloud->points[location_2d[i].x + IMG_WIDTH*location_2d[i].y].intensity = mi;
        }
    }

    //merge output into frame
    cv::resize(frame, blend, Size(src.cols,src.rows),INTER_NEAREST);
    cv::addWeighted(src, 0.5, blend, 0.5, 0.0, src);

    //Move valid points into the point cloud and the corresponding images
    for(int i = 0; i < (IMG_WIDTH); i++){
        for(int j = 0; j < (IMG_HEIGHT); j++){
            segmented_cloud->points[j*IMG_WIDTH + i].x = cloud->points[j*IMG_WIDTH + i].x;
            segmented_cloud->points[j*IMG_WIDTH + i].y = cloud->points[j*IMG_WIDTH + i].y;
            segmented_cloud->points[j*IMG_WIDTH + i].z = cloud->points[j*IMG_WIDTH + i].z;
            segmented_cloud->points[j*IMG_WIDTH + i].r = src.at<Vec3b>(j, i)[0];
            segmented_cloud->points[j*IMG_WIDTH + i].g = src.at<Vec3b>(j, i)[1];
            segmented_cloud->points[j*IMG_WIDTH + i].b = src.at<Vec3b>(j, i)[2];
        }
    }

    pcl_conversions::toPCL(ros::Time::now(), segmented_cloud->header.stamp);
    pointcloud_pub.publish(segmented_cloud);
}


//Callback when new pointcloud arrives
//
void cloud_cb(const PointCloud::ConstPtr& cloud){
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat grHistogram(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, Scalar(0, 0, 0));
    cv::Mat outt(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, Scalar(0, 0, 0));
    cv::Mat color(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, Scalar(0, 0, 0));

    for(int j = 0; j < (IMG_HEIGHT); j++){
        for(int i = 0; i < (IMG_WIDTH); i++){
          uint8_t intensity = std::min(std::max((log(cloud->points[i + IMG_WIDTH*j].intensity)*30), 0.0f), 255.0f);
          grHistogram.at<uchar>(j, i) = intensity;
        }
    }

    equalizeHist( grHistogram, outt );
    cv::cvtColor(outt, color, cv::COLOR_GRAY2RGB);

    detect_in_cloud(color, cloud);

    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;
    header.seq = 1;
    header.stamp = ros::Time::now(); 
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, color);
    sensor_msgs::Image img_msg; 
    img_bridge.toImageMsg(img_msg);
    
    pub_img.publish(img_msg); 
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    ROS_INFO_STREAM("nimbus_segmentation runtime: " << elapsed.count() << " seconds.");
}


int main(int argc,char ** argv){
    // Initialize ROS
    ros::init(argc, argv, "nimbus_segmentation");
    ros::NodeHandle nh;
    pub_img = nh.advertise<sensor_msgs::Image>("nimbus_segmentation", 1);
    pointcloud_pub = nh.advertise<PointCloudRGB>("segmented_pointcloud", 1);

    //Initialize Pointcloud
    segmented_cloud->points.resize(IMG_WIDTH*IMG_HEIGHT);
    segmented_cloud->width = IMG_WIDTH;
    segmented_cloud->height = IMG_HEIGHT;
    segmented_cloud->is_dense = false;        //<-- because invalid points are being set to NAN
    segmented_cloud->header.frame_id = "nimbus";

    //Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe<pcl::PointCloud<pcl::PointXYZI>>("/nimbus/pointcloud", 1, cloud_cb);

    //tf lite model
    string model_path;
    ros::param::get("/nimbus_segmentation/model_path", model_path);
    std::unique_ptr<FlatBufferModel> model = FlatBufferModel::BuildFromFile(model_path.c_str());

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