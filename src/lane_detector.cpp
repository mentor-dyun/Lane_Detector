/*
 * lane_detector.cpp
 *
 *      Author:
 *         Nicolas Acero
 */

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <lane_detector/preprocessor.h>
#include <lane_detector/featureExtractor.h>
#include <lane_detector/fitting.h>
#include <cv.h>
#include <sstream>
#include <dynamic_reconfigure/server.h>
#include <lane_detector/DetectorConfig.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <lane_detector/LaneDetector.hh>
#include <lane_detector/mcv.hh>
#include <lane_detector/utils.h>
#include <ros/package.h>
#include <boost/filesystem.hpp>

#include <geometry_msgs/PoseStamped.h>
#include <rosgraph_msgs/Clock.h>

cv_bridge::CvImagePtr currentFrame_ptr;
Preprocessor preproc;
FeatureExtractor extractor;
Fitting fitting_phase;
image_transport::Publisher resultImg_pub;
ros::Publisher lane_pub;
lane_detector::DetectorConfig dynConfig;
LaneDetector::CameraInfo cameraInfo;
LaneDetector::LaneDetectorConf lanesConf;

geometry_msgs::PoseStamped gnss_pose;
lane_detector::point localizer_position;
double localizer_yaw, localizer_pitch, localizer_roll;
rosgraph_msgs::Clock simTime;

/**
 * readCameraInfo reads and sets the camera parameters if received on topic "camera_info".
 * Otherwise the parameters are set with some constant values related to the camera used
 * in our experiments.
 */
void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr& cm, bool* done) {
/*
  if(cm != NULL) {
    cameraInfo.focalLength.x = cm->P[0];
    cameraInfo.focalLength.y = cm->P[5];
    cameraInfo.opticalCenter.x = cm->P[2];
    cameraInfo.opticalCenter.y = cm->P[6];
    cameraInfo.imageWidth = cm->width;
    cameraInfo.imageHeight = cm->height;
    cameraInfo.cameraHeight = dynConfig.camera_height;
    cameraInfo.pitch = dynConfig.camera_pitch * CV_PI/180;
    cameraInfo.yaw = 0.0;
  }
  else {
    cameraInfo.focalLength.x = 270.076996;
    cameraInfo.focalLength.y = 300.836426;
    cameraInfo.opticalCenter.x = 325.678818;
    cameraInfo.opticalCenter.y = 250.211312;
    cameraInfo.imageWidth = 640;
    cameraInfo.imageHeight = 480;
    cameraInfo.cameraHeight = dynConfig.camera_height;
    cameraInfo.pitch = dynConfig.camera_pitch * CV_PI/180;
    cameraInfo.yaw = 0.0;
  }
*/

  cameraInfo.focalLength.x = 270.076996;
  cameraInfo.focalLength.y = 300.836426;
  cameraInfo.opticalCenter.x = 325.678818;
  cameraInfo.opticalCenter.y = 250.211312;

  cameraInfo.imageWidth = 640;
  cameraInfo.imageHeight = 480;
  cameraInfo.cameraHeight = dynConfig.camera_height;
  cameraInfo.pitch = 0.7; //dynConfig.camera_pitch * CV_PI/180;
  cameraInfo.yaw = 0.0;

  /*
  cameraInfo.focalLength.x = 707.0912; // 7.5mm
  cameraInfo.focalLength.y = 707.0912; // 7.5mm
  cameraInfo.opticalCenter.x = 601.8873; //1.6890;
  cameraInfo.opticalCenter.y = 183.1104; //0.400;

  cameraInfo.imageWidth = 640;
  cameraInfo.imageHeight = 480;
  cameraInfo.cameraHeight = dynConfig.camera_height;
  cameraInfo.pitch = 0.6; //dynConfig.camera_pitch * CV_PI/180;
  cameraInfo.yaw = 0.0;
  */

  *done = true;
}

//Callback function for Dynamic Reconfigre
void configCallback(lane_detector::DetectorConfig& config, uint32_t level)
{
        preproc.setConfig(config);
        extractor.setConfig(config);
        fitting_phase.setConfig(config);
        dynConfig = config;
        ROS_DEBUG("Config was set");
}

//Callback function for topic "lane_detector/driving_orientation"
void drivingOrientationCB(const std_msgs::Int32::ConstPtr& driving_orientation)
{
  if(driving_orientation->data == 0)
      fitting_phase.setDrivingOrientation(lane_detector::on_the_right);
  else if(driving_orientation->data == 1)
      fitting_phase.setDrivingOrientation(lane_detector::on_the_left);
}

//Callback function for topic "/gnss_pose"
void poseCB(const geometry_msgs::PoseStamped msg) 
{
  /*
  printf( "--- @%f sec\n", simTime.clock.toSec() );
  printf("  Pos X = %f m, Pos Y = %f m, Pos Z = %f m\n", receive_buffer[POS_X_IDX], receive_buffer[POS_Y_IDX], receive_buffer[POS_Z_IDX]);
  printf("  Roll = %f deg, Pitch = %f deg, Yaw = %f deg\n", receive_buffer[ROLL_IDX], receive_buffer[PITCH_IDX], receive_buffer[YAW_IDX]);
  */

  // Save current position information
  gnss_pose.pose.position.x = msg.pose.position.x;
  gnss_pose.pose.position.y = msg.pose.position.y;
  gnss_pose.pose.position.z = msg.pose.position.z;
  gnss_pose.pose.orientation = msg.pose.orientation;

  localizer_position = lane_detector::point(gnss_pose.pose.position.y, gnss_pose.pose.position.x);
  lane_detector::toEulerAngle(gnss_pose.pose.orientation, localizer_yaw, localizer_pitch, localizer_roll);

}

void clockCB(const rosgraph_msgs::Clock clock) 
{
  simTime = clock;
}

void processImage(LaneDetector::CameraInfo& cameraInfo, LaneDetector::LaneDetectorConf& lanesConf) 
{
  if (currentFrame_ptr) 
  {
    //information paramameters of the IPM transform
    LaneDetector::IPMInfo ipmInfo;

    // detect bounding boxes arround the lanes
    std::vector<LaneDetector::Box> boxes;

    cv::Mat processed_bgr = currentFrame_ptr->image;
    preproc.preprocess(currentFrame_ptr->image, processed_bgr, ipmInfo, cameraInfo);

    cv::Mat preprocessed = processed_bgr.clone();
    lane_detector::utils::scaleMat(processed_bgr, processed_bgr);

    if (processed_bgr.channels() == 1) 
        cv::cvtColor(processed_bgr, processed_bgr, CV_GRAY2BGR);

    //cv::imshow("Pre", processed_bgr);

    extractor.extract(processed_bgr, preprocessed, boxes);
    lane_detector::Lane current_lane = fitting_phase.fitting(currentFrame_ptr->image, processed_bgr, preprocessed, ipmInfo, cameraInfo, boxes);
    lane_pub.publish(current_lane);

    cv::Mat img = currentFrame_ptr->image;

    // Calculate steering offset - this is used to overlay info on image
    // Same logic used in he simulink_gateway
    if (current_lane.guide_line.size() > 0)
    {
        int center = img.cols/2;
        int steering_angle_degrees = center - current_lane.guide_line[3].x;
	double alpha = 0.5;
	int baseline = 0;

	// Draw transparent rectangle fill that will be the background border for the display
	// Top 
        cv::Mat roi_top = img(cv::Rect(2, 2, 640-4, 52));
        cv::Mat color_top(roi_top.size(), CV_8UC3, CV_RGB(0,0,0));
        cv::addWeighted(color_top, alpha, roi_top, 1.0 - alpha, 0.0, roi_top);

	// Bottom
        alpha = 0.3;
        cv::Mat roi_bottom = img(cv::Rect(4, 394, 634, 82));
	cv::Mat color_bottom(roi_bottom.size(), CV_8UC3, CV_RGB(0,0,0));
        cv::addWeighted(color_bottom, alpha, roi_bottom, 1.0 - alpha, 0.0, roi_bottom);	

	// Test
	// Draw arrow line
	int thickness = 1;
	int line_type = CV_AA; // anti-aliased
	int shift = 0;
	double tipLength = 0.05;
	char text_pose[200];

	// Draw 3-axis
	// x-axis (roll)
	/*
	cv::arrowedLine(img, cv::Point(10,100), cv::Point(90,60), CV_RGB(255,0,0), thickness, line_type, shift, tipLength);
	cv::putText(img,                                // image
                    "Vehicle Position",                          // text
                    cv::Point(10, 15),                  // top-left position
                    cv::FONT_HERSHEY_DUPLEX,            // font
                    0.5,                                // font height
                    CV_RGB(255,255,255),                // font color
                    1);                                 // font thickness

        // y-axis (pitch)
	cv::arrowedLine(img, cv::Point(10,100), cv::Point(80,160), CV_RGB(255,0,0), thickness, line_type, shift, tipLength);
       
        // z-axis (pitch)
        cv::arrowedLine(img, cv::Point(10,100), cv::Point(10,20), CV_RGB(255,0,0), thickness, line_type, shift, tipLength);
	*/

	
	// Draw text - vehicle position
	/*
  	printf( "--- @%f sec\n", simTime.clock.toSec() );
  	printf("  Pos X = %f m, Pos Y = %f m, Pos Z = %f m\n", receive_buffer[POS_X_IDX], receive_buffer[POS_Y_IDX], receive_buffer[POS_Z_IDX]);
  	printf("  Roll = %f deg, Pitch = %f deg, Yaw = %f deg\n", receive_buffer[ROLL_IDX], receive_buffer[PITCH_IDX], receive_buffer[YAW_IDX]);
  	*/

	// Clock right-align text to the right side
	sprintf(text_pose, "@%f sec", simTime.clock.toSec() );
	cv::Size textSize = getTextSize(text_pose, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
        cv::putText(img, text_pose, cv::Point(640-10-textSize.width,15), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);

	// Pos X
        sprintf(text_pose, "Pos X = %f m", gnss_pose.pose.position.x);
	cv::putText(img, text_pose, cv::Point(10,15), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);
	// Pos Y
	sprintf(text_pose, "Pos Y = %f m", gnss_pose.pose.position.y);
        cv::putText(img, text_pose, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);
	// Pos Z
	sprintf(text_pose, "Pos Z = %f m", gnss_pose.pose.position.z);
        cv::putText(img, text_pose, cv::Point(10,45), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);


	// Roll
        sprintf(text_pose, "Roll  = %f deg", DEG(localizer_pitch));
        cv::putText(img, text_pose, cv::Point(240,15), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);
        // Pitch
        sprintf(text_pose, "Pitch = %f deg", DEG(localizer_roll));
        cv::putText(img, text_pose, cv::Point(240,30), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);
        // Yaw
        sprintf(text_pose, "Yaw  = %f deg", DEG(localizer_yaw));
        cv::putText(img, text_pose, cv::Point(240,45), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255), 1);


	// The roll and pitch is swapped, trying to match what simulink_gateway is outputing
        //sprintf(text_pose, "Roll = %f deg, Pitch = %f deg, Yaw = %f deg", DEG(localizer_pitch), DEG(localizer_roll), DEG(localizer_yaw));

	// Draw text - Steering Angle and degree
	char text[200];
	sprintf(text, "Steering Angle: %d", steering_angle_degrees);
	cv::putText(img, 				// image
		    text, 				// text
		    cv::Point(center - 120, 420), 	// top-left position
		    cv::FONT_HERSHEY_DUPLEX,		// font
		    0.7,				// font height
		    CV_RGB(255,255,255), 		// font color
		    2);					// font thickness

        // Draw white cross hair (long horizontal line and short vertical)
	// Draw vertical line
	cv::Point p3(center,440), p4(center,470);
	int thicknessLine = 2;
	cv::line(img, p3, p4, CV_RGB(255,255,255), thicknessLine);

	// Draw horizontal line
	cv::Point p5(center-150,455), p6(center+150,455);
        cv::line(img, p5, p6, CV_RGB(255,255,255), thicknessLine);

	// Draw current vertical line status (steering angle)
	int offset = center - steering_angle_degrees;
	cv::Point p7(offset,440), p8(offset,470);
        cv::line(img, p7, p8, CV_RGB(255,255,0), thicknessLine+1);
    }

    cv::imshow("Orig", currentFrame_ptr->image);
    //cv::imshow("Out", processed_bgr);
    cv::waitKey(1);
  }
}

//Callback function for a new image on topic "image".
void readImg(const sensor_msgs::ImageConstPtr& img) {

        try
        {
                currentFrame_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
                processImage(cameraInfo, lanesConf);
                resultImg_pub.publish(*currentFrame_ptr->toImageMsg());
        }
        catch (cv_bridge::Exception& e)
        {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
        }

}

void laneDetectionFromFiles(std::string& path) {

std::vector<std::string> fileNames;

if (boost::filesystem::is_directory(path))
 {
   for (boost::filesystem::directory_iterator itr(path); itr!=boost::filesystem::directory_iterator(); ++itr)
   {
     if (boost::filesystem::is_regular_file(itr->status()) && (itr->path().filename().string().find(".jpg") != std::string::npos || itr->path().filename().string().find(".png") != std::string::npos)) {
       fileNames.push_back(itr->path().filename().string());
     }
   }

   std::sort(fileNames.begin(), fileNames.end());

   std::vector<cv_bridge::CvImagePtr> frames;
   sensor_msgs::Image currentFrame;

   for(int i = 0; i < fileNames.size(); i++) {
     cv::Mat img = cv::imread(path + "/" + fileNames.at(i));


     cv_bridge::CvImage img_bridge;
     std_msgs::Header header; // empty header
     header.stamp = ros::Time::now(); // time
     img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, img);

     currentFrame = *img_bridge.toImageMsg(); // from cv_bridge to sensor_msgs::Image
     if(currentFrame.step == 0) {
       std::cout << "Error: No image with name " << fileNames.at(i) << " received" << std::endl;
     }
     try
     {
             currentFrame_ptr = cv_bridge::toCvCopy(currentFrame, sensor_msgs::image_encodings::BGR8);
             frames.push_back(currentFrame_ptr);
     }
     catch (cv_bridge::Exception& e)
     {
             ROS_ERROR("cv_bridge exception: %s", e.what());
             return;
     }
   }

   int i = 0;
   int64 t0 = cv::getTickCount();
   while(i < frames.size()) {
     currentFrame_ptr = frames.at(i);
     processImage(cameraInfo, lanesConf);
     i++;
   }
   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();
   ROS_INFO("%i images processed in %f seconds. Frequency: %fHz", i, secs, (float)i/secs);
   fitting_phase.closeFile();
 }
}


int main(int argc, char **argv){

        ros::init(argc, argv, "lane_detector");

        /**
         * NodeHandle is the main access point to communications with the ROS system.
         * The first NodeHandle constructed will fully initialize this node, and the last
         * NodeHandle destructed will close down the node.
         */
        ros::NodeHandle nh;
        image_transport::ImageTransport it(nh);

        dynamic_reconfigure::Server<lane_detector::DetectorConfig> server;
        dynamic_reconfigure::Server<lane_detector::DetectorConfig>::CallbackType f;

        f = boost::bind(&configCallback, _1, _2);
        server.setCallback(f);

        bool info_set = false;
        bool loadFiles = false;
        ros::param::get("~images_from_folder", loadFiles);
        /**
        * Read camera information
        * IMPORTANT: If images are loaded from a folder the camera parameters have to be set
        * inside the function readCameraInfo (lane_detector.cpp::57)
        */

	// Since we don't have a HW camera, we will comment out the subscriber to the camera and
	// read the hardcoded camera parameters for Prescan
	//
        //ros::Subscriber cameraInfo_sub = nh.subscribe<sensor_msgs::CameraInfo>("camera_info", 1, std::bind(readCameraInfo, std::placeholders::_1, &info_set));
        //if(loadFiles) readCameraInfo(NULL,&info_set);
	readCameraInfo(NULL,&info_set);

        while (!info_set) {
          ros::spinOnce();
          ROS_WARN("No information on topic camera_info received");
        }

        //Stop the Subscriber
        //cameraInfo_sub.shutdown();

        //Set cameraInfo
        preproc.setCameraInfo(cameraInfo);

        /**
         * The advertise() function is how you tell ROS that you want to
         * publish on a given topic name. This invokes a call to the ROS
         * master node, which keeps a registry of who is publishing and who
         * is subscribing. After this advertise() call is made, the master
         * node will notify anyone who is trying to subscribe to this topic name,
         * and they will in turn negotiate a peer-to-peer connection with this
         * node.  advertise() returns a Publisher object which allows you to
         * publish messages on that topic through a call to publish().  Once
         * all copies of the returned Publisher object are destroyed, the topic
         * will be automatically unadvertised.
         *
         * The second parameter to advertise() is the size of the message queue
         * used for publishing messages.  If messages are published more quickly
         * than we can send them, the number here specifies how many messages to
         * buffer up before throwing some away.
         */

        /**
         * Subscriber on topic "lane_detector/driving_orientation".
         * This subscriber allows to change lane while driving and to select the desired
         * driving direction
         */

	// initialize localizer position values
        ros::Subscriber driving_orientation_sub = nh.subscribe<std_msgs::Int32>("lane_detector/driving_orientation", 1, drivingOrientationCB);
	ros::Subscriber pose_sub = nh.subscribe("gnss_pose", 1, poseCB);
	ros::Subscriber clock_sub = nh.subscribe("/clock", 1, clockCB);

        image_transport::Subscriber image_sub = it.subscribe("/image_raw", 1, readImg);

	// Publisher
        resultImg_pub = it.advertise("lane_detector/result", 1);
        lane_pub = nh.advertise<lane_detector::Lane>("lane_detector/lane", 1);

        std::string imagesPath = "";
        ros::param::get("~images_path", imagesPath);
	
        if(loadFiles) laneDetectionFromFiles(imagesPath); // Whether to load the images from a folder (data set) or from the kinect

        //ros::MultiThreadedSpinner spinner(0); // Use one thread for core
        //spinner.spin(); // spin() will not return until the node has been shutdown
        ros::spin();
        return 0;
}
