#include <boost/version.hpp>
#if ((BOOST_VERSION / 100) % 1000) >= 53
#include <boost/thread/lock_guard.hpp>
#endif

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dynamic_reconfigure/server.h>
#include <turtlebot3_autorace_camera/ImageProjectionParamsConfig.h>

namespace image_projection {

class ImageProjectionNodelet : public nodelet::Nodelet
{
    // ROS communication
    boost::shared_ptr<image_transport::ImageTransport> it_;

    boost::mutex connect_mutex_;
    image_transport::Publisher pub_project_;
    image_transport::Publisher pub_calibrate_;
    image_transport::Subscriber sub_image_;
    int queue_size_;

    // Dynamic reconfigure
    boost::recursive_mutex config_mutex_;
    typedef turtlebot3_autorace_camera::ImageProjectionParamsConfig Config;
    typedef dynamic_reconfigure::Server<Config> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server_;
    // Config config_;

    // Parameters
    bool is_calibration_mode;           // unused in nodelet
    int top_x, top_y, bottom_x, bottom_y;

    virtual void onInit();

    void connectCB();
    void disconnectCB();

    void imageCB(const sensor_msgs::ImageConstPtr& image_msg);

    void configCB(Config &config, uint32_t level);
};

void ImageProjectionNodelet::onInit()
{
    // Get NodeHandles
    ros::NodeHandle &nh         = getNodeHandle();
    ros::NodeHandle &private_nh = getPrivateNodeHandle();
    it_.reset(new image_transport::ImageTransport(nh));

    // Read parameters
    private_nh.param("queue_size", queue_size_, 5);
    private_nh.param("is_extrinsic_camera_calibration_mode", is_calibration_mode, false);
    NODELET_INFO("[Image Projection] queue_size = %d", queue_size_);
    NODELET_INFO("[Image Projection] is_calibration_mode = %s", (is_calibration_mode?"True":"False"));

    private_nh.param("camera/extrinsic_camera_calibration/top_x", top_x, 75);
    private_nh.param("camera/extrinsic_camera_calibration/top_y", top_y, 35);
    private_nh.param("camera/extrinsic_camera_calibration/bottom_x", bottom_x, 165);
    private_nh.param("camera/extrinsic_camera_calibration/bottom_y", bottom_y, 120);
    NODELET_INFO("[Image Projection] top_x: %d, top_y: %d, bottom_x: %d, bottom_y: %d", 
                top_x, top_y, bottom_x, bottom_y);

    // Set up dynamic reconfigure
    if (is_calibration_mode) {
        reconfigure_server_.reset(new ReconfigureServer(config_mutex_, private_nh));
        ReconfigureServer::CallbackType f = boost::bind(&ImageProjectionNodelet::configCB, this, _1, _2);
        reconfigure_server_->setCallback(f);
    }

    // Initialize image-transport publisher
    // Monitor whether anyone is subscribed to the output
    image_transport::SubscriberStatusCallback connect_cb = boost::bind(&ImageProjectionNodelet::connectCB, this);
    image_transport::SubscriberStatusCallback disconnect_cb = boost::bind(&ImageProjectionNodelet::disconnectCB, this);
    // Make sure we don't enter connectCb() between advertising and assigning to pub_rect_
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    pub_project_ = it_->advertise("image_output", 1, connect_cb, disconnect_cb);
    pub_calibrate_ = it_->advertise("image_calib", 1, connect_cb, disconnect_cb);
}

void ImageProjectionNodelet::connectCB()
{
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if (!sub_image_) {
        NODELET_INFO("[Image Projection] Activate image subscriber...");
        image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
        sub_image_ = it_->subscribe("image_input", queue_size_, &ImageProjectionNodelet::imageCB, this, hints);
    }
}

void ImageProjectionNodelet::disconnectCB()
{
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if (pub_project_.getNumSubscribers() == 0) {
        if ((is_calibration_mode == false) ||
            (is_calibration_mode == true && pub_calibrate_.getNumSubscribers() == 0)) 
        {
            NODELET_INFO("[Image Projection] Shutdown image subscriber...");
            sub_image_.shutdown();
        }
    }
}

void ImageProjectionNodelet::imageCB(const sensor_msgs::ImageConstPtr& image_msg)
{
    int hist_size = 255;
    double min_gray, max_gray;
    double alpha = 0, beta = 0;

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
    }
    catch (cv_bridge::Exception& e) {
        NODELET_ERROR("cv_bridge exception: %s", e.what());
        return;
    }    
    const cv::Mat cv_image_origin = cv_ptr->image;

    // {
    //     boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
    //     top_x = config_.top_x;
    //     top_y = config_.top_y;
    //     bottom_x = config_.bottom_x;
    //     bottom_y = config_.bottom_y;
    // }

    if (is_calibration_mode) {
        cv::Mat cv_image_calib;
        cv_image_origin.copyTo(cv_image_calib);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        int thichness = 1;

        cv::line(cv_image_calib, cv::Point(160 - top_x, 180 - top_y), cv::Point(160 + top_x, 180 - top_y), color, thichness);
        cv::line(cv_image_calib, cv::Point(160 - bottom_x, 120 + bottom_y), cv::Point(160 + bottom_x, 120 + bottom_y), color, thichness);
        cv::line(cv_image_calib, cv::Point(160 + bottom_x, 120 + bottom_y), cv::Point(160 + top_x, 180 - top_y), color, thichness);
        cv::line(cv_image_calib, cv::Point(160 - bottom_x, 120 + bottom_y), cv::Point(160 - top_x, 180 - top_y), color, thichness);
    
        sensor_msgs::ImagePtr calib_msg = cv_bridge::CvImage(image_msg->header, "bgr8", cv_image_calib).toImageMsg();    
        pub_calibrate_.publish(calib_msg);
    }

    // cv_image_origin.copyTo(cv_image_projected);

    // adding Gaussian blur to the image of original
    // cv::GaussianBlur(cv_image_projected, cv_image_projected, cv::Size(5, 5), 0);    
    cv::GaussianBlur(cv_image_origin, cv_image_origin, cv::Size(5, 5), 0);    
 
    // homography transform process
    // selecting 4 points from the original image
    std::vector<cv::Point2f> pts_src;
    pts_src.push_back(cv::Point2f(160 - top_x, 180 - top_y));
    pts_src.push_back(cv::Point2f(160 + top_x, 180 - top_y));
    pts_src.push_back(cv::Point2f(160 + bottom_x, 120 + bottom_y));
    pts_src.push_back(cv::Point2f(160 - bottom_x, 120 + bottom_y));

    // selecting 4 points from image that will be transformed
    std::vector<cv::Point> pts_dst;
    pts_dst.push_back(cv::Point2f(200, 0));
    pts_dst.push_back(cv::Point2f(800, 0));
    pts_dst.push_back(cv::Point2f(800, 600));
    pts_dst.push_back(cv::Point2f(200, 600));

    // finding homography matrix
    cv::Mat h = cv::findHomography(pts_src, pts_dst);
    
    // homography process
    cv::Mat cv_image_projected;
    cv::warpPerspective(cv_image_origin, cv_image_projected, h, cv::Size(1000, 600));
   
    // fill the empty space with black triangles on left and right side of bottom
    cv::Point triangles[][3] = {
        { cv::Point(0, 599), cv::Point(0, 340), cv::Point(200, 599)},
        { cv::Point(999, 599), cv::Point(999, 340), cv::Point(799, 599)},
    };
    const cv::Point* ppt[2] = { triangles[0], triangles[1] };
    int npts[2] = { 3, 3 };
    cv::Scalar black = cv::Scalar(0, 0, 0);
    cv::fillPoly(cv_image_projected, ppt, npts, 2, black);

    sensor_msgs::ImagePtr projected_msg = cv_bridge::CvImage(image_msg->header, "bgr8", cv_image_projected).toImageMsg();
    pub_project_.publish(projected_msg);
}

void ImageProjectionNodelet::configCB(Config &config, uint32_t level)
{
    NODELET_INFO("[Image Projection] Extrinsic Camera Calibration parameter reconfigured to");
    NODELET_INFO("top_x: %d, top_y: %d, bottom_x: %d, bottom_y: %d", 
                config.top_x, config.top_y, config.bottom_x, config.bottom_y);

    boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
    // config_ = config;
    top_x = config.top_x;
    top_y = config.top_y;
    bottom_x = config.bottom_x;
    bottom_y = config.bottom_y;
}

}       // image_project namespace

// Register nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(image_projection::ImageProjectionNodelet, nodelet::Nodelet)