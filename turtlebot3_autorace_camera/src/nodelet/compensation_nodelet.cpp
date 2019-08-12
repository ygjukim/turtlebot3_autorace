#include <boost/version.hpp>
#if ((BOOST_VERSION / 100) % 1000) >= 53
#include <boost/thread/lock_guard.hpp>
#endif

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <dynamic_reconfigure/server.h>
#include <turtlebot3_autorace_camera/ImageCompensationParamsConfig.h>

namespace image_compensation {

class ImageCompensationNodelet : public nodelet::Nodelet
{
    // ROS communication
    boost::shared_ptr<image_transport::ImageTransport> it_;

    boost::mutex connect_mutex_;
    image_transport::Publisher pub_compensate_;
    image_transport::Subscriber sub_image_;
    int queue_size_;

    // Dynamic reconfigure
    boost::recursive_mutex config_mutex_;
    typedef turtlebot3_autorace_camera::ImageCompensationParamsConfig Config;
    typedef dynamic_reconfigure::Server<Config> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server_;
    // Config config_;

    // Parameters
    bool is_calibration_mode;       // unused in nodelet
    double clip_hist_percent;

    virtual void onInit();

    void connectCb();

    void imageCb(const sensor_msgs::ImageConstPtr& image_msg);

    void configCb(Config &config, uint32_t level);
};

void ImageCompensationNodelet::onInit()
{
    // Get NodeHandles
    ros::NodeHandle &nh         = getNodeHandle();
    ros::NodeHandle &private_nh = getPrivateNodeHandle();
    it_.reset(new image_transport::ImageTransport(nh));

    // Read parameters
    private_nh.param("queue_size", queue_size_, 5);
    private_nh.param("is_extrinsic_camera_calibration_mode", is_calibration_mode, false);
    NODELET_INFO_STREAM("[Image Compensation] queue_size = " << queue_size_);
    NODELET_INFO("[Image Compensation] is_calibration_mode = %s", (is_calibration_mode?"True":"False"));

    private_nh.param("camera/extrinsic_camera_calibration/clip_hist_percent", clip_hist_percent, 1.0);
    NODELET_INFO("[Image Compensation] clip_hist_percent = %f", clip_hist_percent);
    
    // Set up dynamic reconfigure
    if (is_calibration_mode) {
        reconfigure_server_.reset(new ReconfigureServer(config_mutex_, private_nh));
        ReconfigureServer::CallbackType f = boost::bind(&ImageCompensationNodelet::configCb, this, _1, _2);
        reconfigure_server_->setCallback(f);
    }

    // Monitor whether anyone is subscribed to the output
    image_transport::SubscriberStatusCallback connect_cb = boost::bind(&ImageCompensationNodelet::connectCb, this);
    // Make sure we don't enter connectCb() between advertising and assigning to pub_rect_
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    pub_compensate_  = it_->advertise("image_output",  1, connect_cb, connect_cb);
}

void ImageCompensationNodelet::connectCb()
{
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if (pub_compensate_.getNumSubscribers() == 0)
    {
        NODELET_INFO("[Image Compensation] Shutdown image subscriber...");
        sub_image_.shutdown();
    }
    else if (!sub_image_)
    {
        NODELET_INFO("[Image Compensation] Activate image subscriber...");
        image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
        sub_image_ = it_->subscribe("image_input", queue_size_, &ImageCompensationNodelet::imageCb, this, hints);
    }
}

void ImageCompensationNodelet::imageCb(const sensor_msgs::ImageConstPtr& image_msg)
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

    cv::Mat cv_image_compenated;
    cv_image_origin.copyTo(cv_image_compenated);

    cv::Mat gray;
    cv::cvtColor(cv_image_compenated, gray, CV_BGR2GRAY);

    // {
    //     boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
    //     clip_hist_percent = config_.clip_hist_percent;
    // }

    if (clip_hist_percent == 0.0) {
        cv::minMaxLoc(gray, &min_gray, &max_gray);
    }
    else {
        // const int* channels = { 0 };
        // float channel_range[] = { 0.0, 255.0 };
        // const float* channel_ranges = channel_range;
        // cv::MatND hist;
        // cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, &hist_size, &channel_ranges);

        int hist[256] = { 0, };
        int value;    
        for (int y = 0; y < cv_image_compenated.rows; y++) {
            for (int x = 0; x < cv_image_compenated.cols; x++) {
                value = gray.at<uchar>(y, x);
                hist[value] += 1;
            }
        }

        int accumulator[256] = { 0, };
        int sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += hist[i];
            accumulator[i] = sum;
        }

        int max = accumulator[255];
        double clip_percent = (double)max * clip_hist_percent / 100;
        clip_percent /= 2.0;
        int index = 0;

        for (index=0; accumulator[index] < clip_percent; index++) {}
        min_gray = (double)index;

        clip_percent = (double)max - clip_percent;
        for (index=255; accumulator[index] > clip_percent; index--) {}
        max_gray = (double)index;
    }

    // NODELET_INFO("min_gray = %f, max_gray = %f", min_gray, max_gray);

    alpha = (double)(hist_size - 1) / (max_gray - min_gray);
    beta = -min_gray * alpha;
    cv::convertScaleAbs(cv_image_compenated, cv_image_compenated, alpha, beta);

//    sensor_msgs::ImagePtr compensated_msg = cv_bridge::CvImage(image_msg->header, image_msg->encoding, cv_image_compenated).toImageMsg();
    sensor_msgs::ImagePtr compensated_msg = cv_bridge::CvImage(image_msg->header, "bgr8", cv_image_compenated).toImageMsg();
    pub_compensate_.publish(compensated_msg);
}

void ImageCompensationNodelet::configCb(Config &config, uint32_t level)
{
    NODELET_INFO("[Image Compensation] Extrinsic Camera Calibration parameter reconfigured to");
    NODELET_INFO("clip_hist_percent : %f", config.clip_hist_percent);

    boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
    // config_ = config;
    clip_hist_percent = config.clip_hist_percent;
}

}       // image_compensation namespace

// Register nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS( image_compensation::ImageCompensationNodelet, nodelet::Nodelet)