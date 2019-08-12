#include <boost/version.hpp>
#if ((BOOST_VERSION / 100) % 1000) >= 53
#include <boost/thread/lock_guard.hpp>
#endif

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt8.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <dynamic_reconfigure/server.h>
#include <turtlebot3_autorace_detect/DetectLaneParamsConfig.h>

// #include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <gsl/gsl_multifit.h>
#include <stdbool.h>
#include <math.h>

#include "polyfit_queue.hpp"

using namespace std;
using namespace cv;

namespace lane_detection {

class DetectLaneNodelet : public nodelet::Nodelet
{
    // ROS communication
    boost::shared_ptr<image_transport::ImageTransport> it_;

    image_transport::Subscriber sub_projected_image_;

    boost::mutex connect_mutex_;
    image_transport::Publisher pub_image_lane_;
    image_transport::Publisher pub_image_yellow_lane_;
    image_transport::Publisher pub_image_white_lane_;

    ros::Publisher pub_lane_;
    ros::Publisher pub_yellow_lane_reliability_;
    ros::Publisher pub_white_lane_reliability_;

    // Dynamic reconfigure
    boost::recursive_mutex config_mutex_;
    typedef turtlebot3_autorace_detect::DetectLaneParamsConfig Config;
    typedef dynamic_reconfigure::Server<Config> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server_;
    // Config config_;

    // Parameters
    bool is_calibration_mode;
    int queue_size_;

    //// reconfigurable parameters
    int hue_white_l, hue_white_h;
    int saturation_white_l, saturation_white_h;
    int lightness_white_l, lightness_white_h;
    int hue_yellow_l, hue_yellow_h;
    int saturation_yellow_l, saturation_yellow_h;
    int lightness_yellow_l, lightness_yellow_h;

    // Properties
    uint8_t reliability_yellow_lane;
    uint8_t reliability_white_lane;

    polyfit_t white_lane_fit;
    polyfit_t yellow_lane_fit;
    polyfit_t lane_fit_bef;

    PolyfitQueue mov_avg_left;
    PolyfitQueue mov_avg_right;

    int counter;        // number of skipped input images

    // Methods
    virtual void onInit();

    void connectCB();
    void disconnectCB();
    void imageCB(const sensor_msgs::ImageConstPtr& image_msg);
    void configCB(Config &config, uint32_t level);

    vector<double> polynomialfit(int obs, int degree, int *dx, int *dy);
    // vector<double> polynomialfit(int obs, int degree, double *dx, double *dy);
    // vector<double> polyRegression(const vector<int>& x, const vector<int>& y);
    int count_nonzero(const Mat& image);
    int argmax(vector<int>& values, int first, int last);
    int arghalfmax(vector<int>& values, int first, int last, bool reverse=false);

    int maskWhiteLaneFromHSV(const Mat& hsv_img, Mat& mask, vector<Point>& locations);
    int maskYellowLaneFromHSV(const Mat& hsv_img, Mat& mask, vector<Point>& locations);
    int fit_from_lines(const vector<Point>& nonzero_pts, vector<double>& lane_fit);
    int sliding_window(const Mat& image, const vector<Point>& nonzero_pts, int left_or_right, vector<double>& lane_fit);
    int getPointsFromPolyfit(const vector<double>& poly_fit, int size, vector<Point>& points);
    int getCenterPointsFromPolyfit(const vector<double>& left_poly_fit, 
                const vector<double>& right_poly_fit,
                int offset, int size, vector<Point>& points);
    void make_lane(Mat& image, int white_fraction, int yellow_fraction);
};

// vector<double> DetectLaneNodelet::polynomialfit(int obs, int degree, double *dx, double *dy)
vector<double> DetectLaneNodelet::polynomialfit(int obs, int degree, int *dx, int *dy)
{
    //
    // ref. site : https://rosettacode.org/wiki/Polynomial_regression
    //
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;

    int i, j;

    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);

    for(i=0; i < obs; i++) {
        for(j=0; j < degree; j++) {
            gsl_matrix_set(X, i, j, pow(dx[i], j));
        }
        gsl_vector_set(y, i, dy[i]);
    }

    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    /* store result ... */
    vector<double> coeffs;
    for(i=degree-1; i >= 0; i--)
    {
        coeffs.push_back(gsl_vector_get(c, i));
    }

    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);
    return coeffs;
}

/* 
// possible since C++17
vector<double> DetectLaneNodelet::polyRegression(const vector<int>& x, const vector<int>& y) 
{
    int n = x.size();
    vector<int> r(n);
    iota(r.begin(), r.end(), 0);
    double xm = accumulate(x.begin(), x.end(), 0.0) / x.size();
    double ym = accumulate(y.begin(), y.end(), 0.0) / y.size();
    double x2m = transform_reduce(r.begin(), r.end(), 0.0, plus<double>{}, [](double a) {return a * a; }) / r.size();
    double x3m = transform_reduce(r.begin(), r.end(), 0.0, plus<double>{}, [](double a) {return a * a * a; }) / r.size();
    double x4m = transform_reduce(r.begin(), r.end(), 0.0, plus<double>{}, [](double a) {return a * a * a * a; }) / r.size();
 
    double xym = transform_reduce(x.begin(), x.end(), y.begin(), 0.0, plus<double>{}, multiplies<double>{});
    xym /= fmin(x.size(), y.size());
 
    double x2ym = transform_reduce(x.begin(), x.end(), y.begin(), 0.0, plus<double>{}, [](double a, double b) { return a * a * b; });
    x2ym /= fmin(x.size(), y.size());
 
    double sxx = x2m - xm * xm;
    double sxy = xym - xm * ym;
    double sxx2 = x3m - xm * x2m;
    double sx2x2 = x4m - x2m * x2m;
    double sx2y = x2ym - x2m * ym;
 
    vector<double> coeffs;
    double c = (sx2y * sxx - sxy * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    coeffs.push_back(c);
    double b = (sxy * sx2x2 - sx2y * sxx2) / (sxx * sx2x2 - sxx2 * sxx2);
    coeffs.push_back(b);
    coeffs.push_back(ym - b * xm - c * x2m);

    return coeffs;
}
*/

int DetectLaneNodelet::count_nonzero(const Mat& image) 
{
    vector<Point> locations;            // locations of non-zero pixels
    findNonZero(image, locations);      // find locatopns of non-zero elements
    return locations.size();            // return number of non-zero-element
}

int DetectLaneNodelet::argmax(vector<int>& values, int first, int last)
{
    int *data = values.data();
    int pos = first;
    if (first < last) {
        data += first;
        int max_value = *data++;
        for (int i=first+1; i<=last; i++) {
            if (max_value < *data) {
                max_value = *data;
                pos = i;
            }
            data++;
        }
    }
    return pos;
}

int DetectLaneNodelet::arghalfmax(vector<int>& values, int first, int last, bool reverse/*=false*/)
{
    int *data = values.data();
    int pos = first;
    if (first < last) {
        // get half of max value
        data += first;
        int max_value = *data++;
        for (int i=first+1; i<=last; i++, data++) {
            if (max_value < *data) {
                max_value = *data;
            }
        }        
        
        // search position of half of max value
        // max_value /= 2;
        max_value /= 4;
        if (!reverse) {
            data = values.data() + first;
            for (pos=first; pos<=last; pos++, data++) {
                if (*data >= max_value)  break;
            }        
        }
        else {
            data = values.data() + last;
            for (pos=last; pos>=first; pos--, data--) {
                if (*data >= max_value)  break;
            }        
        }
    }
    return pos;
}

int DetectLaneNodelet::maskWhiteLaneFromHSV(const Mat& hsv_img, Mat& mask, vector<Point>& locations)
{
    // Define range of white color in HSV
    Scalar lower_white(hue_white_l, saturation_white_l, lightness_white_l);
    Scalar upper_white(hue_white_h, saturation_white_h, lightness_white_h);

    // Threshold the HSV image to get only white-color pixels
    inRange(hsv_img, lower_white, upper_white, mask);
    findNonZero(mask, locations);

    // Adjust lightness threshold value in terms of intensity of white-colors
    int fraction_num = locations.size();
    if (fraction_num > 0 && !is_calibration_mode) {
        if (fraction_num > 35000) {
            if (lightness_white_l < 250)  lightness_white_l += 5;
        }
        else if (fraction_num < 5000) {
            if (lightness_white_l > 50)  lightness_white_l -= 5;
        }
    }

    // Count much-short rows and adjust white-lane-reliability value
    int much_short_rows = 0;
    if (fraction_num > 0) {
        for (int row=0; row<mask.rows; row++) {
            uchar* ptr = mask.ptr<uchar>(row);
            int col = 0;
            for (col=0; col<mask.cols; col++, ptr++) {
                if (*ptr != 0)  break;
            }
            if (col >= mask.cols)  much_short_rows++;
        }
    }

    if (much_short_rows > 100) {
        if (reliability_white_lane >= 5)  reliability_white_lane -= 5;
    }
    else if (much_short_rows <= 100 ){
        if (reliability_white_lane <= 99)  reliability_white_lane += 5;
    }

    // Publish white-lane-reliability value and masked image
    std_msgs::UInt8 msg_white_lane_reliability;
    msg_white_lane_reliability.data = reliability_white_lane;
    pub_white_lane_reliability_.publish(msg_white_lane_reliability); 

    if (is_calibration_mode) {
        sensor_msgs::ImagePtr white_lane_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
        pub_image_white_lane_.publish(white_lane_msg);
    }

    return fraction_num;
}

int DetectLaneNodelet::maskYellowLaneFromHSV(const Mat& hsv_img, Mat& mask, vector<Point>& locations)
{
    // Define range of yellow color in HSV
    Scalar lower_yellow(hue_yellow_l, saturation_yellow_l, lightness_yellow_l);
    Scalar upper_yellow(hue_yellow_h, saturation_yellow_h, lightness_yellow_h);

    // Threshold the HSV image to get only yellow-color pixels
    inRange(hsv_img, lower_yellow, upper_yellow, mask);
    findNonZero(mask, locations);

    // Adjust lightness threshold value in terms of intensity of yellow-colors
    int fraction_num = locations.size();
    if (fraction_num > 0 && !is_calibration_mode) {
        if (fraction_num > 35000) {
            if (lightness_yellow_l < 250)  lightness_yellow_l += 20;
        }
        else if (fraction_num < 5000) {
            if (lightness_yellow_l > 90)  lightness_yellow_l -= 20;
        }
    }

    // Count mush-short rows and adjust yellow-lane-reliability value
    int much_short_rows = 0;
    if (fraction_num > 0) {
        for (int row=0; row<mask.rows; row++) {
            uchar* ptr = mask.ptr<uchar>(row);
            int col = 0;
            for (col=0; col<mask.cols; col++, ptr++) {
                if (*ptr != 0)  break;
            }
            if (col >= mask.cols)  much_short_rows++;
        }
    }

    if (much_short_rows > 100) {
        if (reliability_yellow_lane >= 5)  reliability_yellow_lane -= 5;
    }
    else if (much_short_rows <= 100 ){
        if (reliability_yellow_lane <= 99)  reliability_yellow_lane += 5;
    }

    // Publish yellow-lane-reliability value and masked image
    std_msgs::UInt8 msg_yellow_lane_reliability;
    msg_yellow_lane_reliability.data = reliability_yellow_lane;
    pub_yellow_lane_reliability_.publish(msg_yellow_lane_reliability); 

    if (is_calibration_mode) {
        sensor_msgs::ImagePtr yellow_lane_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
        pub_image_yellow_lane_.publish(yellow_lane_msg);
    }

    return fraction_num;
}

#define LANE_FIT_DEGREE     3       // 2-order polynomial fitting

int DetectLaneNodelet::fit_from_lines(const vector<Point>& nonzero_pts, vector<double>& lane_fit)
{
    int pt_size = nonzero_pts.size();
    vector<int> x_vals;
    vector<int> y_vals;
    int margin = 100;
    for (int i=0; i<pt_size; i++) {
        int x = nonzero_pts[i].x;
        int y = nonzero_pts[i].y;
        int  est_x = (int)(lane_fit[0]*y*y + lane_fit[1]*y + lane_fit[2]);
        if (abs(x - est_x) < margin) {
            x_vals.push_back(x);
            y_vals.push_back(y);
        }        
    }

    // Fit a second-order polynormial to each
    int pts_size = y_vals.size();
    if (pts_size > 0) {
        // lane_fit = polyRegression(y_vals, x_vals);
        lane_fit = polynomialfit(pts_size, LANE_FIT_DEGREE, y_vals.data(), x_vals.data());
    }

    return pts_size;
}

#define LEFT_SIDE       0
#define RIGHT_SIDE      1

int DetectLaneNodelet::sliding_window(const Mat& image, const vector<Point>& nonzero_pts, int left_or_right, vector<double>& lane_fit)
{
    int width = image.cols;         // image width
    int height = image.rows;        // image height

    Mat out_image;
    if (is_calibration_mode) {
        cvtColor(image, out_image, CV_GRAY2BGR);
    }

    vector<int> histogram(width);
    for (int i=height/2; i<height; i++) {
        const uchar* ptr = image.ptr<uchar>(i);
        for (int j=0; j<width; j++) {
            if (*ptr++ != 0)  histogram[j]++;
        }
    }

    int lane_base = 0;
    if (left_or_right == LEFT_SIDE) {
        // lane_base = argmax(histogram, 0, width/2-1);
        lane_base = arghalfmax(histogram, 0, width/2-1);
    }
    else if (left_or_right == RIGHT_SIDE) {
        // lane_base = argmax(histogram, width/2, width-1);
        lane_base = arghalfmax(histogram, width/2, width-1, true);
    }

    int nwindows = 20;
    int window_height = height / nwindows;
//  int window_width = 100;
    int window_width = 150;     // 100 => 150 in terms of lane width
    int x_current = lane_base;
//    int minpixels = 50;
    int minpixels = 25;

    vector<Point> lane_inds;
    int acc_x_pos = 0;
    int x_count = 0;
    int win_top, win_left;

    // Step through the windows one by one
    for (int i=0; i<nwindows; i++) {
        // Identify window boundaries in x and y
        win_top = height - ((i+1) * window_height);
        win_left = x_current - (window_width/2);
        if (win_left < 0)  win_left = 0;
        acc_x_pos = 0;
        x_count = 0;

        // Identify the nonzero pixels in x and y within the window
        for (int k=0; k<nonzero_pts.size(); k++) {
            int x = nonzero_pts[k].x;
            int y = nonzero_pts[k].y;
            if ((y >= win_top && (y - win_top) < window_height) && 
                (x >= win_left && (x - win_left) < window_width)) {
                lane_inds.push_back(nonzero_pts[k]);
                acc_x_pos += x;
                x_count++;
            }
        }

        // If you found > minpix pixels, recenter next window on their mean position
        if (x_count > minpixels) {
            x_current = acc_x_pos / x_count;
        }

        if (is_calibration_mode) {
            //# Draw the windows on the visualization image
            rectangle(out_image, Point(win_left, win_top), 
                Point(win_left+window_width-1, win_top+window_height-1), 
                Scalar(0, 255, 0), 2);
        }
    }

    int pts_size = lane_inds.size();
    if (pts_size > 0) {
        // Extract line pixel positions
        vector<int> x_vals(pts_size);
        vector<int> y_vals(pts_size);
        int *xp = x_vals.data();
        int *yp = y_vals.data();
        for (int i=0; i<pts_size; i++) {
            *xp++ = lane_inds[i].x;
            *yp++ = lane_inds[i].y;
        }

        // Fit a second-order polynormial to each
        try {
            // lane_fit = polyRegression(y_vals, x_vals));
            lane_fit = polynomialfit(pts_size, LANE_FIT_DEGREE, y_vals.data(), x_vals.data());
            lane_fit_bef = lane_fit;
        } catch (int ex) {
            lane_fit = lane_fit_bef;
        }
    }
    else {
        lane_fit = lane_fit_bef;
    }

    if (is_calibration_mode) {
        // FOR TESTING...
        // vector<Point> pts;
        // getPointsFromPolyfit(lane_fit, out_image.rows, pts);
        // const Point* ppt[] = { pts.data() };
        // int npts[] = { out_image.rows };
        // polylines(out_image, ppt, npts, 1, false, Scalar(255, 0, 0), 12);

        if (left_or_right == LEFT_SIDE) {
            sensor_msgs::ImagePtr yellow_lane_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
            pub_image_yellow_lane_.publish(yellow_lane_msg);
        }
        else {
            sensor_msgs::ImagePtr white_lane_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
            pub_image_white_lane_.publish(white_lane_msg);
        }
    }

    return pts_size;
}

int DetectLaneNodelet::getPointsFromPolyfit(const vector<double>& poly_fit, int size, vector<Point>& points)
{
    for (int y=0; y<size; y++) {
        points.push_back(Point((int)(poly_fit[0]*y*y+poly_fit[1]*y+poly_fit[2]), y));
    }
    return size;
}

int DetectLaneNodelet::getCenterPointsFromPolyfit(const vector<double>& left_poly_fit, const vector<double>& right_poly_fit,
                int offset, int size, vector<Point>& points)
{
    int center_x;

    if (offset == 0) {
        for (int y=0; y<size; y++) {
            center_x = (int)((left_poly_fit[0]*y*y+left_poly_fit[1]*y+left_poly_fit[2] 
                            + right_poly_fit[0]*y*y+right_poly_fit[1]*y+right_poly_fit[2])/2);
            points.push_back(Point(center_x, y));
        }
    }
    else if (offset > 0) {
        for (int y=0; y<size; y++) {
            center_x = (int)(left_poly_fit[0]*y*y+left_poly_fit[1]*y+left_poly_fit[2]) + offset;
            points.push_back(Point(center_x, y));
        }
    }
    else {
        for (int y=0; y<size; y++) {
            center_x = (int)(right_poly_fit[0]*y*y+right_poly_fit[1]*y+right_poly_fit[2]) + offset;
            points.push_back(Point(center_x, y));
        }
    }

    return size;
}

#define SEPARATION_DISTANCE         180     // 225     // 320 => 400

void DetectLaneNodelet::make_lane(Mat& image, int white_fraction, int yellow_fraction)
{
    // Create an image to draw the lines on
    Mat color_warp = Mat::zeros(image.rows, image.cols, image.type());
    Mat color_warp_lines = Mat::zeros(image.rows, image.cols, image.type());

    int npts[] = { image.rows };
    vector<Point> white_pts;
    vector<Point> yellow_pts;
    vector<Point> center_pts;

    if (white_fraction > 3000) {
        getPointsFromPolyfit(white_lane_fit, image.rows, white_pts);
        const Point* ppt[] = { white_pts.data() };
        polylines(color_warp_lines, ppt, npts, 1, false, Scalar(255, 255, 0), 25);
    }

    if (yellow_fraction > 3000) {
        getPointsFromPolyfit(yellow_lane_fit, image.rows, yellow_pts);
        const Point* ppt[] = { yellow_pts.data() };
        polylines(color_warp_lines, ppt, npts, 1, false, Scalar(0, 0, 255), 25);
    }
    
    bool is_center_x_exist = true;

    if (reliability_white_lane > 50 && reliability_yellow_lane > 50) {   
        if (white_fraction > 3000 && yellow_fraction > 3000) {
            getCenterPointsFromPolyfit(yellow_lane_fit, white_lane_fit, 0, image.rows, center_pts);

            // Draw the lane onto the warped blank image
            // Concatenate white-point vector to yellow-point vector
            for (int i=white_pts.size()-1; i>=0; i--) {
                yellow_pts.push_back(white_pts[i]);
            }
            const Point* ppt2[] = { yellow_pts.data() };
            int npts2[] = { (int)yellow_pts.size() };
            fillPoly(color_warp, ppt2, npts2, 1, Scalar(0, 255, 0), 8);
        }
        else if (white_fraction > 3000 && yellow_fraction <= 3000) {
            getCenterPointsFromPolyfit(yellow_lane_fit, white_lane_fit, -SEPARATION_DISTANCE, image.rows, center_pts);
        }
        else if (white_fraction <= 3000 && yellow_fraction > 3000) {
            getCenterPointsFromPolyfit(yellow_lane_fit, white_lane_fit, SEPARATION_DISTANCE, image.rows, center_pts);
        }
    }
    else if (reliability_white_lane <= 50 && reliability_yellow_lane > 50) {
            getCenterPointsFromPolyfit(yellow_lane_fit, white_lane_fit, SEPARATION_DISTANCE, image.rows, center_pts);
    }
    else if (reliability_white_lane > 50 && reliability_yellow_lane <= 50) {
            getCenterPointsFromPolyfit(yellow_lane_fit, white_lane_fit, -SEPARATION_DISTANCE, image.rows, center_pts);
    }
    else {
        is_center_x_exist = false;
    }

    // Publishes lane center
    if (is_center_x_exist) {
        const Point* ppt[] = { center_pts.data() };
        polylines(color_warp_lines, ppt, npts, 1, false, Scalar(0, 255, 255), 12);

        std_msgs::Float64Ptr msg_desired_center = boost::shared_ptr<std_msgs::Float64>(new std_msgs::Float64());
        msg_desired_center->data = center_pts[450].x;
        // msg_desired_center->data = center_pts[350].x;
        pub_lane_.publish(msg_desired_center);
    }

    // Combine the result with the original image
    addWeighted(image, 1, color_warp, 0.2, 0, image);
    addWeighted(image, 1, color_warp_lines, 1, 0, image);

    // Publishes lane image
    sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    pub_image_lane_.publish(image_msg);
}

void DetectLaneNodelet::onInit()
{
    // Get NodeHandles
    ros::NodeHandle &nh         = getNodeHandle();
    ros::NodeHandle &private_nh = getPrivateNodeHandle();
    it_.reset(new image_transport::ImageTransport(nh));

    // Read parameters
    private_nh.param("queue_size", queue_size_, 5);
    private_nh.param("is_detection_calibration_mode", is_calibration_mode, false);

    NODELET_INFO("[Detct Lane] queue_size = %d", queue_size_);
    NODELET_INFO("[Detct Lane] is_calibration_mode = %s", (is_calibration_mode?"True":"False"));

    // Initialize properties
    reliability_white_lane = 100;
    reliability_yellow_lane = 100;

    counter = 0;

    // Set up dynamic reconfigure
    if (is_calibration_mode) {
        reconfigure_server_.reset(new ReconfigureServer(config_mutex_, private_nh));
        ReconfigureServer::CallbackType f = boost::bind(&DetectLaneNodelet::configCB, this, _1, _2);
        reconfigure_server_->setCallback(f);
    }
//    else {
        private_nh.param("detect/lane/white/hue_l", hue_white_l, 0);
        private_nh.param("detect/lane/white/hue_h", hue_white_h, 25);
        private_nh.param("detect/lane/white/saturation_l", saturation_white_l, 0);
        private_nh.param("detect/lane/white/saturation_h", saturation_white_h, 36);
        private_nh.param("detect/lane/white/lightness_l", lightness_white_l, 180);
        private_nh.param("detect/lane/white/lightness_h", lightness_white_h, 255);
        private_nh.param("detect/lane/yellow/hue_l", hue_yellow_l, 27);
        private_nh.param("detect/lane/yellow/hue_h", hue_yellow_h, 41);
        private_nh.param("detect/lane/yellow/saturation_l", saturation_yellow_l, 130);
        private_nh.param("detect/lane/yellow/saturation_h", saturation_yellow_h, 255);
        private_nh.param("detect/lane/yellow/lightness_l", lightness_yellow_l, 160);
        private_nh.param("detect/lane/yellow/lightness_h", lightness_yellow_h, 255);

        NODELET_INFO("[Detect Lane] Extrinsic Camera Calibration parameter");
        NODELET_INFO("hue_white_l: %d, hue_white_h: %d", hue_white_l, hue_white_h);
        NODELET_INFO("saturation_white_l: %d, saturation_white_h: %d", saturation_white_l, saturation_white_h);
        NODELET_INFO("lightness_white_l: %d, lightness_white_h: %d", lightness_white_l, lightness_white_h);
        NODELET_INFO("hue_yellow_l: %d, hue_yellow_h: %d", hue_yellow_l, hue_yellow_h);
        NODELET_INFO("saturation_yellow_l: %d, saturation_yellow_h: %d", saturation_yellow_l, saturation_yellow_h);
        NODELET_INFO("lightness_yellow_l: %d, lightness_yellow_h: %d", lightness_yellow_l, lightness_yellow_h);
//    }

    // Initialize image-transport publisher
    // Monitor whether anyone is subscribed to lane-detection output
    ros::SubscriberStatusCallback connect_cb = boost::bind(&DetectLaneNodelet::connectCB, this);
    ros::SubscriberStatusCallback disconnect_cb = boost::bind(&DetectLaneNodelet::disconnectCB, this);
    // Make sure we don't enter connectCb() between advertising and assigning to pub_lane_
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    
    pub_lane_ = nh.advertise<std_msgs::Float64>("/detect/lane", 1, connect_cb, disconnect_cb);
    pub_yellow_lane_reliability_ = nh.advertise<std_msgs::UInt8>("/detect/yellow_lane_relibility", 1);
    pub_white_lane_reliability_ = nh.advertise<std_msgs::UInt8>("/detect/white_lane_relibility", 1);
    
    pub_image_lane_ = it_->advertise("/detect/image_output", 1);
    if (is_calibration_mode) {
        pub_image_white_lane_ = it_->advertise("/detect/image_output_sub1", 1);
        pub_image_yellow_lane_ = it_->advertise("/detect/image_output_sub2", 1);
    }
}

void DetectLaneNodelet::configCB(Config &config, uint32_t level)
{
    NODELET_INFO("[Detect Lane] Extrinsic Camera Calibration parameter reconfigured to");
    NODELET_INFO("hue_white_l: %d, hue_white_h: %d", config.hue_white_l, config.hue_white_h);
    NODELET_INFO("saturation_white_l: %d, saturation_white_h: %d", config.saturation_white_l, config.saturation_white_h);
    NODELET_INFO("lightness_white_l: %d, lightness_white_h: %d", config.lightness_white_l, config.lightness_white_h);
    NODELET_INFO("hue_yellow_l: %d, hue_yellow_h: %d", config.hue_yellow_l, config.hue_yellow_h);
    NODELET_INFO("saturation_yellow_l: %d, saturation_yellow_h: %d", config.saturation_yellow_l, config.saturation_yellow_h);
    NODELET_INFO("lightness_yellow_l: %d, lightness_yellow_h: %d", config.lightness_yellow_l, config.lightness_yellow_h);

    boost::lock_guard<boost::recursive_mutex> lock(config_mutex_);
    // config_ = config;
    hue_white_l = config.hue_white_l;
    hue_white_h = config.hue_white_h;
    saturation_white_l = config.saturation_white_l;
    saturation_white_h = config.saturation_white_h;
    lightness_white_l = config.lightness_white_l;
    lightness_white_h = config.lightness_white_h;
    hue_yellow_l = config.hue_yellow_l;
    hue_yellow_h = config.hue_yellow_h;
    saturation_yellow_l = config.saturation_yellow_l;
    saturation_yellow_h = config.saturation_yellow_h;
    lightness_yellow_l = config.lightness_yellow_l;
    lightness_yellow_h = config.lightness_yellow_h;
}

void DetectLaneNodelet::connectCB()
{
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if (!sub_projected_image_) {
        NODELET_INFO("[Detct Lane] Activate image subscriber...");
        image_transport::TransportHints hints("raw", ros::TransportHints(), getPrivateNodeHandle());
        sub_projected_image_ = it_->subscribe("/detect/image_input", queue_size_, &DetectLaneNodelet::imageCB, this, hints);
    }
}

void DetectLaneNodelet::disconnectCB()
{
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    if (pub_lane_.getNumSubscribers() == 0) {
        NODELET_INFO("[Detct Lane] Shutdown image subscriber...");
        sub_projected_image_.shutdown();
    }
}

void DetectLaneNodelet::imageCB(const sensor_msgs::ImageConstPtr& image_msg)
{
    // Change the frame rate by yourself. Now, it is set to 1/3 (10fps). 
    // Unappropriate value of frame rate may cause huge delay on entire recognition process.
    // This is up to your computer's operating power.
    if (counter % 3 != 0) {
        counter += 1;
        return;
    }
    else counter = 1;

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
    }
    catch (cv_bridge::Exception& e) {
        NODELET_ERROR("cv_bridge exception: %s", e.what());
        return;
    }        
    cv::Mat cv_image;
    (cv_ptr->image).copyTo(cv_image);

    // Find White and Yellow Lanes
    Mat hsv_img;
    cvtColor(cv_image, hsv_img, CV_BGR2HSV);

    cv::Mat cv_white_lane;
    vector<Point> white_lane_pts;
    int white_fraction = maskWhiteLaneFromHSV(hsv_img, cv_white_lane, white_lane_pts);
    
    cv::Mat cv_yellow_lane;
    vector<Point> yellow_lane_pts;
    int yellow_fraction = maskYellowLaneFromHSV(hsv_img, cv_yellow_lane, yellow_lane_pts);    
    // NODELET_INFO("yellow_fraction = %d, white_fraction = %d", yellow_fraction, white_fraction);

    // try {
    //     NODELET_INFO("Execute fit_from_lines()...");
    //     if (yellow_fraction > 3000) {
    //         fit_from_lines(cv_yellow_lane, yellow_lane_fit);
    //         mov_avg_left.append(yellow_lane_fit);
    //     }

    //     if (white_fraction > 3000) {
    //         fit_from_lines(cv_white_lane, white_lane_fit);
    //         mov_avg_right.append(white_lane_fit);
    //     }
    // }
    // catch (int ex) {
        // NODELET_INFO("Execute sliding_window() on yellow mask...");
        if (yellow_fraction > 3000) {
            sliding_window(cv_yellow_lane, yellow_lane_pts, LEFT_SIDE, yellow_lane_fit);
            // mov_avg_left.init(yellow_lane_fit);
            mov_avg_left.append(yellow_lane_fit);
        }

        // NODELET_INFO("Execute sliding_window() on white mask...");
        if (white_fraction > 3000) {
            sliding_window(cv_white_lane, white_lane_pts, RIGHT_SIDE, white_lane_fit);
//            mov_avg_right.init(white_lane_fit);
            mov_avg_right.append(white_lane_fit);
        }
    // }

    // NODELET_INFO("Get average lane fit....");
    yellow_lane_fit = mov_avg_left.mean();
    white_lane_fit = mov_avg_right.mean();
    // NODELET_INFO("yellow_lane_fit : %f %f %f", yellow_lane_fit.at(0), yellow_lane_fit.at(1), yellow_lane_fit.at(2));
    // NODELET_INFO("white_lane_fit : %f %f %f", white_lane_fit.at(0), white_lane_fit.at(1), white_lane_fit.at(2));

    make_lane(cv_image, white_fraction, yellow_fraction);
}

}       // lane_detection namespace

// Register nodelet
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(lane_detection::DetectLaneNodelet, nodelet::Nodelet)
