#include <ros/ros.h>
#include <nodelet/loader.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detect_lane_node", ros::init_options::AnonymousName);

    if (ros::names::remap("/detect/image_input") == "/detect/image_input") {
        ROS_WARN("Topic '/detect/image_input' has not been remapped!!");
    }

    // Shared parameters to be propagated to nodelet private namespaces
    ros::NodeHandle private_nh("~");
    XmlRpc::XmlRpcValue shared_params;
    int queue_size;
    if (private_nh.getParam("queue_size", queue_size))
        shared_params["queue_size"] = queue_size;
    bool is_calibration_mode;    
    if (private_nh.getParam("is_detection_calibration_mode", is_calibration_mode))
        shared_params["is_detection_calibration_mode"] = is_calibration_mode;

    nodelet::Loader manager(false);
    nodelet::M_string remappings;
    nodelet::V_string my_argv(argv + 1, argv + argc);
    my_argv.push_back("--shutdown-on-close"); // Internal

    manager.load(ros::this_node::getName(), "turtlebot3_autorace_detect/detect_lane", remappings, my_argv);

    ros::spin();
    return 0;
}