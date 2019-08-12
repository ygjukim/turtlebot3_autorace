#include <ros/ros.h>
#include <nodelet/loader.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_compensation", ros::init_options::AnonymousName);

    if (ros::names::remap("image_input") == "image_input") {
        ROS_WARN("Topic 'image_input' has not been remapped!!");
    }
    if (ros::this_node::getNamespace() == "/")
    {
        ROS_WARN("Started in the global namespace! This is probably wrong.\n"
                "Start image_compensation in the camera namespace.");
    }

    // Shared parameters to be propagated to nodelet private namespaces
    ros::NodeHandle private_nh("~");
    XmlRpc::XmlRpcValue shared_params;
    int queue_size;
    if (private_nh.getParam("queue_size", queue_size))
        shared_params["queue_size"] = queue_size;
    bool is_calibration_mode;    
    if (private_nh.getParam("is_extrinsic_camera_calibration_mode", is_calibration_mode))
        shared_params["is_extrinsic_camera_calibration_mode"] = is_calibration_mode;

    nodelet::Loader manager(false);
    nodelet::M_string remappings;
    nodelet::V_string my_argv(argv + 1, argv + argc);
    my_argv.push_back("--shutdown-on-close"); // Internal

    // remappings["imagae_input"] = ros::names::resolve("imagae_rect_color");
    // remappings["imagae_input/compressed"] = ros::names::resolve("imagae_rect_color/compressed");
    // remappings["imagae_output"] = ros::names::resolve("imagae_compensated");
    // remappings["imagae_output/compressed"] = ros::names::resolve("imagae_compensated/compressed");
 
    manager.load(ros::this_node::getName(), "turtlebot3_autorace_camera/image_compensation", remappings, my_argv);

    ros::spin();
    return 0;
}