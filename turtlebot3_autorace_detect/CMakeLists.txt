################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(turtlebot3_autorace_detect)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS cmake_modules cv_bridge dynamic_reconfigure image_transport nodelet roscpp rospy)

#find_package(OpenCV REQUIRED core imgproc opencv_imgcodecs opencv_calib3d)
find_package(OpenCV  ${OpenCV_VERSION} EXACT REQUIRED opencv_calib3d)
find_package(GSL REQUIRED)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################
generate_dynamic_reconfigure_options(
  cfg/DetectLaneParams.cfg
  cfg/DetectLevelParams.cfg
  cfg/DetectTrafficLightParams.cfg
  cfg/DetectCalibrationParams.cfg
)

################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure
)

################################################################################
# Build
################################################################################
include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${GSL_INCLUDE_DIR})

add_library(lane_detection SHARED src/nodelet/detect_lane_nodelet.cpp)
target_link_libraries(lane_detection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} gsl gslcblas m)
#target_link_libraries(lane_detection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})
add_dependencies(lane_detection ${PROJECT_NAME}_gencfg)

add_executable(detect_lane_node src/node/detect_lane_node.cpp)
target_link_libraries(detect_lane_node ${catkin_LIBRARIES})

################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/detect_lane
  nodes/detect_level
  nodes/detect_parking
  nodes/detect_sign
  nodes/detect_traffic_light
  nodes/detect_tunnel
  nodes/detect_calibration
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg file launch param
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS lane_detection
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS detect_lane_node
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################

