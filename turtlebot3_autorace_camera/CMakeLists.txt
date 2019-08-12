################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(turtlebot3_autorace_camera)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS cmake_modules cv_bridge dynamic_reconfigure image_transport nodelet roscpp rospy)

#find_package(OpenCV REQUIRED core imgproc opencv_imgcodecs opencv_calib3d)
find_package(OpenCV  ${OpenCV_VERSION} EXACT REQUIRED opencv_calib3d)


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
  cfg/ImageProjectionParams.cfg
  cfg/ImageCompensationParams.cfg
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
include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(image_ext_calibration SHARED 
        src/nodelet/compensation_nodelet.cpp 
        src/nodelet/projection_nodelet.cpp)
target_link_libraries(image_ext_calibration ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(image_ext_calibration ${PROJECT_NAME}_gencfg)

add_executable(image_compensation_node src/node/image_compensation_node.cpp)
target_link_libraries(image_compensation_node ${catkin_LIBRARIES})

add_executable(image_projection_node src/node/image_projection_node.cpp)
target_link_libraries(image_projection_node ${catkin_LIBRARIES})

################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/image_compensation
  nodes/image_projection
 	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY calibration cfg data launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS image_ext_calibration
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(TARGETS image_compensation_node
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS image_projection_node
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################
