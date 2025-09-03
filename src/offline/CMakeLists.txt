cmake_minimum_required(VERSION 3.0.2)
project(epic_offline)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
find_package(PkgConfig REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    rospy 
    geometry_msgs 
    std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Install Python scripts
catkin_install_python(PROGRAMS
  neural_tsp_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python modules
catkin_python_setup()

# Install other files
install(FILES
  config.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY
  agent/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/agent
)
