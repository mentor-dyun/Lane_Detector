cmake_minimum_required(VERSION 2.8.3)
project(lane_detector)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  std_msgs
  geometry_msgs
  dynamic_reconfigure
  message_generation
)
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MRPT REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Lane.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

##Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
    cfg/Detector.cfg
#   cfg/DynReconf2.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(CATKIN_DEPENDS message_runtime geometry_msgs)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${MRPT_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(LaneDetector src/LaneDetector.cc)
add_library(MCV src/mcv.cc)
add_library(InversePerspectiveMapping src/InversePerspectiveMapping.cc)
add_library(PREPROCESSOR src/preprocessor.cpp)
add_library(EXTRACTOR src/featureExtractor.cpp)
add_library(FITTING src/fitting.cpp)
add_library(FITTING_APPROACH src/fittingApproach.cpp)
add_library(TRACKER src/Ctracker.cpp)
add_library(HUNGARIAN_ALG src/HungarianAlg.cpp)
add_library(KALMAN_FILTER src/Kalman.cpp)

## Declare a C++ executable
add_executable(lane_detector src/lane_detector.cpp)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
add_dependencies(lane_detector ${PROJECT_NAME}_gencfg)
add_dependencies(FITTING ${PROJECT_NAME}_gencfg)
add_dependencies(EXTRACTOR ${PROJECT_NAME}_gencfg)
add_dependencies(PREPROCESSOR ${PROJECT_NAME}_gencfg)
add_dependencies(lane_detector ${PROJECT_NAME}_gencpp)
add_dependencies(FITTING ${PROJECT_NAME}_gencpp)
add_dependencies(EXTRACTOR ${PROJECT_NAME}_gencpp)
add_dependencies(PREPROCESSOR ${PROJECT_NAME}_gencpp)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(lane_detector ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against

target_link_libraries(MCV ${OpenCV_LIBRARIES})
target_link_libraries(InversePerspectiveMapping ${OpenCV_LIBRARIES} MCV)
target_link_libraries(LaneDetector ${OpenCV_LIBRARIES} MCV InversePerspectiveMapping)
target_link_libraries(PREPROCESSOR ${OpenCV_LIBRARIES} LaneDetector)
target_link_libraries(EXTRACTOR ${OpenCV_LIBRARIES} LaneDetector)
target_link_libraries(FITTING_APPROACH ${OpenCV_LIBRARIES} ${MRPT_LIBS})
target_link_libraries(KALMAN_FILTER ${OpenCV_LIBRARIES})
target_link_libraries(TRACKER ${OpenCV_LIBRARIES} KALMAN_FILTER HUNGARIAN_ALG)
target_link_libraries(FITTING ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} FITTING_APPROACH TRACKER LaneDetector)
target_link_libraries(lane_detector ${catkin_LIBRARIES} PREPROCESSOR FITTING EXTRACTOR)
target_link_libraries(lane_detector ${OpenCV_LIBRARIES} LaneDetector)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS lane_detector lane_detector_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#Disabling deprecating warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -fpermissive ")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wno-deprecated")

find_package(OpenMP)
if(OPENMP_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

#############
## Enabling C++11##
#############
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_lane_detector.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
