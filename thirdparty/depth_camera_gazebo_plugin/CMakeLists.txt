cmake_minimum_required(VERSION 2.8.3)
project(depth_camera_gazebo_plugin)
# package adapted by Rui Pimentel de Figueiredo to allow for

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
    roscpp
    gazebo_ros
    image_transport
    camera_info_manager
    sensor_msgs
)
find_package(gazebo REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

catkin_package(
    DEPENDS
        roscpp
        gazebo_ros
        image_transport
        camera_info_manager
        sensor_msgs
)

include_directories(
    include)
include_directories(
    SYSTEM
    ${catkin_INCLUDE_DIRS}
    ${GAZEBO_INCLUDE_DIRS}
)
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(
    ${PROJECT_NAME}
    src/DepthCameraPlugin.cpp
    src/gazebo_ros_depth_camera.cpp
    )

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})

install(
    TARGETS
        ${PROJECT_NAME}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

foreach(dir include)
    install(DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)
