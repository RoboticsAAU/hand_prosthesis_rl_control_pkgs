cmake_minimum_required(VERSION 3.5.1)
project(move_hand_pkg)
add_compile_options(-std=c++17)
# 
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(catkin REQUIRED COMPONENTS
#   camera_info_manager
  gazebo_dev
  gazebo_msgs
  gazebo_ros
  geometry_msgs
#   image_transport
#   cv_bridge
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
  trajectory_msgs
)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

catkin_package(
  LIBRARIES
    gazebo_ros_move_hand
  CATKIN_DEPENDS
    gazebo_dev
    gazebo_ros
    geometry_msgs
    nav_msgs
    tf2
    tf2_geometry_msgs
    tf2_ros
)


link_directories(${gazebo_dev_LIBRARY_DIRS})

option(ENABLE_PROFILER "Enable Ignition Profiler" FALSE)
if(ENABLE_PROFILER)
  find_package(ignition-common3 QUIET COMPONENTS profiler)
  if(ignition-common3_FOUND)
    add_definitions("-DIGN_PROFILER_ENABLE=1" "-DIGN_PROFILER_REMOTERY=1")
    message(STATUS "Profiler is active")
  else()
    message(WARNING "Can't find Ignition common3. Profiler will not be actived")
    add_definitions("-DIGN_PROFILER_ENABLE=0" "-DIGN_PROFILER_REMOTERY=0")
  endif()
endif()

# gazebo_ros_simple_quad
add_library(gazebo_ros_move_hand SHARED
  src/gazebo_ros_move_hand.cpp
)
target_include_directories(gazebo_ros_move_hand PUBLIC include ${GAZEBO_INCLUDE_DIRS} ${geometry_msgs_INCLUDE_DIRS})

add_dependencies(gazebo_ros_move_hand ${catkin_EXPORTED_TARGETS})

if(ENABLE_PROFILER)
  target_include_directories(gazebo_ros_move_hand PUBLIC ${ignition-common3_INCLUDE_DIRS})
  target_link_libraries(gazebo_ros_move_hand ${ignition-common3_LIBRARIES})
endif()


# install(DIRECTORY
#   world
#   DESTINATION share/${PROJECT_NAME}/
# )

install(
  DIRECTORY src launch description
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
        DESTINATION include)

install(TARGETS
  gazebo_ros_move_hand
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(NOT GAZEBO_VERSION VERSION_LESS 9.5)
install(TARGETS
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
endif()