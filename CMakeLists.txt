cmake_minimum_required(VERSION 2.8.3)
project(waypoint_follower)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  roscpp
)

catkin_package(CATKIN_DEPENDS)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)
