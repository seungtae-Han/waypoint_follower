#!/usr/bin/env python

import rospy
import pandas as pd
import geodesy.utm
import math
from numpy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64, Int8
from sensor_msgs.msg import NavSatFix


class waypoint_follower:

    def __init__(self):

        self.LOOKAHEAD_DISTANCE = 2 # meters
        self.read_waypoints()
        self.goal_x = 0
        self.goal_y = 0
        self.past_point_x = 0
        self.past_point_y = 0
        self.counter = 0
        self.velocity = 40
        self.last_idx = 0

        # a matrix of distance between cars and coordinates
        self.dist_arr= np.zeros(len(self.path_points_y))

        self.steer_pub = rospy.Publisher('/steer_planner', Float64, queue_size=1)
        self.velocity_pub = rospy.Publisher('/velocity_planner', Float64, queue_size=1)
        self.sub_gps = rospy.Subscriber('/gps/fix', NavSatFix, self.callback_gps)  

    
    def read_waypoints(self):
        # read data(latitude, longitude, yaw) from csv file
        filename = "/home/seungtae/catkin_ws/src/test.csv"
        wayPoint = pd.read_csv(filename, names=['x', 'y', 'w'])
        path_points_lat = wayPoint.loc[:, 'x']
        path_points_long = wayPoint.loc[:, 'y'] 
        self.path_points_w = wayPoint.loc[:, 'w']

        # convert coordinates of the csv file to the utm coordinates
        self.path_points_x = []
        self.path_points_y = []

        count = len(path_points_lat)
        for i in range(0, count):
            wayPoint_utm_x = geodesy.utm.fromLatLong(path_points_lat[i], path_points_long[i]).toPoint().x
            wayPoint_utm_y = geodesy.utm.fromLatLong(path_points_lat[i], path_points_long[i]).toPoint().y

            self.path_points_x.append(wayPoint_utm_x)
            self.path_points_y.append(wayPoint_utm_y)


    # calculating distance
    def dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    # calculates the angle between two vectors
    def find_angle(self, v1, v2):
        cosang = np.dot(v1, v2)
        sinang = la.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)


    def callback_gps(self, data):
        # gps data
        gps_lat = data.latitude
        gps_long = data.longitude

        # convert gps data to the utm coordinates data
        gps_utm_x = geodesy.utm.fromLatLong(gps_lat, gps_long).toPoint().x
        gps_utm_y = geodesy.utm.fromLatLong(gps_lat, gps_long).toPoint().y
        
        # plot(waypoints, current location, goal point)
        plt.ion()
        if (self.counter % 2 == 0):
            plt.plot(self.path_points_x, self.path_points_y, 'ob', markersize=2)
            plt.plot(gps_utm_x, gps_utm_y, 'or', markersize=10)
            plt.plot(self.goal_x, self.goal_y, 'ok', markersize=8)
            plt.draw()
            plt.pause(0.00000000001)
            plt.clf()
        self.counter += 1

        # calculate yaw
        vector_1 = np.array([gps_utm_x - self.past_point_x, gps_utm_y - self.past_point_y])
        vector_1_distance = np.sqrt((gps_utm_x - self.past_point_x)**2 + (gps_utm_y - self.past_point_y)**2)
        
        vector_2 = np.array([1, 0])
        vector_2_distance = 1

        yaw_rad = np.arcsin(np.cross(vector_1, vector_2)/(vector_1_distance * vector_2_distance))
        yaw_deg = np.rad2deg(yaw_rad)

        # save current coordinates
        self.past_point_x = gps_utm_x
        self.past_point_y = gps_utm_y

        # calculate the distance between the current position and the waypoints
        for i in range(len(self.path_points_x)):
            self.dist_arr[i] = self.dist((self.path_points_x[i], self.path_points_y[i]), (gps_utm_x, gps_utm_y))

        # only the dots within the set distance are collected
        goal_arr = np.where((self.dist_arr < self.LOOKAHEAD_DISTANCE * 1.2) & (self.dist_arr > self.LOOKAHEAD_DISTANCE * 0.8))[0]
        
        # set the target point as a past target
        goal = self.last_idx

        # seek goal point
        for idx in goal_arr:
            v1 = [self.path_points_x[idx] - gps_utm_x , self.path_points_y[idx] - gps_utm_y]
            v2 = [np.cos(yaw_rad), np.sin(yaw_rad)]
            
            temp_angle = self.find_angle(v1, v2)
            
            if abs(temp_angle) < np.pi/2 and self.last_idx < idx:
                goal = idx
                break
        
        self.last_idx = goal

        self.goal_x = self.path_points_x[goal]
        self.goal_y = self.path_points_y[goal]
        
        # pure pursuit
        L = self.dist_arr[goal]
        alpha = self.path_points_w[goal] - yaw_rad
        k = 2 * math.sin(alpha) / L
        angle_i = math.atan(k)

        # NAN error handling
        if np.isnan(angle_i):
            angle_i = np.nan_to_num(angle_i)

        # +/- number handling
        if self.goal_x > gps_utm_x:
            angle = angle_i
        else:
            angle = -angle_i

        angle_deg = np.rad2deg(angle)
        angle_erp = angle_deg * 71 # final steering angle
        
        # control velocity
        if (abs(angle_erp) >= 1000):
            self.LOOKAHEAD_DISTANCE = 1.5
            self.velocity -= 1

            if (self.velocity <= 20):
                self.velocity = 20

        else:
            self.LOOKAHEAD_DISTANCE = 2
            self.velocity += 1

            if (self.velocity >= 70):
                self.velocity = 70

        # make publisher
        steer_msg = Float64()
        steer_msg.data = angle_erp
        self.steer_pub.publish(steer_msg)

        velocity_msg = Int8()
        velocity_msg.data = self.velocity
        self.velocity_pub.publish(self.velocity)

        #to make csv file
        # gps_imu_msg = Vector3Stamped()
        # gps_imu_msg.vector.x = gps_lat
        # gps_imu_msg.vector.y = gps_long
        # gps_imu_msg.vector.z = yaw_rad
        # self.pub_gps_imu.publish(gps_imu_msg)

        #DEBUGGING
        # print 'last_idx : ', self.last_idx
        # print 'goal : ', goal
        # print 'yaw : ', yaw_deg
        # print 'angle : ', angle_erp
        # print 'velocity : ', self.velocity
        # print 'past_point_x : ', self.past_point_x
        # print '***********************'


if __name__ == '__main__':

    try:

        rospy.init_node('waypoint_follower')
        waypoint_follower()
        rospy.spin()

    except rospy.ROSInterruptException:

        pass
