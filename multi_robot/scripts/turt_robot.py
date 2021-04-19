#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import sys
import numpy as np
import os
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates, ModelState
from sensor_msgs.msg import LaserScan
import roslaunch
import random

class RobotState():
    def __init__(self):
        self.odom = np.zeros(6)
        self.laser_range = np.zeros(360)
        self.t = 0
        self.name = 'name'

class TurtBot():
    def __init__(self, name='Robot', x_pos = 0.6, y_pos = 0.0, z_or=-100):
        self.state = RobotState()
        self.robot_x_list = [0.6,   1.9,    0.5,    -0.8,   -1.9,   0.5,    2, 0, -0.1, -2,2.059, 1.627, 1.631, 1.529, 0.96, 0.948, 0.426, 0.42, 0.415, -0.119, -0.136, -0.682, -0.735, -1.278, -1.301, -1.868, -1.997, -1.113, -0.515, 0.601, 1.159, -2.033, -1.122, -0.51, 0.536, 1.091, 1.686, -1.392, -0.277, 0.597, 0.769, 1.507, 2.075, 2.043, 1.539, 1.043, 1.757, -0.636, -0.734, -1.217]
        self.robot_y_list = [0,     -0.5,   -1.9,   -0.9,   1.1,    -1.5,   1.5, -1, 1.6, -0.8, 1.04, 2.049, 1.466, 0.873, 1.422, 2.044, 2.069, 1.514, 0.983, 2.054, 0.961, 2.051, 1.105, 2.064, 1.127, 2.02, 0.592, 0.659, 0.656, 0.653, 0.693, -0.016, 0.045, 0.064, 0.08, 0.116, 0.083, -0.736, -0.464, -0.45, -0.979, -1.029, -1.078, -1.618, -1.597, -2.09, -2.093, -2.055, -1.478, -2.064]
        self.robot_name = name
        self.robot_ns = name+'_ns'
        self.robot_tf_ns = name+'_tf'
        self.check_model = False
        self.pos = Pose()
        self.pos.position.x = x_pos
        self.pos.position.y = y_pos
        if z_or == -100:
            self.pos.orientation.z = (1 + 1) * np.random.random_sample() - 1
        else:
            self.pos.orientation.z = z_or
        self.pos.orientation.w = 1
        # rospy.loginfo(self.pos.orientation.z)
        self.gazebo_model_state = ModelState()
        self.gazebo_model_state.model_name = self.robot_name
        self.n_odom_messages = 0 #number of recieved odometry messages
        self.n_laser_scan_messages = 0 #number of recieved laser scanner messages
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.spawn_robot()
        while not self.check_model:
            pass
        rospy.loginfo("spawned robot at:")
        rospy.loginfo(self.pos)


    def checkModel(self, model):
        self.check_model = False
        # print(self.check_model)
        # print(model.name)
        for i in range(len(model.name)):
            if model.name[i] == self.robot_name:
                self.check_model = True
                # print(self.check_model)
    def spawn_robot(self):
        model=rospy.get_param('/robot_description')
        rospy.set_param('/{}/tf_prefix'.format(self.robot_ns), self.robot_tf_ns)
        rospy.wait_for_service('gazebo/spawn_urdf_model')
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_urdf_model', SpawnModel)
        spawn_model_prox(self.robot_name, model, self.robot_ns, self.pos, "world")
        self.sub_odom = rospy.Subscriber('/{}/odom'.format(self.robot_ns), Odometry, self.odometry_callback, queue_size=1)
        self.sub_laser_scan = rospy.Subscriber('/{}/scan'.format(self.robot_ns), LaserScan, self.laser_data_callback, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('/{}/cmd_vel'.format(self.robot_ns), Twist, queue_size=5)
        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        #wait until the first message of each sunscribed topic is recieved
        # help(self.sub_odom)
        while self.n_odom_messages < 1 or self.n_laser_scan_messages < 1:
            pass

    def odometry_callback(self,odom):
        # rospy.loginfo(odom)
        self.odom = odom
        self.n_odom_messages += 1

    def laser_data_callback(self, data):
        # rospy.loginfo(data)
        self.laser_raw = data
        self.n_laser_scan_messages += 1

    def send_vel(self, vx, wz):
        vel_cmd = Twist()
        vel_cmd.linear.x = vx
        vel_cmd.angular.z = wz
        self.pub_cmd_vel.publish(vel_cmd)
        return vel_cmd

    def quaternion_to_euler(self,vec):
        x, y, z, w = vec
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

    def get_odom(self):
        # rospy.loginfo(self.odom)
        t = self.odom.header.stamp.secs + self.odom.header.stamp.nsecs*1e-9
        x = self.odom.pose.pose.position.x
        y = self.odom.pose.pose.position.y
        x_dot = self.odom.twist.twist.linear.x
        y_dot = self.odom.twist.twist.linear.y
        wz = self.odom.twist.twist.angular.z
        orientation = self.odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.quaternion_to_euler(orientation_list)
        return np.array([x, y, yaw, x_dot, y_dot, wz]), t

    def get_laser_scan(self):
        max_value = self.laser_raw.range_max
        min_value = self.laser_raw.range_min
        ranges = np.array(self.laser_raw.ranges)
        ranges[ranges == np.inf] = max_value
        return np.nan_to_num(ranges)

    def get_state(self):
        '''
        Returns a dictionary state
        '''
        self.state.odom, self.state.t = self.get_odom()
        self.state.laser_range = self.get_laser_scan()
        self.state.name = self.robot_name
        return self.state

    def reset(self):
        # self.sub_laser_scan.unregister()
        # self.sub_odom.unregister()
        #
        #
        # # self.pub_cmd_vel.unregister()
        self.sub_odom = rospy.Subscriber('/{}/odom'.format(self.robot_ns), Odometry, self.odometry_callback, queue_size=1)
        self.sub_laser_scan = rospy.Subscriber('/{}/scan'.format(self.robot_ns), LaserScan, self.laser_data_callback, queue_size=1)
        # self.pub_cmd_vel = rospy.Publisher('/{}/cmd_vel'.format(self.robot_ns), Twist, queue_size=5)
        self.n_odom_messages = 0 #number of recieved odometry messages
        self.n_laser_scan_messages = 0 #number of recieved laser scanner message
        # self.send_vel(0,0)
        # if np.random.rand() <= 0.1:
        #     indx = np.random.randint(len(self.robot_x_list))
        #     pos_x = self.robot_x_list[indx]
        #     pos_y = self.robot_y_list[indx]
        #     self.gazebo_model_state.pose.position.x = pos_x
        #     self.gazebo_model_state.pose.position.y = pos_y
        #     self.pub_model.publish(self.gazebo_model_state)
        while self.n_odom_messages < 1 or self.n_laser_scan_messages < 1:
            pass
        rospy.logdebug('reset of {} robot'.format(self.robot_ns))

    def deleteModel(self):
        # not working
        self.pub_cmd_vel.unregister()
        self.sub_odom.unregister()
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox(self.robot_name)
