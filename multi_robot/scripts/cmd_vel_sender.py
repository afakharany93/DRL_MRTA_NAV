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
import roslaunch
from respawnGoal import Respawn
from goals_handler import GoalHandler
import random


# goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
# goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]
os.system("roslaunch multi_robot one_robot.launch ns_:=robot1 ns_tf:=robot1_tf x_pos:=0.6 y_pos:=0.0 z_pos:=0.0 robot_name:=Robot1")
os.system("roslaunch multi_robot one_robot.launch ns_:=robot2 ns_tf:=robot2_tf x_pos:=1.9 y_pos:=-0.5 z_pos:=0.0 robot_name:=Robot2")
os.system("roslaunch multi_robot one_robot.launch ns_:=robot3 ns_tf:=robot3_tf x_pos:=0.5 y_pos:=-1.9 z_pos:=0.0 robot_name:=Robot3")
respawn_goal = Respawn('goal_1')
respawn_goal1 = Respawn('goal_2')
p = respawn_goal.getPosition(True)
print(p)
p = respawn_goal1.getPosition(True)
print(p)
# rospy.sleep(5)
# p = respawn_goal.getPosition(delete=True)
# print(p)
def getOdometry(odom):
    rospy.loginfo(odom)
def talker(vx=0.1,wz=2.8):
    # pub = rospy.Publisher('chatter', String, queue_size=10)
    h = GoalHandler(3)
    rospy.loginfo(h.get_all_robots_positions())
    pub_cmd_vel1 = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=5)
    pub_cmd_vel2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=5)
    pub_cmd_vel3 = rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=5)
    sub_odom = rospy.Subscriber('/robot1/odom', Odometry, getOdometry)
    rospy.init_node('vel_ctrlr', anonymous=True)
    rate = rospy.Rate(10) #hz
    rate2 = rospy.Rate(0.05) #hz
    while not rospy.is_shutdown():
        vel_cmd1 = Twist()
        vel_cmd1.linear.x = vx
        vel_cmd1.angular.z = wz
        pub_cmd_vel1.publish(vel_cmd1)

        vel_cmd2 = Twist()
        vel_cmd2.linear.x = vx
        vel_cmd2.angular.z = wz
        pub_cmd_vel2.publish(vel_cmd2)

        vel_cmd3 = Twist()
        vel_cmd3.linear.x = vx
        vel_cmd3.angular.z = wz
        pub_cmd_vel3.publish(vel_cmd3)

        # rospy.loginfo(vel_cmd1)
        # rospy.loginfo(vel_cmd2)
        # rospy.loginfo(vel_cmd3)
        rate.sleep()
        index = random.randrange(0, 3)
        h.respawn_certain_goal_by_index(index)
        # rospy.loginfo(h.get_all_robots_positions())
        rate2.sleep()

if __name__ == '__main__':
    print(sys.version)
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
