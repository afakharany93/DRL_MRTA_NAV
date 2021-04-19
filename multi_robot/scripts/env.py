#!/usr/bin/env python3

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_msgs.msg import String, Float32
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
from turt_robot import TurtBot,RobotState

class Env():
    def __init__(self, goals, collective_states_callback, name='name', x_pos = 0.6, y_pos = 0.0,z_or=-100, env_size=(5,5), red_state=True, hz=5 ):
        self.hz = hz
        self.get_goalbox = False
        self.collision = False
        self.red_state = red_state
        # self.position = Pose()
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        # self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        # self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        #edits
        self.name = name
        self.robot_name = name+'_robot'
        self.robot = TurtBot(name=self.robot_name, x_pos = x_pos, y_pos = y_pos, z_or=z_or)
        self.goals = goals
        self.goals_positions = self.goals.get_all_goal_positions()
        self.robot_state = self.robot.get_state()
        self.max_dist = np.hypot(env_size[0],env_size[1]) /2

        self.collective_states_callback = collective_states_callback

        self.old_goals_data = np.zeros((2,self.goals.n_goals))
        self.old_norm_goals_data = np.zeros((2,self.goals.n_goals))
        self.old_t = self.robot_state.t
        self.old_laser_t = self.old_t
        self.old_laser_range = self.robot_state.laser_range
        self.old_norm_laser_range = self.old_laser_range/3.5

        self.entire_laser_data = np.vstack((self.robot_state.laser_range,np.zeros(360)))
        self.entire_norm_laser_data = np.vstack((self.robot_state.laser_range/3.5 ,np.zeros(360)))

        self.start_t = 0
        self.t = 0
        self.dt = 1
        self.laser_dt = 1
        # self.actions = np.array([0,0])

        self.action = np.array([0,0])
        self.old_distances_ratios = np.zeros(4)

        self.init_goal_data = self.getGoalsRelativeData()
        self.steps = 0.0

        # self.pub_distance_reward = rospy.Publisher('/{}/distance_reward'.format(self.robot.robot_ns), Float32, queue_size=10)
        # self.pub_heading_reward = rospy.Publisher('/{}/heading_reward'.format(self.robot.robot_ns), Float32, queue_size=10)
        # self.pub_ob_reward = rospy.Publisher('/{}/ob_reward'.format(self.robot.robot_ns), Float32, queue_size=10)
        # self.pub_other_robot_dist_reward = rospy.Publisher('/{}/other_robot_dist_reward'.format(self.robot.robot_ns), Float32, queue_size=10)
        # self.pub_reward = rospy.Publisher('/{}/reward'.format(self.robot.robot_ns), Float32, queue_size=10)
        # self.pub_time_reward = rospy.Publisher('/{}/time_reward'.format(self.robot.robot_ns), Float32, queue_size=10)

    def get_robot_state(self):
        self.robot_state = self.robot.get_state()

        self.goals_positions = self.goals.get_all_goal_positions()
        self.t = round(rospy.get_time(), 2)
        self.dt = self.t - self.old_t + 1e-7
        self.old_t = self.t
        self.other_robots_states = self.collective_states_callback(except_name=self.name, norm_data=True)
        self.other_robots_distances = self.collective_states_callback(except_name=self.name, actual_distances=True)

    def getGoalsRelativeData(self):
        difference = self.goals_positions - self.robot_state.odom[:2]
        self.goals_distances = np.around(np.hypot(difference[:,0],difference[:,1]), 2)
        # self.goals_distances = np.hypot(difference[:,0],difference[:,1])
        yaw = self.robot_state.odom[2]

        goal_angle = np.arctan2(difference[:,1], difference[:,0])

        heading = goal_angle - yaw
        heading = np.arctan2(np.sin(heading), np.cos(heading))

        self.heading = np.around(heading, 2)
        # self.heading = heading

        self.goals_data = np.vstack((self.goals_distances, self.heading))
        self.norm_goals_data = self.goals_data/np.array([[self.max_dist], [np.pi]])

        self.d_goals_data = (self.goals_data - self.old_goals_data)/self.dt
        self.d_norm_goals_data = (self.norm_goals_data - self.old_norm_goals_data)/self.dt

        self.goals_rel_odom = np.vstack((self.goals_data, self.d_goals_data ))
        self.norm_goals_rel_odom = np.vstack((self.norm_goals_data, self.d_norm_goals_data ))

        self.old_goals_data = self.goals_data.copy()
        self.old_norm_goals_data = self.norm_goals_data.copy()
        # print(self.d_norm_goals_data)
        # print(self.norm_goals_rel_odom)

        return self.goals_rel_odom, self.norm_goals_rel_odom

    def getLaserScanData(self):
        self.laser_range = self.robot_state.laser_range
        if not np.allclose(self.old_laser_range , self.laser_range):
            self.laser_range = np.around(self.laser_range, 3)
            # print(self.robot_state.name, self.laser_range)
            self.laser_dt = self.robot_state.t - self.old_laser_t + 1e-7
            self.d_laser_range = (self.laser_range - self.old_laser_range)/self.laser_dt

            self.norm_laser_range = self.laser_range/3.5
            self.d_norm_laser_range = (self.norm_laser_range - self.old_norm_laser_range)/self.laser_dt

            self.old_laser_range = self.laser_range.copy()
            self.old_norm_laser_range = self.norm_laser_range.copy()

            self.entire_laser_data = np.vstack((self.laser_range, self.d_laser_range))
            self.entire_norm_laser_data = np.vstack((self.norm_laser_range, self.d_norm_laser_range))

            self.old_laser_t = self.robot_state.t

        return self.entire_laser_data, self.entire_norm_laser_data

    def getState(self):
        self.get_robot_state()
        goals_rel_data, goals_norm_rel_data = self.getGoalsRelativeData()
        laser_data, norm_laser_data = self.getLaserScanData()
        current_t = rospy.get_time()

        difference = self.other_robots_distances - self.robot_state.odom[:2]
        robots_distances = np.hypot(difference[:,0],difference[:,1])
        robots_min_dist = np.min(robots_distances)
        # print(robots_min_dist)

        min_range = 0.2
        done = False
        current_min_dist = np.min(goals_rel_data[0,:])
        if current_min_dist < 0.2:
            done = True
            self.get_goalbox = True
        else:
            self.get_goalbox = False

        self.collision = False
        if not self.get_goalbox:
            if min_range > np.min(laser_data[0,:]) or robots_min_dist < 0.43:
                done = True
                self.collision = True

        return goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done


    def setReward(self, state): #linear rewards
        # rospy.loginfo((self.name, self.robot_state.t, self.t, self.old_t, self.dt))
        max_reward = 15
        factor = 1
        goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done = state
        time_from_steps = self.steps/self.hz # assuming we are operating at 5 hz
        # distances_ratios = goals_rel_data[0]/(self.init_goal_data[0][0] + 1e-7)
        # print(distances_ratios.shape)

        # indx = np.argmin(distances_ratios)
        indx = np.argmin(goals_norm_rel_data[0])
        nearst_robot_dist_ratio_val = np.min(self.other_robots_states[:,0,indx])

        # self.distance_reward = -1*distances_ratios[indx] + 1
        # if not self.old_distances_ratios.any(): #if all old data are zeros
        #     self.distance_reward = 0
        # else:
        self.distance_reward = goals_norm_rel_data[2,indx]*-200
        self.distance_reward = np.minimum(np.maximum(self.distance_reward, -10),10) / factor
        # print(self.distance_reward)
        # self.distance_reward = np.clip(self.distance_reward, -10, 10)
        # heading_reward = 0.3*np.cos(goals_rel_data[1, indx])
        #the 0.4 shift is made so that the reward is negative values start at 0.75 meters
        # self.ob_reward = 4*((2*np.min(norm_laser_data[0])-1)*0.7+0.4)/factor
        self.ob_reward = 0
        #the 0.487 shift is made so that the reward is negative values start at 0.75 meters
        self.other_robot_dist_reward = 7*(np.maximum(0.5*np.log10(nearst_robot_dist_ratio_val),-10)+0.487)/factor
        self.time_reward = np.minimum(-5*current_t / 120.0, 0)/factor

        # self.vx_reward = np.abs(self.action[0])*0.8 /factor
        self.vx_reward = 0
        self.wz_reward = 0
        # if np.abs(self.action[1]) >= 0.7:
        #     self.wz_reward = np.minimum(-1*np.abs(self.action[1]), 0)*1.0/factor

        if time_from_steps >= 119.0 or current_t>=119.0:
            self.time_reward = -1*max_reward/factor

        if self.collision:
            rospy.loginfo(self.robot_name+" had a Collision!!")
            if not self.get_goalbox:
                self.ob_reward = -1*max_reward/factor
            self.robot.send_vel(vx=0, wz=0)
            # self.collision = False

        if self.get_goalbox:
            #if another robot is not at the goal position
            if nearst_robot_dist_ratio_val > 0.02829:
                # rospy.loginfo(self.robot_name+" reached the Goal!!")
                self.distance_reward = max_reward/factor
                self.robot.send_vel(vx=0, wz=0)
                # self.goals.respawn_certain_goal_by_index(indx)
                self.goals_positions = self.goals.get_all_goal_positions()
                self.init_goal_data = self.getGoalsRelativeData()
                # self.get_goalbox = False
            #if another robot is at the goal position
            else:
                rospy.loginfo(self.robot_name+" have reached the goal but sadly another robot is on its way to it!!")
                self.distance_reward = -1*max_reward/factor
                self.robot.send_vel(vx=0, wz=0)
                # self.goals.respawn_certain_goal_by_index(indx)
                self.goals_positions = self.goals.get_all_goal_positions()
                self.init_goal_data = self.getGoalsRelativeData()
        # self.reward = self.reward/30
        self.reward = self.distance_reward + self.ob_reward + self.other_robot_dist_reward + self.time_reward + self.wz_reward + self.vx_reward
        # self.reward = self.distance_reward + self.heading_reward + self.ob_reward + self.other_robot_dist_reward + self.time_reward
        # self.reward = self.distance_reward + self.ob_reward
        # self.reward = np.minimum(np.maximum(self.reward, -max_reward+5),max_reward+5)
        return self.reward


    def generate_state(self):
        state = self.getState()
        goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done = state
        # actions = np.array([ action[0], action[1] ])
        current_t = round(rospy.get_time(),2)
        time_and_actions = np.array([ current_t/120.0, self.action[0], self.action[1] ])
        # return (goals_norm_rel_data, norm_laser_data, self.actions), done
        if self.red_state:
            return (goals_norm_rel_data[:2], self.other_robots_states[:,:2], norm_laser_data[:1], time_and_actions), done
        else:
            return (goals_norm_rel_data, self.other_robots_states, norm_laser_data, time_and_actions), done

    def take_action(self, action):
        self.action = action
        self.robot.send_vel(vx=action[0]*0.26, wz=action[1]*1.8)
        self.steps += 1.0

    def generate_reward_and_next_state(self):
        state = self.getState()
        goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done = state
        current_t = round(rospy.get_time(),2)
        time_and_actions = np.array([ current_t/120.0, self.action[0], self.action[1] ])
        reward = self.setReward(state)
        goals_norm_rel_data_with_other_robots = np.insert(self.other_robots_states, 0, goals_norm_rel_data, axis=0)
        if self.red_state:
            return (goals_norm_rel_data[:2], self.other_robots_states[:,:2], norm_laser_data[:1], time_and_actions), reward, done
        else:
            return (goals_norm_rel_data, self.other_robots_states, norm_laser_data, time_and_actions), reward, done

    def step(self, action):
        #needs testing
        # if not(self.collision or self.get_goalbox):
        self.action = action
        self.robot.send_vel(vx=action[0]*0.26, wz=action[1]*1.8)
        state = self.getState()
        goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done = state
        current_t = round(rospy.get_time(),2)
        time_and_actions = np.array([ current_t/120.0, action[0], action[1] ])
        reward = self.setReward(state)
        goals_norm_rel_data_with_other_robots = np.insert(self.other_robots_states, 0, goals_norm_rel_data, axis=0)
        # print(goals_norm_rel_data)
        self.steps += 1.0
        if self.red_state:
            return (goals_norm_rel_data[:2], self.other_robots_states[:,:2], norm_laser_data[:1], time_and_actions), reward, done
        else:
            return (goals_norm_rel_data, self.other_robots_states, norm_laser_data, time_and_actions), reward, done

    def get_logging_data(self):
        return self.reward, self.distance_reward, self.ob_reward, self.other_robot_dist_reward, self.time_reward, self.wz_reward , self.vx_reward


    def reset(self, do_it_yourself=True):
        self.robot.send_vel(vx=0, wz=0)
        self.robot.reset()
        if do_it_yourself:
            rospy.wait_for_service('gazebo/reset_simulation')
            try:
                self.reset_proxy()
            except (rospy.ServiceException) as e:
                print("gazebo/reset_simulation service call failed")

        self.get_goalbox = False
        self.collision = False
        self.goals_positions = self.goals.get_all_goal_positions()
        self.init_goal_data = self.getGoalsRelativeData()
        self.old_laser_t = 0
        self.old_t = 0
        self.start_t = 0
        self.old_distances_ratios = np.zeros(4)
        self.steps = 0.0
        # self.actions = np.array([0,0])

        state = self.getState()
        goals_rel_data, goals_norm_rel_data, laser_data, norm_laser_data, current_t, done = state
        self.start_t = current_t
        goals_norm_rel_data_with_other_robots = np.insert(self.other_robots_states, 0, goals_norm_rel_data, axis=0)
        # return goals_norm_rel_data_with_other_robots, norm_laser_data, current_t
