#!/usr/bin/env python3
import rospy
from std_srvs.srv import Empty
import numpy as np
from env import Env
import time

class EnvHandler():
    def __init__(self, num_envs, goals, defined_init = False, red_state=True, hz=5):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.unpauseSim()
        self.red_state = red_state
        self.stage = rospy.get_param('/stage_number')
        self.x_list = [0.6,   1.9,    0.5,    -0.8,   -1.9,   0.5,    2, 0, -0.1,
        -2,2.059, 1.627, 1.631, 1.529, 0.96, 0.948, 0.426, 0.42, 0.415, -0.119, -0.136,
        -0.682, -0.735, -1.278, -1.301, -1.868, -1.997, -1.113, -0.515, 0.601, 1.159, -2.033,
        -1.122, -0.51, 0.536, 1.091, 1.686, -1.392, -0.277, 0.597, 0.769, 1.507, 2.075, 2.043,
        1.539, 1.043, 1.757, -0.636, -0.734, -1.217, -0.918, -2.27]
        self.y_list = [0,     -0.5,   -1.9,   -0.9,   1.1,    -1.5,   1.5, -1, 1.6, -0.8,
        1.04, 2.049, 1.466, 0.873, 1.422, 2.044, 2.069, 1.514, 0.983, 2.054, 0.961, 2.051,
        1.105, 2.064, 1.127, 2.02, 0.592, 0.659, 0.656, 0.653, 0.693, -0.016, 0.045, 0.064,
        0.08, 0.116, 0.083, -0.736, -0.464, -0.45, -0.979, -1.029, -1.078, -1.618, -1.597,
        -2.09, -2.093, -2.055, -1.478, -2.064, 0.29, 1.30]
        if self.stage == 4:
            # if np.random.rand()>0.5:
            #     self.init_idx = [47, 50, 16, 39]
            # else:
            self.init_idx = [51, 40, 50, 47]
        else:
            self.init_idx = [13, 37, 24, 41]
        self.num_envs = num_envs
        self.envs = []
        self.goals = goals
        self.indx_list = [-1]
        for i in range(self.num_envs):
            if not defined_init:
                indx = -1
                while indx in self.indx_list:
                    indx = np.random.randint(len(self.x_list))
            else:
                indx = self.init_idx[i]
            self.indx_list.append(indx)
            orientation = -100
            if self.stage != 4:
                if self.x_list[indx]>0 and self.y_list[indx]>0:
                    orientation = 1000
                if self.x_list[indx]>0 and self.y_list[indx]<0:
                    orientation = 1000
                elif self.x_list[indx]<0 and self.y_list[indx]>0:
                    orientation = 0
                elif self.x_list[indx]<0 and self.y_list[indx]<0:
                    orientation = 1
            else:
                if self.x_list[indx]>0 and self.y_list[indx]>0:
                    orientation = 1000
                if self.x_list[indx]>0 and self.y_list[indx]<0:
                    orientation = 1000
                elif self.x_list[indx]<0 and self.y_list[indx]>0:
                    orientation = 0
                elif self.x_list[indx]<0 and self.y_list[indx]<0:
                    orientation = 1

            self.envs.append(Env(self.goals, self.get_collective_states, 'env_'+str(i), self.x_list[indx], self.y_list[indx], z_or=orientation, red_state=self.red_state, hz=hz ))


    def get_collective_states(self, except_name='', norm_data=True, actual_distances = False):
        if actual_distances:
            states = []
            for i in range(self.num_envs):
                if self.envs[i].name != except_name:
                    states.append(self.envs[i].robot_state.odom[:2])
            states_arr = np.array(states)
            # if len(states_arr.shape) == 1:
            #     states_arr = states_arr[np.newaxis,np.newaxis,:]
            # elif len(states_arr.shape) == 2 :
            #     states_arr = states_arr[np.newaxis,:]
        else :
            states = []
            for i in range(self.num_envs):
                if self.envs[i].name != except_name:
                    if norm_data:
                        states.append(self.envs[i].norm_goals_rel_odom )
                    else:
                        states.append(self.envs[i].goals_rel_odom )
            states_arr = np.array(states)
            if len(states_arr.shape) == 1:
                states_arr = states_arr[np.newaxis,np.newaxis,:]
            elif len(states_arr.shape) == 2 :
                states_arr = states_arr[np.newaxis,:]
        return states_arr

    def reset(self):
        x = True
        rospy.logdebug('Resetting')
        for i in range(len(self.envs)):
            self.envs[i].reset(do_it_yourself=x)
            x = x & False


    def check_for_ending_conditions(self):
        collision = False
        reach_all_goals = True
        for env in self.envs:
            collision = collision | env.collision
            reach_all_goals = reach_all_goals & env.get_goalbox
        # for env in self.envs:
        #     env.collision = False
        return reach_all_goals, collision

    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.loginfo ("/gazebo/unpause_physics service call failed")
