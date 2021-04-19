#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import random
import time
import os
import numpy as np
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self, name, pos=np.array([0.6,0]),visualize = True):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('multi_robot/scripts',
                                                'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = pos[0]
        self.init_goal_y = pos[1]
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = name
        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0
        self.vis = visualize

    def checkModel(self, model):
        self.check_model = False
        # print(self.check_model)
        # print(model.name)
        for i in range(len(model.name)):
            if model.name[i] == self.modelName:
                self.check_model = True
                # print(self.check_model)

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo('respawned the goal {}'.format(self.modelName))
                break
            else:
                pass

    def deleteModel(self):
        if self.vis:
            while True:
                if self.check_model:
                    rospy.wait_for_service('gazebo/delete_model')
                    del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                    del_model_prox(self.modelName)
                    rospy.loginfo('deleted the goal {}'.format(self.modelName))
                    break
                else:
                    pass

    def getPosition(self, position_check=False, delete=False, pos=None,other_objects_indicies=[]):
        self.last_index = self.index
        rospy.logdebug('Goal {} started changing position'.format(self.modelName))
        if delete and self.vis:
            rospy.logdebug('Goal {} is gonna delete model'.format(self.modelName))
            self.deleteModel()
        if not (pos is None):
            self.goal_position.position.x = pos[0]
            self.goal_position.position.y = pos[1]

        elif self.stage != 4:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x_list = [0.6,   1.9,    0.5,    -0.8,   -1.9,   0.5,    2, 0, -0.1, -2,2.059, 1.627, 1.631, 1.529, 0.96, 0.948, 0.426, 0.42, 0.415, -0.119, -0.136, -0.682, -0.735, -1.278, -1.301, -1.868, -1.997, -1.113, -0.515, 0.601, 1.159, -2.033, -1.122, -0.51, 0.536, 1.091, 1.686, -1.392, -0.277, 0.597, 0.769, 1.507, 2.075, 2.043, 1.539, 1.043, 1.757, -0.636, -0.734, -1.217]
                goal_y_list = [0,     -0.5,   -1.9,   -0.9,   1.1,    -1.5,   1.5, -1, 1.6, -0.8, 1.04, 2.049, 1.466, 0.873, 1.422, 2.044, 2.069, 1.514, 0.983, 2.054, 0.961, 2.051, 1.105, 2.064, 1.127, 2.02, 0.592, 0.659, 0.656, 0.653, 0.693, -0.016, 0.045, 0.064, 0.08, 0.116, 0.083, -0.736, -0.464, -0.45, -0.979, -1.029, -1.078, -1.618, -1.597, -2.09, -2.093, -2.055, -1.478, -2.064]

                self.index = random.randrange(0, len(goal_x_list))
                # rospy.logdebug('Goal {} : new indx is {}, old index is {}, other goals indeces {}'.format(self.modelName, self.index, self.last_index, other_objects_indicies))
                if (self.last_index == self.index) or (self.index in other_objects_indicies):
                    position_check = True
                else:
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]
            # rospy.logdebug('Goal {} cleared the position check'.format(self.modelName))

        if self.vis:
            # time.sleep(0.5)
            self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return (self.goal_position.position.x, self.goal_position.position.y), self.index, self.last_index
