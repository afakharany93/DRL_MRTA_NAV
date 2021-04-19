#!/usr/bin/env python3
import rospy
from respawnGoal import Respawn
import numpy as np
import threading

class GoalHandler():
    respawn_lock = threading.Lock()
    def __init__(self, n_goals=1, visualize = True,defined_init = False, test=False):
        self.test = test
        self.stage1_x = np.array([[1.481, 0.98, 0.3566, 0.9715, 0.3338, 0.3365, 0.92, 1.5395, 0],
        [-1.5119, -0.9243, -0.318, -0.8987, -0.3421, -0.318, -0.9547, -1.559, 1],
        [-1.5595, -0.8939, -0.302, -0.8966, -0.3062, -0.3104, -0.9319, -1.5119, 0],
        [1.5547, 0.996, 0.3214, 1.025, 0.3469, 0.3745, 0.958, 1.5871, -1]])

        self.stage1_y = np.array([[0.2898, 0.2764, 0.279, 0.97, 0.9646, 1.561, 1.4925, 1.4978, -1],
        [1.5723, 1.5543, 1.5457, 0.9813, 0.957, 0.3379, 0.3921, 0.4572, 0],
        [-0.3251, -0.337, -0.338, -0.9418, -0.9783, -1.523, -1.5295, -1.4508, 1],
        [-1.5328, -1.5458, -1.5523, -1.0102, -0.9859, -0.3901, -0.3684, -0.3797, 0]])

        self.stage2_x = np.array([[1.481, 0.3365, 0.92, 1.5395, 0.5, 0, 0],
        [-1.5119, -0.9243, -0.318, -1.559, 0,  1.5, 1],
        [-1.5595, -0.3104, -0.9319, -1.511, -0.5, -1.5, 0],
        [1.5547, 0.996, 0.3214, 1.025, 0, 0 , -1]])

        self.stage2_y = np.array([[0.2898, 1.561, 1.4925, 1.4978, 0, 0, -1],
        [1.5723, 1.5543, 1.5457, 0.4572, 0.5, 0, 0],
        [-0.3251,  -1.523, -1.5295, -1.4508, 0, 0, 1],
        [-1.5328, -1.5458, -1.5523, -1.0102, -0.5, -1.5, 0]])

        self.stage4_x = np.array([
        [1.62, 0.851, 0.671, 0.133],
        [2.043, 1.908, 1.024, 0.359],
        [-2.1, -0.26, -0.504, -0.791],
        [-2, -0.9, -1.129, -0.155]]) #-0.734, -1.217

        self.stage4_y = np.array([
        [1.95, 1.137, 0.513, 1.407],
        [-1.618, -0.203, -1.924, -2],
        [2.0, 0.754, 2.039, 0.841],
        [-0.8, -0.423, -0.375, -0.215]]) #-1.478, -2.064
        if n_goals==4:
            self.stage1_idx = np.array([[6, 3, 6, 2], [0, 2, 1, 3], [6, 3, 2, 3], [6, 0, 6, 1], [0, 1, 6, 6], [7, 4, 7, 3],
            [1, 3, 6, 6], [1, 2, 7, 2], [7, 6, 6, 6], [3, 3, 0, 4], [4, 6, 0, 5], [3, 2, 6, 2], [1, 3, 6, 2],
            [2, 6, 0, 1], [7, 7, 5, 4], [1, 2, 2, 5], [2, 4, 2, 1], [2, 0, 5, 0], [7, 5, 7, 3], [1, 6, 4, 1],
            [7, 4, 2, 6], [5, 4, 2, 2], [0, 1, 4, 1], [1, 6, 4, 3], [2, 3, 5, 2], [3, 2, 1, 2], [4, 1, 1, 1],
            [1, 7, 2, 7], [0, 0, 0, 4], [7, 0, 1, 2], [-1,-1,-1,-1]])


            self.stage2_idx = np.array([[0, 2, 2, 0], [0, 4, 0, 4], [3, 1, 0, 1], [3, 3, 2, 2], [1, 2, 4, 4],
           [2, 3, 3, 1], [3, 2, 2, 0], [2, 0, 4, 4], [3, 1, 0, 1], [3, 4, 4, 2],
           [4, 0, 0, 3], [2, 0, 0, 0], [0, 2, 1, 1], [1, 2, 0, 2], [4, 3, 1, 3],
           [1, 4, 3, 0], [3, 1, 4, 4], [3, 4, 2, 1], [2, 0, 0, 2], [4, 4, 2, 3],
           [3, 3, 3, 1], [2, 4, 4, 3], [4, 0, 2, 2], [2, 4, 2, 0], [2, 0, 0, 0],
           [4, 1, 3, 4], [0, 2, 0, 1], [2, 3, 4, 1], [1, 2, 3, 0], [0, 0, 3, 3],
           [-1,-1,-1,-1], [-2,-2,-2,-2], [-3,-3,-3,-3]])


            self.stage4_idx = np.array([[2, 1, 3, 1], [3, 3, 0, 1], [3, 1, 1, 3], [2, 3, 2, 0], [1, 2, 2, 1],
           [2, 2, 2, 2], [3, 0, 2, 2], [3, 0, 0, 2], [1, 3, 1, 2], [0, 1, 2, 2], [0, 2, 2, 2], [3, 2, 1, 1],
           [0, 3, 0, 2], [2, 0, 3, 3], [3, 3, 0, 0], [1, 0, 0, 3], [1, 3, 1, 2], [0, 2, 3, 0], [0, 2, 3, 3],
           [2, 3, 3, 2], [1, 3, 1, 2], [2, 3, 1, 0], [1, 2, 0, 3], [0, 3, 0, 1], [1, 3, 1, 1], [0, 0, 3, 0],
           [2, 3, 0, 3], [0, 2, 0, 0], [0, 0, 1, 3], [0, 2, 3, 1]])
        elif n_goals == 2:
            self.stage1_x = self.stage1_x.reshape((2,-1))
            self.stage1_y = self.stage1_y.reshape((2,-1))
            if self.test:
                # self.stage1_x = np.array([[0.9715, -1.5119, 0.3566, -0.9243, 0.3566, 0.3365, 0.92, 0.92, 1.481, 0.3365,
                #  1.481, 1.5395, 0.3566, 0.92, 0.9715,0.3566, 0.3566, 1.5395, 0.3566, 0.3566,
                #  -1.5119, 0.3338, 1.481, 0.9715, 1.481, 0.98, -1.5119, 0.9715, -0.9547, 0.3338,
                #  0.9715, 0.3338, 0.3365, 0.3338, 0.3566],
                # [-0.8966, -1, 0.958, -0.9319, -0.8966, -0.3062, -0.8966, -0.302, 0.958, -0.302,
                #  -0.8939, -1.5119, 0.958, -0.8966, 0.958, 0.958, 0.958, -0.3104, 0.958, -0.8939,
                #  -1, -0.8966, 0.3469, -0.8966, 0.3469, 0.996, -1, 0.958, -1, -0.8939, 0.958,
                #  1.025, -0.302, 0.3214, -0.8966]])
                # self.stage1_y = np.array([[0.97, 1.5723, 0.279, 1.5543, 0.279, 1.561, 1.4925, 1.4925, 0.2898, 1.561,
                # 0.2898, 1.4978, 0.279, 1.4925, 0.97, 0.279, 0.279, 1.4978, 0.279, 0.279, 1.5723,
                #  0.9646, 0.2898, 0.97, 0.2898, 0.2764, 1.5723, 0.97, 0.3921, 0.9646, 0.97,
                #  0.9646, 1.561, 0.9646, 0.279],
                # [-0.9418, 0, -0.3684, -1.5295, -0.9418, -0.9783, -0.9418, -0.338, -0.3684,
                # -0.338, -0.337, -1.4508, -0.3684, -0.9418, -0.3684, -0.3684, -0.3684, -1.523,
                # -0.3684, -0.337, 0, -0.9418, -0.9859, -0.9418, -0.9859, -1.5458, 0, -0.3684, 0,
                #  -0.337, -0.3684, -1.0102, -0.338, -1.5523, -0.9418]])
                # self.stage1_idx = np.ones((35,2),dtype=np.int)*np.arange(35).reshape((35,1))
                self.stage1_idx = np.array([[1,3],[1,6],[1,10],[2,0],[2,1],[2,3],[2,15],[2,16],
                [3,3],[3,4],[3,8],[3,11],[3,15],[4,1],[4,3],[4,10],[4,11],[4,12],[5,2],[5,4],
                [5,10],[6,2],[6,3],[7,5],[7,7],[7,13],[9,17],[10,6],[13,7],[15,17],[17,9],
                [17,11],[17,17],
                [17,13], [13,13], [10,13], [10,10], [11,10], [11,11], [9,11], [9,9], [5,9], [5,5]])
            else:
                self.stage1_idx = np.array([[1,3],[1,6],[1,10],[2,0],[2,1],[2,3],[2,15],[2,16],
                [3,3],[3,4],[3,8],[3,11],[3,15],[4,1],[4,3],[4,10],[4,11],[4,12],[5,2],[5,4],
                [5,10],[6,2],[6,3],[7,5],[7,7],[7,13],[9,17],[10,6],[13,7],[15,17],[17,9],
                [17,11],[17,17]])
            self.stage2_x = self.stage2_x.reshape((2,-1))
            self.stage2_y = self.stage2_y.reshape((2,-1))
            self.stage2_idx = np.array([[ 6,  5],[13, 11],[10,  0],[ 4,  6],[ 2,  0],
            [13,  6],[10,  8],[ 8,  3],[ 3,  3],[ 6,  0],[ 0,  4],[13,  6],[ 1,  1],
            [13,  4],[ 6, 11],[12,  8],[10,  8],[ 1,  7],[ 3,  8],[ 7,  1],[12,  8],
            [ 1,  5],[13,  8],[ 6, 11],[10, 12],[ 1,  0],[11, 12],[ 2, 12],[ 2,  8],
            [ 6, 10]])
            self.stage4_y = self.stage4_y.reshape((2,-1))
            self.stage4_x = self.stage4_x.reshape((2,-1))
            if self.test:
                self.stage4_idx = np.array([[5, 1],[1, 4],[2, 0],[7, 3],[7, 5],[4, 3],[7, 7],
                [1, 7],[4, 7],[5, 3],[5, 6],[2, 7],[1, 4],[5, 1],[7, 2],[7, 0],[2, 5],
                [3, 1],[2, 1],[7, 5],
                [1,1],[1,2],[2,2],[3,3],[4,4],[5,5],[6,6],[6,7],[5,2],[4,2]])
            else:
                self.stage4_idx = np.array([[5, 1],[1, 4],[2, 0],[7, 3],[7, 5],[4, 3],[7, 7],
                [1, 7],[4, 7],[5, 3],[5, 6],[2, 7],[1, 4],[5, 1],[7, 2],[7, 0],[2, 5],
                [3, 1],[2, 1],[7, 5]])

        self.n_goals = n_goals
        self.vis = visualize
        self.index_list = []
        self.stage = rospy.get_param('/stage_number')
        if defined_init:
            self.index_list = [47, 30, 16, 39]
        self.names_list = []
        self.goals_positions = np.zeros((self.n_goals,2))
        self.goals_list = self.spawn_goals()


    def generate_goals_names(self):
        '''
        generates names for each robot:
        returns :
        name_list: a list of robot names following the format goal_^ , where ^ is the robot number
        '''
        name_list = []
        for i in range(self.n_goals):
            name_list.append('goal_'+str(i))
        return name_list

    def get_goals_names(self):
        return self.names_list

    def spawn_goals_procedure(self, stage_idx, stage_x, stage_y):
        idx_of_idx = np.random.randint(len(stage_idx))
        idxs = stage_idx[idx_of_idx]
        goals_list=[]
        for i in range(self.n_goals) :
            name = self.names_list[i]
            goals_list.append(Respawn(name, visualize=self.vis))
            pos = np.array([stage_x[i][idxs[i]], stage_y[i][idxs[i]]])
            p, curr_indx, last_indx = goals_list[-1].getPosition(position_check=True,pos=pos, other_objects_indicies = self.index_list)
            if last_indx in self.index_list:
                self.index_list.remove(last_indx)
            self.index_list.append(curr_indx)
            self.goals_positions[i] = np.array(p)
        return goals_list

    def spawn_goals(self):
        self.names_list = self.generate_goals_names()
        # goals_list = []
        if self.stage == 1:
            goals_list = self.spawn_goals_procedure(self.stage1_idx, self.stage1_x, self.stage1_y)
            # idx_of_idx = np.random.randint(len(self.stage1_idx))
            # idxs = self.stage1_idx[idx_of_idx]
            # for i in range(self.n_goals) :
            #     name = self.names_list[i]
            #     goals_list.append(Respawn(name, visualize=self.vis))
            #     pos = np.array([self.stage1_x[i][idxs[i]], self.stage1_y[i][idxs[i]]])
            #     p, curr_indx, last_indx = goals_list[-1].getPosition(position_check=True,pos=pos, other_objects_indicies = self.index_list)
            #     if last_indx in self.index_list:
            #         self.index_list.remove(last_indx)
            #     self.index_list.append(curr_indx)
            #     self.goals_positions[i] = np.array(p)

        elif self.stage ==4:
            goals_list = self.spawn_goals_procedure(self.stage4_idx, self.stage4_x, self.stage4_y)

        elif self.stage == 2:
            goals_list = self.spawn_goals_procedure(self.stage2_idx, self.stage2_x, self.stage2_y)
        else:
            for i in range(self.n_goals) :
                name = self.names_list[i]
                goals_list.append(Respawn(name, visualize=self.vis))
                p, curr_indx, last_indx = goals_list[-1].getPosition(position_check=True, other_objects_indicies = self.index_list)
                if last_indx in self.index_list:
                    self.index_list.remove(last_indx)
                self.index_list.append(curr_indx)
                self.goals_positions[i] = np.array(p)
        return goals_list

    def get_all_goal_positions(self):
        return self.goals_positions

    def respawn_certain_goal_by_index(self, indx, extra_list = []):
        with GoalHandler.respawn_lock:
            if self.stage == 1 or self.stage == 4 or self.stage == 2:
                self.respawn_all_goals()
            else:
                p, curr_indx, last_indx = self.goals_list[indx].getPosition(position_check=True, delete=True, other_objects_indicies = self.index_list+extra_list)
                # rospy.logdebug((curr_indx, last_indx, self.index_list, last_indx in self.index_list))
                if last_indx in self.index_list:
                    self.index_list.remove(last_indx)
                    self.index_list.append(curr_indx)
                    self.goals_positions[indx] = np.array(p)
                    rospy.loginfo('Goal '+self.names_list[indx]+' has been respawned in position : '+str(p))

    def respawn_goals_procedure(self, stage_idx, stage_x, stage_y):
        idx_of_idx = np.random.randint(len(stage_idx))
        idxs = stage_idx[idx_of_idx]
        for i in range(self.n_goals) :
            name = self.names_list[i]
            # goals_list.append(Respawn(name, visualize=self.vis))
            pos = np.array([stage_x[i][idxs[i]], stage_y[i][idxs[i]]])
            p, curr_indx, last_indx = self.goals_list[i].getPosition(position_check=True,pos=pos, other_objects_indicies = self.index_list)
            if last_indx in self.index_list:
                self.index_list.remove(last_indx)
            self.index_list.append(curr_indx)
            self.goals_positions[i] = np.array(p)

    def respawn_all_goals(self):
        self.delete_all_goals()
        if self.stage == 1:
            goals_list = self.respawn_goals_procedure(self.stage1_idx, self.stage1_x, self.stage1_y)

        elif self.stage ==4:
            goals_list = self.respawn_goals_procedure(self.stage4_idx, self.stage4_x, self.stage4_y)

        elif self.stage == 2:
            goals_list = self.respawn_goals_procedure(self.stage2_idx, self.stage2_x, self.stage2_y)
        else:
            for i in range(self.n_goals) :
                name = self.names_list[i]
                # goals_list.append(Respawn(name, visualize=self.vis))
                p, curr_indx, last_indx = self.goals_list[i].getPosition(position_check=True, other_objects_indicies = self.index_list)
                if last_indx in self.index_list:
                    self.index_list.remove(last_indx)
                self.index_list.append(curr_indx)
                self.goals_positions[i] = np.array(p)
        # self.goals_list = self.spawn_goals()
        # rospy.loginfo(self.goals_positions)


    def delete_all_goals(self):
        for g in self.goals_list:
            rospy.loginfo('Deleting: '+g.modelName)
            g.deleteModel()
        # self.goals_positions = np.zeros((self.n_goals,2))
        # self.names_list = []
        # self.goals_list = []
        # self.index_list = []


# Exception in thread Thread-21:
# Traceback (most recent call last):
#   File "/usr/lib/python3.5/threading.py", line 914, in _bootstrap_inner
#   File "/home/cairo/catkin_ws/src/multi_robot/scripts/run.py", line 646, in run
#   File "/home/cairo/catkin_ws/src/multi_robot/scripts/run.py", line 456, in runEpisode
#     self.stacked_states_.reset()
#   File "/home/cairo/catkin_ws/src/multi_robot/scripts/goals_handler.py", line 210, in respawn_all_goals
#     self.goals_list = self.spawn_goals()
#   File "/home/cairo/catkin_ws/src/multi_robot/scripts/goals_handler.py", line 147, in spawn_goals
#     goals_list.append(Respawn(name, visualize=self.vis))
#   File "/home/cairo/catkin_ws/src/multi_robot/scripts/respawnGoal.py", line 36, in __init__
#     self.stage = rospy.get_param('/stage_number')
#   File "/opt/ros/kinetic/lib/python2.7/dist-packages/rospy/client.py", line 465, in get_param
#     return _param_server[param_name] #MasterProxy does all the magic for us
#   File "/opt/ros/kinetic/lib/python2.7/dist-packages/rospy/msproxy.py", line 121, in __getitem__
#     code, msg, value = self.target.getParam(rospy.names.get_caller_id(), resolved_key)
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1092, in __call__
#     return self.__send(self.__name, args)
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1432, in __request
#     verbose=self.__verbose
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1134, in request
#     return self.single_request(host, handler, request_body, verbose)
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1146, in single_request
#     http_conn = self.send_request(host, handler, request_body, verbose)
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1259, in send_request
#     self.send_content(connection, request_body)
#   File "/usr/lib/python3.5/xmlrpc/client.py", line 1289, in send_content
#     connection.endheaders(request_body)
#   File "/usr/lib/python3.5/http/client.py", line 1130, in endheaders
#     self._send_output(message_body)
#   File "/usr/lib/python3.5/http/client.py", line 946, in _send_output
#     self.send(msg)
#   File "/usr/lib/python3.5/http/client.py", line 889, in send
#     self.connect()
#   File "/usr/lib/python3.5/http/client.py", line 861, in connect
#     (self.host,self.port), self.timeout, self.source_address)
#   File "/usr/lib/python3.5/socket.py", line 693, in create_connection
#     for res in getaddrinfo(host, port, 0, SOCK_STREAM):
#   File "/usr/lib/python3.5/socket.py", line 732, in getaddrinfo
#     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
# OSError: [Errno 24] Too many open files
#
# Quit (core dumped)
