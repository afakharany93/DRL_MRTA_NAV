from datetime import datetime
import time, random, threading
import csv
import rospy

class StepDataDumper():
    def __init__ (self, expr_path, name='', threads=2, h=None, start_time=0):
        self.file_number = 0
        self.line_count = 0
        self.start_time = start_time
        self.name = name
        self.expr_path = expr_path
        self.f = open(self.expr_path+"/{}_{}.csv".format(str(self.file_number).zfill(3), self.name), "w")
        self.writer = csv.writer(self.f)
        self.h = h
        self.threads = threads
        self.prepare_csv_header()

    def prepare_csv_header(self, stop_signal=False):
        columnTitleRow = ['absolute_time', 'ros_time', 'episode','steps']
        for i in range(self.threads):
            columnTitleRow.append(self.h.names_list[i]+'_x')
            columnTitleRow.append(self.h.names_list[i]+'_y')
        columnTitleRow = columnTitleRow + ['robot_x', 'robot_y', 'robot_yaw', 'robot_x_dot', 'robot_y_dot', 'robot_wz']
        columnTitleRow = columnTitleRow + ['action_l', 'action_r', 'value', 'done']
        columnTitleRow = columnTitleRow + ['reward', 'distance_reward', 'ob_reward', 'other_robot_dist_reward', 'time_reward', 'w_reward', 'v_reward']
        columnTitleRow = columnTitleRow + ['collision', 'reach_goal', 'reach_all_goals', 'duration_done', 'HZ', 'min_ditance']
        if not stop_signal:
            self.writer.writerow(columnTitleRow)
            self.f.flush()

    def check_new_file(self):
        if self.line_count >= 30000:
            self.f.close()
            self.file_number +=1
            self.line_count = 0
            self.f = open(self.expr_path+"/{}_{}.csv".format(str(self.file_number).zfill(3), self.name), "w")
            self.writer = csv.writer(self.f)
            self.prepare_csv_header()

    def write_csv_row(self, network_op, episode_state, robot_info, stop_signal=False):
        abs_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))
        # self.total_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        ros_time = rospy.get_time()
        collision = episode_state[0]
        reach_all_goals = episode_state[2]
        duration_done = episode_state[3]

        local_n_episode = robot_info[0]
        steps = robot_info[1]
        robot_name = robot_info[2]
        odom = robot_info[3]
        logging_data = robot_info[4]
        min_distance = robot_info[5]*3.5

        row = [abs_time, ros_time, local_n_episode, steps]
        goals_positions = self.h.get_all_goal_positions()
        for i in range(self.threads):
            row.append(goals_positions[i,0])
            row.append(goals_positions[i,1])
        row = row + list(odom)
        row = row + network_op
        row = row + list(logging_data)
        row = row + episode_state
        row.append(min_distance)
        if not stop_signal:
            self.writer.writerow(row)
            self.f.flush()
            self.line_count += 1
            self.check_new_file()
    def end(self):
        self.f.close()
