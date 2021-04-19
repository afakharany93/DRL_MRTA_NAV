from datetime import datetime
import time, random, threading
import csv
import rospy

class ActionDataDumper():
    def __init__ (self, expr_path, name='', start_time=0):
        self.file_number = 0
        self.line_count = 0
        self.start_time = start_time
        self.name = name
        self.expr_path = expr_path
        self.f = open(self.expr_path+"/{}_{}_actions.csv".format(str(self.file_number).zfill(3), self.name), "w")
        self.writer = csv.writer(self.f)
        self.prepare_csv_header()
        self.tick = rospy.get_time()

    def prepare_csv_header(self, stop_signal=False):
        columnTitleRow = ['absolute_time', 'ros_time', 'episode', 'steps', 'category', 'duration']
        if not stop_signal:
            self.writer.writerow(columnTitleRow)
            self.f.flush()

    def check_new_file(self):
        if self.line_count >= 30000:
            self.f.close()
            self.file_number +=1
            self.line_count = 0
            self.f = open(self.expr_path+"/{}_{}_actions.csv".format(str(self.file_number).zfill(3), self.name), "w")
            self.writer = csv.writer(self.f)
            self.prepare_csv_header()

    def write_csv_row(self, category, robot_info, stop_signal):
        abs_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))
        # self.total_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        ros_time = rospy.get_time()
        duration = ros_time - self.tick

        local_n_episode = robot_info[0]
        steps = robot_info[1]

        row = [abs_time, ros_time, local_n_episode, steps, category, duration]

        if not stop_signal:
            self.writer.writerow(row)
            self.f.flush()
            self.line_count += 1
            self.check_new_file()
        self.tick = rospy.get_time()

    def end(self):
        self.f.close()
