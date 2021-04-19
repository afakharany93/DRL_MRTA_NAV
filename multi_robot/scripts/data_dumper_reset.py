import csv, time, rospy

class ResetDataDumper():
    def __init__ (self, expr_path, start_time=0):
        self.expr_path = expr_path
        self.start_time = start_time
        self.f = open(self.expr_path+"/{}.csv".format('reset'), "w")
        self.writer = csv.writer(self.f)

        self.prepare_csv_header()

    def prepare_csv_header(self):
        columnTitleRow = ['absolute_time', 'ros_time', 'episode', 'steps', 'name',
         'collision', 'reach_all_goals', 'duration_done']
        self.writer.writerow(columnTitleRow)
        self.f.flush()

    def write_csv_row(self, episode_state, robot_info, stop_signal = False):
        abs_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time))
        # self.total_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        ros_time = rospy.get_time()

        collision = episode_state[0]
        reach_all_goals = episode_state[2]
        duration_done = episode_state[3]

        local_n_episode = robot_info[0]
        steps = robot_info[1]
        robot_name = robot_info[2]

        row = [abs_time, ros_time, local_n_episode,
                        steps, robot_name, collision, reach_all_goals, duration_done]

        if not stop_signal:
            self.writer.writerow(row)
            self.f.flush()
    def end(self):
        self.f.close()



# f_reset = open(expr_path+"/{}.csv".format('reset'), "w")
# writer_reset = csv.writer(f_reset)
# def prepare_reset_csv_header():
#     columnTitleRow = ['absolute_time', 'ros_time', 'episode', 'steps', 'name',
#      'collision', 'reach_all_goals', 'duration_done']
#     writer_reset.writerow(columnTitleRow)
#     f_reset.flush()
#
#
#
# prepare_reset_csv_header()

# write_reset_csv_row([abs_time, ros_time, self.local_n_episode,
#                 self.steps, self.env.robot_name, collision, reach_all_goals, duration_done])
