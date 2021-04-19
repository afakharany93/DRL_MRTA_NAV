#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import csv
import os
import time, random, threading
import argparse
import contextlib

class CreateReport():
    def __init__(self, path='', last_n=-1, new_dir=False):
        self.path = path+'/'
#         self.tr = tracker.SummaryTracker()
        self.last_n = last_n
        self.number_robots = 0
        self.stage1_x = np.array([[1.481, 0.98, 0.3566, 0.9715, 0.3338, 0.3365, 0.92, 1.5395, 0],
        [-1.5119, -0.9243, -0.318, -0.8987, -0.3421, -0.318, -0.9547, -1.559, 1],
        [-1.5595, -0.8939, -0.302, -0.8966, -0.3062, -0.3104, -0.9319, -1.5119, 0],
        [1.5547, 0.996, 0.3214, 1.025, 0.3469, 0.3745, 0.958, 1.5871, -1]])

        self.stage1_y = np.array([[0.2898, 0.2764, 0.279, 0.97, 0.9646, 1.561, 1.4925, 1.4978, -1],
        [1.5723, 1.5543, 1.5457, 0.9813, 0.957, 0.3379, 0.3921, 0.4572, 0],
        [-0.3251, -0.337, -0.338, -0.9418, -0.9783, -1.523, -1.5295, -1.4508, 1],
        [-1.5328, -1.5458, -1.5523, -1.0102, -0.9859, -0.3901, -0.3684, -0.3797, 0]])

        self.stage1_x = self.stage1_x.reshape((2,-1))
        self.stage1_y = self.stage1_y.reshape((2,-1))

    def update(self, last_n=-1, report_mem=True, test=False):
        self.last_n = last_n
        self.get_files_names(self.path+'*')

        if self.ordered_robot_files != []:
            # try:
            self.analyze_robot_files()

    def __del__(self):
        self.successful_episodes_handle.close()


    def get_files_names(self, path='*'):
        files_names = glob.glob(path)
        self.robot_files = [f for f in files_names if 'robot.csv' in f]
        self.robot_files.sort()
        self.ordered_robot_files = self.handel_robot_file_names()
        # print(self.ordered_robot_files)
        # self.robot_files = self.robot_files.sort()
        self.brain_files = [f for f in files_names if 'brain.csv' in f]
        self.reset_files = [f for f in files_names if 'reset.csv' in f]
        self.brain_files.sort()
        self.reset_files.sort()
        [os.remove(f) for f in files_names if 'indxs.txt' in f]
        [os.remove(f) for f in files_names if 'report.csv' in f]
        self.successful_episodes_handle = open(self.path+'successful_episodes.csv', 'a')
        self.successful_episodes_handle.flush()
        self.writer = csv.writer(self.successful_episodes_handle)

    def handel_robot_file_names(self):
        number_list = []
        name_list = []
        if self.robot_files != [] :
            for f in self.robot_files :
                # print(name_list)
                name = f.split('/')[-1]
                if name.split('_')[2].isdigit():
                    robot_number = int(name.split('_')[2])
                    # print(robot_number)
                    # indx = name.split('_')[0]
                    if robot_number not in number_list:
                        number_list.append(robot_number)
                        name_list.append([])
                    name_list[robot_number].append(f)
                    name_list[robot_number].sort()
        self.number_robots = len(number_list)
        # print(self.number_robots)
        return name_list

    def analyze_robot_files(self):
        colors=['red','green','blue','yellow', 'cyan']
        i=-1
        general_df = pd.DataFrame(columns=['episode','speed', 'min_init_distance', 'min_curnt_distance', 'min_time', 'act_time'])
        robot_pos = []
        all_robots_rewards = []
        state_dfs = []
        for a in range(len(self.ordered_robot_files)):
            tmp_df = pd.DataFrame(columns=['episode','speed', 'min_init_distance', 'min_curnt_distance', 'min_time', 'act_time'])
            robot_names_list =self.ordered_robot_files[a]
            name = robot_names_list[0][:-4].split('/')[-1]
            robot_num = name[8]

            i+=1
            if len(robot_names_list) > 1 :
                f =robot_names_list[0]
                data = pd.read_csv(f)
                robot_pos.append([data[['robot_x']].iloc[0][0] ,data[['robot_y']].iloc[0][0]])
                # print(robot_pos)
                for t in range(1,len(robot_names_list)):
                    f =robot_names_list[t]
                    data = pd.concat([data, pd.read_csv(f)], ignore_index=True)

            else:
                f =robot_names_list[0]
                data = pd.read_csv(f)
                robot_pos.append([data[['robot_x']].iloc[0][0] ,data[['robot_y']].iloc[0][0]])
                    # print(robot_pos)
            if self.last_n != -1 :
                data=data[-1*self.last_n:]
                self.episode_last_n = data['episode'].iloc[0]

            elapsed_time = data["absolute_time"].iloc[-1]
#             self.writer.writerow([name, data["absolute_time"].iloc[-1]])
#             self.successful_episodes_handle.flush()
            data_reach_all_g = data[data['reach_all_goals']==True]
            data_reach_all_g.to_csv(self.path+'successful_episodes.csv', mode='a')
            print(data_reach_all_g[['episode','ros_time', 'goal_0_x' ,
                                                      'goal_0_y',  'goal_1_x','goal_1_y']])
            goals_values = (data_reach_all_g[['goal_0_x' ,'goal_1_x']].values).T
            indxs = [np.where(self.stage1_x == goals_values[:,q:q+1])[1] for q in range(len(goals_values[0]))]
            indxs = [indxs[q] for q in range(len(indxs)) if len(indxs[q])==2 ]
            indxs = self.clear_duplicates(indxs)
            # print(type(indxs))
            # print(type(indxs[0]))
            # print(type(indxs[0][0]))
            indexes_string = ''.join( str(indxs) )
            indexes_string = indexes_string.replace(' ','')
            indexes_string = indexes_string.replace('array(','')
            indexes_string = indexes_string.replace(')','')
            print('The indesxes Are: \n',indexes_string)
            print('number of indexes: \n',len(indxs))
            # np.savetxt(, indxs, delimiter=",")
            with open(self.path+'indxs.txt'.format(i), 'a') as f:
                f.write( indexes_string )
                f.write('\n')
    def clear_duplicates(self, x):
        y = np.unique(x, axis=0)
        return list(y)
    def bar_error_plot(self,df,name='', legend=[], xlabel='', ylabel=''):
        labels = [(x.replace('_', ' ')).capitalize() for x in df.columns]
        x_pos = np.arange(len(labels))
        means = df.mean()
        stds = df.std()
        fig, ax = plt.subplots()
        ax.bar(x_pos, means,
               yerr=stds,
               align='center',
               alpha=0.5,
               ecolor='black',
               capsize=10)
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        if legend != []:
            ax.legend(legend)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.yaxis.grid(True)
        fig.tight_layout()
        fig.savefig(self.path+name)
        plt.close(fig)

    def plot(self, df, name='', legend=[], xlabel='', ylabel='',alpha=1.0):
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        fig, ax = plt.subplots()
        df.plot(ax=ax,alpha=alpha, grid=True)
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        if legend != []:
            ax.legend(legend)
        fig.tight_layout()
        fig.savefig(self.path+name)
        plt.close(fig)

    def scatter(self, df, x='',y='',c='red',name='', legend=[], xlabel='', ylabel=''):
        fig, ax = plt.subplots()
        df.plot.scatter(x=x, y=y,c=c,ax=ax, grid=True)
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        if legend != []:
            ax.legend(legend)
        fig.tight_layout()
        fig.savefig(self.path+name)
        plt.close(fig)


    def describtions_of_binary_col(self, df, col_name, condition = True, count_repeat = False):
        df_filtered = df[['ros_time', col_name]]
        # if self.last_n != -1 :
        #     df_filtered[col_name] = df_filtered[col_name]>0.5
        if not count_repeat:
            df_filtered = df_filtered[df_filtered[col_name]-df_filtered[col_name].shift() != 0 ]
        df_filtered = df_filtered[df_filtered[col_name] == condition]
        describtion = df_filtered.describe(include='all')
        return describtion
if __name__ == '__main__':
    import os
    path = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--path', type=str,default='None',
                            help='experiment directory path')

    args = parser.parse_args()

    if args.path != 'None':
        path = args.path

    report = CreateReport(path)
    report.update()
