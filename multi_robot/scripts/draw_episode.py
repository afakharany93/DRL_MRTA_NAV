#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as font_manager
import glob
import csv
import os
import time, random, threading
import argparse
import contextlib
from pympler import summary
from pympler import muppy
from pympler import tracker
from collections import deque
import io
# import seaborn as sns
# sns.set()

class multi_plot():
    def __init__ (self, path=''):
        self.path = path
        self.legends = []
        self.dfs = []
    def plot(self, df, name='', legend='', xlabel='', ylabel='',alpha=1.0, xlim=[], ylim=[]):
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        self.dfs.append(df)
        self.legends.append(legend)
        dfs = pd.concat(self.dfs, axis=1, sort=False)
        fig, ax = plt.subplots()
        dfs.plot(ax=ax,alpha=alpha,grid=True)
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        if self.legends != []:
            ax.legend(self.legends)
        if xlim != []:
            ax.set_xlim(xlim)
        if ylim != []:
            ax.set_ylim(ylim)
        plt.savefig(self.path+name)
        plt.close(fig)

class multi_subplot():
    def __init__ (self, path='',dim=11, figsize=[9,6],dpi=180 ):
        self.path = path
        self.legends = []
        self.dfs = []
        # self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None)
        self.fig = plt.figure(figsize=figsize,dpi=dpi)

        self.i=dim*10
    def plot(self, df, name='', legend=[],alpha=1.0, xlim=[], ylim=[], xlabel='', ylabel=''):
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        self.i+=1
        ax = self.fig.add_subplot(self.i)
        # dfs = pd.concat(self.dfs, axis=1, sort=False)
        df.plot(ax=ax,alpha=alpha, grid=True)
        if legend != []:
            ax.legend(legend)
        if xlim != []:
            ax.set_xlim(xlim)
        if ylim != []:
            ax.set_ylim(ylim)
        if xlabel != '':
            ax.set_xlabel(xlabel)
        if ylabel != '':
            ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.fig.savefig(self.path+name)
        # plt.close(fig)
class multi_scatterplot():
    def __init__ (self, path='', figsize=[9,6],dpi=180 ):
        self.path = path
        self.legends = []
        self.dfs = []


        # self.fig = plt.figure(figsize=figsize,dpi=dpi)
        self.scatter_fig, self.scatter_ax = plt.subplots(figsize=figsize,dpi=dpi)
        self.scatter_ax.set_xlim([-2.5, 2.5])
        self.scatter_ax.set_ylim([-2.5, 2.5])

    def scatter(self, df, x='',y='',c=None, edgecolors=None, name='', s=None, invert_x = False, x_name=None, y_name=None, marker='o', alpha=None):
        if invert_x:
            df[[x]] = -1*df[[x]]
        df.plot.scatter(x=x, y=y,c=c,ax=self.scatter_ax, s=s, grid=True, marker=marker, edgecolors=edgecolors, alpha=alpha, fontsize=12)
        if x_name != None:
            self.scatter_ax.set_xlabel(x_name,fontdict={'fontsize':15})
        if y_name != None:
            self.scatter_ax.set_ylabel(y_name,fontdict={'fontsize':15})
        self.scatter_fig.tight_layout()
        self.scatter_fig.savefig(self.path+name)

class animator():
    def __init__(self, path='', episode = 0,figsize=[9,6],dpi=180 ):
        self.path = path
        self.episode = episode
        self.fig, self.ax = plt.subplots(figsize=[9,6],dpi=180)
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.x_list = []
        self.y_list = []
        self.imgs = []
        self.goal_pos = []
        self.colors=['red','green','blue','yellow', 'cyan']
    def add_goal_pos(self,x,y):
        self.goal_pos.append([x,y])
    def add_data(self,x,y):
        print(x.shape)
        print(y.shape)
        self.x_list.append(x)
        self.y_list.append(y)
        print(len(self.x_list))
        print(len(self.y_list))
    def animate(self):
        max_length = max_idx = 0
        for i in range(len(self.x_list)):
            if len(self.x_list[i]) > max_length:
                max_length = len(self.x_list[i])
                max_idx = i
        print(max_length)
        pts_x = [[] for i in range(4)]
        pts_y = [[] for i in range(4)]
        for row_idx in range(max_length):
            for col_idx in range(4):
                if row_idx < len(self.x_list[col_idx]):
                    pts_x[col_idx].append(self.x_list[col_idx][row_idx])
                    pts_y[col_idx].append(self.y_list[col_idx][row_idx])
                    self.imgs.append([self.ax.scatter(pts_x[col_idx],
                                    pts_y[col_idx], c=self.colors[col_idx],alpha = 0.7)])
                    # for i in range(4):
                    #     self.imgs.append([self.ax.scatter(self.goal_pos[i][0],
                    #                 self.goal_pos[i][1], c='black')])

        print(len(self.imgs))
        ani1 = animation.ArtistAnimation(self.fig, self.imgs, interval=10, blit=False,
                                        repeat_delay=1000)
        print((self.path+str(self.episode)+'.mp4'))
        ani1.save(self.path+str(self.episode)+'.mp4')





class CreateReport():
    def __init__(self, path='',episode = 0, last_n=-1, new_dir=False):
        self.path = path+'/'
        self.episode = episode
        self.tr = tracker.SummaryTracker()
        self.last_n = last_n
        self.ani = animator(path=self.path, episode = self.episode,figsize=[9,6],dpi=180 )
        self.mean_rewards_plot = multi_plot(self.path)
        self.mean_rewards_subplot = multi_subplot(self.path, dim=41, figsize=[5,8],dpi=250 )
        self.new_dir = new_dir
        self.number_robots = 0
        if self.new_dir:
            self.dir_path = self.path+str(self.episode)
            if not os.path.isdir(self.dir_path):
                os.mkdir(self.dir_path)
                os.chmod(self.dir_path, 0o777)
            self.dir_path = self.dir_path+'/'
            self.robot_pos_scatter = multi_scatterplot(self.dir_path, figsize=[5,4],dpi=100)
        else:
            self.robot_pos_scatter = multi_scatterplot(self.path, figsize=[5,4],dpi=100)

    def update(self, last_n=-1, report_mem=True, test=False):
        self.last_n = last_n
        self.get_files_names(self.path+'*')
        if self.ordered_robot_files != []:
            self.analyze_robot_files()
            # try:
            # except Exception as e: print(e.args)
        # self.ani.animate()


    def get_files_names(self, path='*'):
        files_names = glob.glob(path)
        self.robot_files = [f for f in files_names if 'robot.csv' in f]
        self.robot_files.sort()
        self.ordered_robot_files = self.handel_robot_file_names()

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
        return name_list

    def analyze_robot_files(self):
        # colors=['red','green','blue','yellow', 'cyan']
        colors=[(1,0,0),(0.1,1,0.1),(0,0,1),(0.5,0.5,0), (0,0.5,0.5)]
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
                # robot_pos.append([data[['robot_x']].iloc[0][0] ,data[['robot_y']].iloc[0][0]])
                # print(robot_pos)
                for t in range(1,len(robot_names_list)):
                    f =robot_names_list[t]
                    data = pd.concat([data, pd.read_csv(f)], ignore_index=True)

            else:
                f =robot_names_list[0]
                data = pd.read_csv(f)
                # robot_pos.append([data[['robot_x']].iloc[0][0] ,data[['robot_y']].iloc[0][0]])
                    # print(robot_pos)
            if self.last_n != -1 :
                data=data[-1*self.last_n:]
                self.episode_last_n = data['episode'].iloc[0]
            df_ep = (data.loc[data['episode'] == self.episode]).copy()
            df_ep = df_ep.reset_index(drop=True)
            df_ep['index'] = df_ep.index
            df_episode_state = df_ep.groupby('episode')[['episode', 'collision', 'reach_goal', 'reach_all_goals','duration_done']].apply(lambda x: x.tail(1))
            df_episode_state.columns = ['episode_'+str(a), 'collision_'+str(a), 'reach_goal_'+str(a), 'reach_all_goals_'+str(a),'duration_done_'+str(a)]
            state_dfs.append(df_episode_state.reset_index())
            # df_ep[['robot_y']] = -1 * df_ep[['robot_y']]
            if not self.new_dir:
                for q in range(self.number_robots):
                    self.robot_pos_scatter.scatter(df_ep[['goal_{}_x'.format(q),'goal_{}_y'.format(q)]].iloc[[0]], y='goal_{}_y'.format(q),x='goal_{}_x'.format(q), c='black',name=str(self.episode)+'_robot_positions.png', invert_x=False)
                self.robot_pos_scatter.scatter(df_ep[['robot_x','robot_y']].iloc[[0]], y='robot_y',x='robot_x', c='black', name=str(self.episode)+'_robot_positions.png', invert_x=False, marker='s')
                self.robot_pos_scatter.scatter(df_ep[['robot_x','robot_y']], y='robot_x',x='robot_y', c=colors[i],name=str(self.episode)+'_robot_positions.png', invert_x=False, x_name='x (m)', y_name='y (m)')
            else:
                s = 350
                for q in range(self.number_robots):
                    self.robot_pos_scatter.scatter(df_ep[['goal_{}_x'.format(q),'goal_{}_y'.format(q)]].iloc[[0]], y='goal_{}_y'.format(q),x='goal_{}_x'.format(q),
                    s=s,name=str(self.episode)+'_robot_positions.png', invert_x=False, edgecolors='black', alpha=0.5)

                weights = df_ep[['steps']].values
                weights = weights - weights.min()
                weights = weights/weights.max()
                weights = (np.squeeze(weights) * 0.7) + 0.1
                color = np.zeros((weights.shape[-1], 4))
                color[:,:3] = color[:,:3]+colors[i]
                color[:,-1] = weights
                # print(df_ep['collision'].iloc[-1])
                s=s//4
                self.robot_pos_scatter.scatter(df_ep[['robot_x','robot_y']].iloc[[0]], y='robot_y',x='robot_x', c='black', name=str(self.episode)+'_robot_positions.png', invert_x=False, marker='s', s=s)
                self.robot_pos_scatter.scatter(df_ep[['robot_x','robot_y']], y='robot_y',x='robot_x', c=color,name=str(self.episode)+'_robot_positions.png', invert_x=False, x_name='x (m)', y_name='y (m)', s=s)
                s=int(s*2)
                if df_ep['collision'].iloc[-1] == True:
                    self.robot_pos_scatter.scatter(df_ep[['robot_x','robot_y']].iloc[[-1]], y='robot_y',x='robot_x', c='black', name=str(self.episode)+'_robot_positions.png', invert_x=False, x_name='x (m)', y_name='y (m)', marker='x', s=s)
                try:
                    idx = df_ep.index.get_loc(df_ep[df_ep['done'] == True].index[0])
                    df_ep = df_ep.iloc[0: idx + 1]
                    self.plot(df_ep[['HZ']], name+'_hz.png',xlabel='Step', ylabel='HZ', mean=True)
                    # self.plot_sns(df_ep[['HZ', 'index']], name=name+'_hz.png', x = 'index', kind="line")
                    self.plot(df_ep[['reward']], name+'_reward.png', xlabel='Step', ylabel='Reward', mean=False)
                    dist_reward = df_ep['distance_reward'].iloc[-1]
                    print(dist_reward)
                    if dist_reward >= 15:
                        loca = "upper left"
                    else:
                        loca = "best"
                    self.plot(df_ep[['distance_reward', 'ob_reward',
                    'other_robot_dist_reward', 'time_reward']], name+'_all_rewards.png', mean=False, xlabel='Step', ylabel='Reward',
                    legend=['$r^{\,d}$','$r^{\,col}$','$r^{\,o}$', '$r^{\,t}$'], loc=loca)
                    self.plot(df_ep[['action_l','action_r']], name+'_actions.png', mean=True, xlabel='Step', ylabel='Action')
                    self.plot(df_ep[['action_l']], name+'_action_l.png', mean=True, xlabel='Step', ylabel='Action Linear')
                    self.plot(df_ep[['action_r']], name+'_actions_r.png', mean=True, xlabel='Step', ylabel='Action Rotational')
                    # self.plot(df_ep[['action_l','action_r']].ewm(alpha=0.01).mean(), name+'_ewm_actions.png', xlabel='Step', ylabel='Smoothed action')
                except:
                    pass
            del(data)
        all_states_df = pd.concat(state_dfs, axis=1, sort=False)
        all_states_df = all_states_df.fillna(False)
        success_rate_col_list = ['reach_goal_0']
        # all_states_df['reach_all_goals'] = all_states_df['reach_all_goals_0'] | all_states_df['reach_all_goals_1'] | all_states_df['reach_all_goals_2'] | all_states_df['reach_all_goals_3']
        all_states_df['reach_all_goals'] = all_states_df['reach_all_goals_0']
        for q in range(1,self.number_robots):
            all_states_df['reach_all_goals'] = all_states_df['reach_all_goals'] | all_states_df['reach_all_goals_{}'.format(q)]
            success_rate_col_list.append('reach_goal_{}'.format(q))
        all_states_df['success_rate'] = all_states_df[success_rate_col_list].mean(axis=1)
        all_states_df.loc[all_states_df['reach_all_goals']==True, 'success_rate'] = 1.0
        for r_goal_str in success_rate_col_list:
            all_states_df.loc[all_states_df['reach_all_goals']==True, r_goal_str] = True
        # all_states_df.loc[all_states_df['reach_all_goals']==True, 'reach_goal_0','reach_goal_1', 'reach_goal_2', 'reach_goal_3'] = True
        # all_states_df[['reach_goal_0','reach_goal_1', 'reach_goal_2', 'reach_goal_3']]
        # print(all_states_df[['reach_all_goals_0','reach_all_goals_1', 'reach_all_goals_2', 'reach_all_goals_3', 'reach_all_goals']])
        print_list = ['success_rate']
        for q in range(self.number_robots):
            print_list.append('reach_goal_{}'.format(q))
        print(all_states_df[print_list])
        # self.bar_error_plot(all_states_df[['success_rate']], name='comparison_success_rate.png', legend=[], xlabel='', ylabel='')
        all_states_desc = all_states_df.describe()
        print(all_states_desc[['success_rate']])

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
        if self.new_dir:
            fig.savefig(self.dir_path+name)
        else:
            fig.savefig(self.path+name)
        plt.close(fig)

    def plot(self, df, name='', legend=[], xlabel='', ylabel='',alpha=1.0, mean=False, loc="best", xlim=[], ylim=[]):
        color = ['blue','red','green','magenta','brown', 'orange', 'black','goldenrod']
        light_color = ['lightblue','salmon','lightgreen','pink','rosybrown', 'wheat', 'grey','lightyellow']
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        # df[0].rolling(10).mean()
        fig, ax = plt.subplots(figsize=[5,4])
        if mean :
            df.plot(ax=ax,alpha=0.75, grid=True, color = light_color, legend=False, fontsize=12)
            (df.rolling(10, min_periods=1).mean()).plot(ax=ax, grid=True, alpha=0.9,color=color, fontsize=12)
        else:
            df.plot(ax=ax,alpha=alpha, grid=True, fontsize=12, legend=False)
        if xlabel != '':
            # plt.xlabel(xlabel)
            ax.set_xlabel(xlabel,fontdict={'fontsize':15})
        if ylabel != '':
            # plt.ylabel(ylabel)
            ax.set_ylabel(ylabel,fontdict={'fontsize':15})
        if xlim != []:
            ax.set_xlim(xlim)
        if ylim != []:
            ax.set_ylim(ylim)
        if legend != []:
            # font = font_manager.FontProperties(stretch=1000)
            # family='Times New Roman',
                                   # weight='bold',
                                   # style='normal', size=16)
            ax.legend(legend,handletextpad=0.5, loc=loc,  prop={'size': 12})
            # ax.legend(legend)
        fig.tight_layout()
        if self.new_dir:
            fig.savefig(self.dir_path+name)
        else:
            fig.savefig(self.path+name)
        plt.close(fig)

    def plot_sns(self, df, name, x = "", kind="line"):
        df_m = pd.melt(df, [x])
        sns_plot = sns.lineplot(x=x, y="value", hue="variable", data=df_m);
        sns_plot.savefig(self.dir_path+name)

    def analyze_memory(self):
        with open(self.path+'mem_diff.txt','a') as f:
            with contextlib.redirect_stdout(f):
                self.tr.print_diff()
        with open(self.path+'mem_sum.txt','a') as f:
            with contextlib.redirect_stdout(f):
                all_objects = muppy.get_objects()
                sum1 = summary.summarize(all_objects)
                summary.print_(sum1)

    def describtions_of_binary_col(self, df, col_name, condition = True, count_repeat = False):
        # if self.last_n != -1 :
        #     df_filtered[col_name] = df_filtered[col_name]>0.5
        if not count_repeat:
            df_filtered = df[['ros_time', 'episode',col_name]]
            # df_filtered = df_filtered[df_filtered[col_name]-df_filtered[col_name].shift() != 0 ]
            # df_filtered.groupby('episode')
            # idx = df_filtered.groupby(['episode'])[col_name].transform(max) == df_filtered[col_name]
            df_filtered =  df_filtered.loc[df_filtered.groupby(['episode'])[col_name].idxmax()]
            # df_filtered = df_filtered[idx]
        else:
            df_filtered = df[['ros_time', col_name]]
        df_filtered = df_filtered[df_filtered[col_name] == condition]
        describtion = df_filtered[['ros_time', col_name]].describe(include='all')
        return describtion

if __name__ == '__main__':
    import os
    path = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--path', type=str,default='None',
                            help='experiment directory path')

    parser.add_argument('-n', '--last_lines', type=int, default=-1,
                            help='last lines to read form robot files')
    parser.add_argument('-e', '--episode', type=int, default=0,
                            help='epidoe to animate')

    parser.add_argument('--memory_report', action='store_true', help='report memory')
    parser.add_argument('--test', action='store_true', help='Testing the network')
    parser.add_argument('--new_dir', action='store_true', help='create new directory for the results')


    args = parser.parse_args()

    if args.path != 'None':
        path = args.path
    n = args.last_lines
    MEM = args.memory_report
    TEST = args.test
    NEW_DIR = args.new_dir
    e = args.episode
    print(path)
    report = CreateReport(path, episode = e, new_dir=NEW_DIR)
    report.update(n, MEM,TEST)
