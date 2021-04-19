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
from pympler import summary
from pympler import muppy
from pympler import tracker
from collections import deque
import io
# import seaborn as sns

from reching_goal_prop_dist import RechingGoalPropDist
from collision_prop_dist import CollisionPropDist
from action_time_report import ActionTimeReport

class multi_plot():
    def __init__ (self, path=''):
        self.path = path
        self.legends = []
        self.dfs = []
        self.current_color = 0
    def plot(self, df, name='', legend='', xlabel='', ylabel='',alpha=1.0, xlim=[], ylim=[], mean=True, rolling_number=10):
        color = ['blue','red','goldenrod','green','magenta','brown', 'orange', 'black']
        light_color = ['lightblue','salmon','gold','lightgreen','pink','rosybrown', 'wheat', 'grey']
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        self.dfs.append(df)
        self.legends.append(legend)
        dfs = pd.concat(self.dfs, axis=1, sort=False)
        fig, ax = plt.subplots()
        # dfs.plot(ax=ax,alpha=alpha,grid=True)
        if mean :
            dfs.plot(ax=ax,alpha=0.75, grid=True, color = light_color, legend=False)
            (dfs.rolling(rolling_number, min_periods=1).mean()).plot(ax=ax, grid=True, alpha=0.9,color=color)
            self.current_color += 1
        else:
            dfs.plot(ax=ax,alpha=alpha, grid=True)
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
        self.current_color = 0
    def plot(self, df, name='', legend=[],alpha=1.0, xlim=[], ylim=[], xlabel='', ylabel='', mean=True, rolling_number=10):
        color = ['blue','red','green','magenta','brown', 'orange', 'black','goldenrod']
        light_color = ['lightblue','salmon','lightgreen','pink','rosybrown', 'wheat', 'grey','gold']
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        self.i+=1
        ax = self.fig.add_subplot(self.i)
        # dfs = pd.concat(self.dfs, axis=1, sort=False)
        # df.plot(ax=ax,alpha=alpha, grid=True)
        if mean :
            df.plot(ax=ax,alpha=0.75, grid=True, color = light_color, legend=False)
            (df.rolling(rolling_number, min_periods=1).mean()).plot(ax=ax, grid=True, alpha=0.9,color=color)
        else:
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
        self.scatter_ax.set_xlim([-3, 3])
        self.scatter_ax.set_ylim([-3, 3])

    def scatter(self, df, x='',y='',c='red',name='', invert_x = False):
        if invert_x:
            df[[x]] = -1*df[[x]]
        df.plot.scatter(x=x, y=y,c=c,ax=self.scatter_ax, grid=True)
        self.scatter_fig.tight_layout()
        self.scatter_fig.savefig(self.path+name)


class CreateReport():
    def __init__(self, path='', last_n=-1, new_dir=False, action_timer=True):
        self.path = path+'/'
        self.tr = tracker.SummaryTracker()
        self.last_n = last_n
        self.number_robots = 0
        self.action_timer = action_timer

        self.mean_rewards_plot = multi_plot(self.path)
        self.mean_rewards_subplot = multi_subplot(self.path, dim=41, figsize=[5,8],dpi=250 )
        self.robot_pos_scatter = multi_scatterplot(self.path, figsize=[9,6],dpi=180)

        self.distribution_drawer = RechingGoalPropDist(lower_limit=-3, upper_limit=3, n=6)
        self.collision_distribution_drawer = CollisionPropDist(lower_limit=-3, upper_limit=3, n=6)
        if self.action_timer:
            self.action_timer = ActionTimeReport(path)

    def update(self, last_n=-1, report_mem=True, test=False):
        self.last_n = last_n
        self.get_files_names(self.path+'*')
        if not test:
            try:
                self.analyze_update_files()
            except Exception as e: print(e)
        if self.reset_files != [] and self.last_n==-1:
            try:
                self.analyze_reset_files()
            except Exception as e: print(e)
        if self.ordered_robot_files != []:
            # try:
            self.analyze_robot_files()
            # except Exception as e: print(e.args)
        if self.reset_files != [] and self.last_n!=-1:
            try:
                self.analyze_reset_files()
            except Exception as e: print(e)
        if report_mem:
            try:
                self.analyze_memory()
            except Exception as e: print(e)
        if self.action_timer:
            self.action_timer.update()

    def __del__(self):
        self.report_handle.close()


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
        [os.remove(f) for f in files_names if 'report.csv' in f]
        self.report_handle = open(self.path+'report.csv', 'a')
        self.report_handle.flush()
        self.writer = csv.writer(self.report_handle)

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

    def analyze_reset_files(self):
        for f in self.reset_files :
            name = 'reset'
            self.writer.writerow([name])
            self.report_handle.flush()
            data = pd.read_csv(f)
            data.steps = data.steps.diff()
            if self.last_n != -1:
                data = data[data['episode']>= self.episode_last_n]
            self.plot(data[['reach_all_goals']].astype('float32', errors='ignore'), name+'_reach_all_goals.png', xlabel='Episode', ylabel='Reach all goals')
            self.plot(data[['collision']].astype('float32', errors='ignore'), name+'_collisions.png', xlabel='Episode', ylabel='Collisions')
            self.plot(data[['duration_done']].astype('float32', errors='ignore'), name+'_duration_done.png', xlabel='Episode', ylabel='Duration exhausted')
            self.plot(data[['ros_time']], name+'_Episode_time.png',legend=['Episode time'], mean=True, xlabel='Episode', ylabel='Episode time')
            data_described = data.describe()
            data_described[['steps']].to_csv(self.path+'report.csv', mode='a')
            columns = [ 'collision', 'reach_all_goals', 'duration_done']
            dfs = []
            for c in columns:
                dfs.append(self.describtions_of_binary_col(data, col_name=c, condition = True, count_repeat = True))
            bin_dfs = pd.concat(dfs, axis=1, sort=False)
            print(bin_dfs)
            bin_dfs.to_csv(self.path+'report.csv', mode='a')
            self.report_handle.flush()
            del(data, data_described, bin_dfs)

    def analyze_update_files(self):
        for f in self.brain_files :
            with open(f, 'r') as infile:
                lines_n = len(infile.readlines(1000))
            if  lines_n > 1 :
                name = 'updates'
                self.writer.writerow([name])
                self.report_handle.flush()
                data = pd.read_csv(f)
                # data['relative_time'] = data.absolute_time.diff()
                data['cumsum_actual_batch_size'] = data.actual_batch_size.cumsum()
                data_described = data.describe()
                data_described.to_csv(self.path+'report.csv', mode='a')
                self.report_handle.flush()
                self.plot(data[['loss']], name+'_loss.png', xlabel='Update', ylabel='Loss', mean=True, rolling_number=20)
                self.plot(data[['policy_loss']], name+'_policy_loss.png', xlabel='Update', ylabel='Policy Loss', mean=True, rolling_number=20)
                self.plot(data[['value_loss']], name+'_value_loss.png', xlabel='Update', ylabel='Value Loss', mean=True, rolling_number=20)
                self.plot(data[['actual_batch_size']], name+'_batch_size.png', xlabel='Update', ylabel='Batch size', mean=True)
                del(data, data_described)

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
            # print(name)
            # print(name[8])
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

            self.distribution_drawer.make_the_distribution(data, len(self.ordered_robot_files), self.path+name+'_distribution.png')
            self.collision_distribution_drawer.make_the_distribution(data, self.path+name+'_collision_distribution.png')
            elapsed_time = data["absolute_time"].iloc[-1]
            self.writer.writerow([name, data["absolute_time"].iloc[-1]])
            self.report_handle.flush()
            data['reward_cumsum'] = data[(data['done']==False)|(data['reward']-data['reward'].shift() != 0)].groupby('episode')['reward'].cumsum()
            #comparison metrics
            data['act_speed'] = np.sqrt(data['robot_x_dot']**2 + data['robot_y_dot']**2)
            data['act_speed'] = data.groupby('episode')['act_speed'].transform('mean')/0.26
            init_dist_str_list = []
            for q in range(self.number_robots):
                init_dist_str_list.append('init_distance_'+str(q))
                data['init_distance_'+str(q)] = np.sqrt((robot_pos[a][0]-data['goal_{}_x'.format(q)])**2 + (robot_pos[a][1]-data['goal_{}_y'.format(q)])**2)
            data['min_init_distance'] = data[init_dist_str_list].min(axis=1)
            data['min_curnt_distance'] = np.sqrt((data['robot_x'].shift()-data['robot_x'])**2 + (data['robot_y'].shift()-data['robot_y'])**2)
            data['min_curnt_distance'] = data.groupby('episode')['min_curnt_distance'].transform('sum')
            data['min_curnt_distance'] = np.where(data['min_curnt_distance']<data['min_init_distance'], data['min_init_distance'], data['min_curnt_distance'])
            df_episode_state = data.groupby('episode')[['episode', 'collision', 'reach_goal', 'reach_all_goals','duration_done']].apply(lambda x: x.tail(1))
            df_episode_state.columns = ['episode_'+str(a), 'collision_'+str(a), 'reach_goal_'+str(a), 'reach_all_goals_'+str(a),'duration_done_'+str(a)]
            state_dfs.append(df_episode_state.reset_index())
            # print(df_episode_state.reset_index())
            #save cimparison metrics
            tmp_df['episode'] = data[(data['reach_goal']==True)&(data['reach_goal']-data['reach_goal'].shift() != 0)]['episode']
            tmp_df['speed'] = data[(data['reach_goal']==True)&(data['reach_goal']-data['reach_goal'].shift() != 0)]['act_speed']
            tmp_df['min_init_distance'] = data[(data['reach_goal']==True)&(data['reach_goal']-data['reach_goal'].shift() != 0)]['min_init_distance']
            tmp_df['min_curnt_distance'] = data[(data['reach_goal']==True)&(data['reach_goal']-data['reach_goal'].shift() != 0)]['min_curnt_distance']
            tmp_df['min_time'] = tmp_df['min_init_distance']/0.26
            tmp_df['act_time'] = data[(data['reach_goal']==True)&(data['reach_goal']-data['reach_goal'].shift() != 0)]['ros_time']
            tmp_df['act_time'] = np.where(tmp_df['act_time']<tmp_df['min_time'],tmp_df['min_time'], tmp_df['act_time'])

            general_df = pd.concat([general_df,tmp_df], ignore_index=True)
            reward_mean = data[(data['done']==False)|(data['reward']-data['reward'].shift() != 0)].groupby('episode')['reward'].mean()
            all_robots_rewards.append(reward_mean.copy())
            data_described = data.describe()
            data_described.to_csv(self.path+'report.csv', mode='a')
            self.report_handle.flush()
            columns = ['done', 'collision', 'reach_goal', 'reach_all_goals', 'duration_done']
            dfs = []
            for c in columns:
                dfs.append(self.describtions_of_binary_col(data, col_name=c, condition = True, count_repeat = False))
            bin_dfs = pd.concat(dfs, axis=1, sort=False)
            bin_dfs.to_csv(self.path+'report.csv', mode='a')
            self.report_handle.flush()
            self.plot(reward_mean, name+'_reward_mean.png', mean=True,xlabel='Episode', ylabel='Reward')
            # self.plot( pd.concat([reward_mean, reward_mean.ewm(alpha=0.05).mean()], axis=1, sort=False), name+'_reward_means.png', legend=['mean reward','smoothed mean reward'], xlabel='Episode', ylabel='Reward')
            self.mean_rewards_plot.plot( df=reward_mean.ewm(alpha=0.05).mean(), name='rewards_mean_allrobot.png', legend='Robot '+robot_num, xlabel='Episode', ylabel='Reward',alpha=0.9, ylim=[], mean=True)
            self.mean_rewards_subplot.plot(df=reward_mean.ewm(alpha=0.05).mean(), name='rewards_mean_allrobot_subplot.png', legend=['Robot '+robot_num], ylim=[], xlabel='Episode', ylabel='Mean Reward', mean=True)
            # self.plot(data[['reward_cumsum']], name+'_reward_cumsum.png', xlabel='Step', ylabel='Reward cumulitive sum')
            self.plot(data[['HZ']], name+'_hz.png', mean=True, xlabel='Step', ylabel='HZ', rolling_number=100)
            # self.plot(data[['HZ']].ewm(alpha=0.001).mean(), name+'_ewm_hz.png', xlabel='Step', ylabel='Smoothed HZ')
            # self.plot(data[['reward_cumsum']].ewm(alpha=0.001).mean(), name+'_ewm_reward_cumsum.png', xlabel='Step', ylabel='Smoothed reward cumulative sum')
            try:
                self.plot(data[['distance_reward', 'ob_reward','other_robot_dist_reward', 'time_reward','w_reward', 'v_reward']], name+'_all_rewards.png',xlabel='Step', ylabel='Reward', mean=True)
                self.plot(data[['distance_reward', 'ob_reward',
                 'other_robot_dist_reward', 'time_reward', 'w_reward', 'v_reward']].ewm(alpha=0.001).mean(), name+'_ewm_all_rewards.png',xlabel='Step', ylabel='Sommthed reward')
                self.plot(data[['action_l','action_r']].ewm(alpha=0.001).mean(), name+'_ewm_actions.png', xlabel='Step', ylabel='Smoothed action')
                self.plot(data[['action_l','action_r']], name+'_actions.png', mean=True,xlabel='Step', ylabel='Action')
                self.plot(data[(data['reach_goal']==True)|(data['reward']-data['reward'].shift() != 0)][['reach_goal']].astype('float32', errors='ignore'), name+'_reach_goal.png', xlabel='Step', ylabel='Reach a goal')
                self.plot(data[(data['collision']==True)|(data['reward']-data['reward'].shift() != 0)][['collision']].astype('float32', errors='ignore'), name+'_collision.png',xlabel='Step', ylabel='Collision')
                self.plot(data[(data['duration_done']==True)|(data['reward']-data['reward'].shift() != 0)][['duration_done']].astype('float32', errors='ignore'), name+'_duration_done.png', xlabel='Step', ylabel='Duration exhausted')
            except:
                pass
            self.robot_pos_scatter.scatter(data[['robot_x','robot_y']], y='robot_y',x='robot_x', c=colors[i],name='robot_positions.png', invert_x=False)
            del(data, data_described, bin_dfs, dfs)
        all_rewards_df = pd.concat(all_robots_rewards, axis=1, sort=False, ignore_index=True)
        # self.plot((all_rewards_df.mean(axis=1)).ewm(alpha=0.05).mean(),'reward_combined_mean.png',xlabel='Episode', ylabel='Reward')
        self.plot(all_rewards_df.mean(axis=1),'reward_combined_mean.png', mean=True,xlabel='Episode', ylabel='Reward', rolling_number=20)
        general_df['extra_time'] = general_df['act_time'] - general_df['min_time']
        general_df['extra_distance'] = general_df['min_curnt_distance'] - general_df['min_init_distance']
        general_df['temporal_efficiency'] = general_df['min_time']/general_df['act_time']
        general_df['spacial_efficiency'] = general_df['min_init_distance']/general_df['min_curnt_distance']
        self.bar_error_plot(general_df[['temporal_efficiency', 'spacial_efficiency']]
                            ,name='comparison_data_efficiency.png', legend=[], xlabel='', ylabel='')
        self.bar_error_plot(general_df[['extra_time', 'extra_distance']]
                            ,name='comparison_data_extra.png', legend=[], xlabel='', ylabel='')
        general_df.to_csv(self.path+'comp_data.csv', mode='w')
        self.writer.writerow(['Metrics'])
        self.report_handle.flush()
        general_df = general_df.describe()
        general_df.to_csv(self.path+'report.csv', mode='a')
        self.report_handle.flush()
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
        print(all_states_df[success_rate_col_list+['episode_0', 'success_rate']].ix[all_states_df[['success_rate']].idxmax()])
        # print(all_states_df[['reach_goal_0','reach_goal_1', 'reach_goal_2', 'reach_goal_3','success_rate']])
        self.bar_error_plot(all_states_df[['success_rate']], name='comparison_success_rate.png', legend=[], xlabel='', ylabel='')
        # self.plot(all_states_df[['success_rate']], 'sucess_rate_per_episode.png', xlabel='Episode', ylabel='Success rate')
        all_states_desc = all_states_df.describe()
        print(all_states_desc[['success_rate']])
        all_states_desc[['success_rate']].to_csv(self.path+'report.csv', mode='a')
        self.report_handle.flush()
        # all_states_df[['episode_0']].apply(lambda x: x.index if isinstance(x, bool),axis=1)
        # print(all_states_df[['episode_0', 'success_rate']])
        self.scatter(all_states_df.reset_index(), s=15,x='index',y='success_rate',c='red',name='sucess_rate_per_episode.png',  xlabel='Episode', ylabel='Success rate')
        # except Exception as e: print(e.args)
        del(general_df)
        print('elapsed time : ', elapsed_time)

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

    def plot(self, df, name='', legend=[], xlabel='', ylabel='',alpha=1.0, mean=False, rolling_number=10):
        color = ['blue','red','green','magenta','brown', 'orange', 'black','goldenrod']
        light_color = ['lightblue','salmon','lightgreen','pink','rosybrown', 'wheat', 'grey','lightyellow']
        df.rename(columns=lambda x: (str(x).replace('_', ' ')).capitalize(), inplace=True)
        # df[0].rolling(10).mean()
        fig, ax = plt.subplots()
        if mean :
            df.plot(ax=ax,alpha=0.75, grid=True, color = light_color, legend=False)
            (df.rolling(rolling_number, min_periods=1).mean()).plot(ax=ax, grid=True, alpha=0.9,color=color)
        else:
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

    def scatter(self, df, x='',y='',c='red',name='', legend=[], xlabel='', ylabel='', s=None):
        fig, ax = plt.subplots()
        df.plot.scatter(x=x, y=y,c=c,s=s,ax=ax, grid=True)
        if xlabel != '':
            plt.xlabel(xlabel)
        if ylabel != '':
            plt.ylabel(ylabel)
        if legend != []:
            ax.legend(legend)
        fig.tight_layout()
        fig.savefig(self.path+name)
        plt.close(fig)


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
    print(path)
    report = CreateReport(path)
    report.update(n, MEM,TEST)
