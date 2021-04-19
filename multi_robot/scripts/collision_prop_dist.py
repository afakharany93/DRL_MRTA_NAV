#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '12'

class CollisionPropDist():
    def __init__(self, lower_limit=-2, upper_limit=2, n=4):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.n = n

    def get_collision_coordinates(self, data):
        self.robot_pos = [data[['robot_x']].iloc[0][0] ,data[['robot_y']].iloc[0][0]]
        data1 = data[(data['done']==False)|(data['reward']-data['reward'].shift() != 0)]
        data1 = data1.groupby('episode').apply(lambda x: x.tail(1))

        # robot_pos = []
        # for q in range(number_of_robots):
        #     data1['distance_'+str(q)] = np.sqrt((data1['robot_x']-data1['goal_{}_x'.format(q)])**2 + (data1['robot_y']-data1['goal_{}_y'.format(q)])**2)
        #
        # for q in range(number_of_robots-1):
        #     data1['reached_goal'] = np.where(data1['distance_{}'.format(q)]<data1['distance_{}'.format(q+1)], 'goal_{}'.format(q), 'goal_{}'.format(q+1))
        #     data1['reached_goal_x'] = np.where(data1['distance_{}'.format(q)]<data1['distance_{}'.format(q+1)], data1['goal_{}_x'.format(q)], data1['goal_{}_x'.format(q+1)])
        #     data1['reached_goal_y'] = np.where(data1['distance_{}'.format(q)]<data1['distance_{}'.format(q+1)], data1['goal_{}_y'.format(q)], data1['goal_{}_y'.format(q+1)])
        data1 = data1[data1['ob_reward'] == -15.0]

        return data1['robot_x'].to_numpy(), data1['robot_y'].to_numpy()

    def plot_the_dist(self, X, Y, path):
        fig, ax = plt.subplots(figsize=[6,5])
        ax.set_xlim([self.lower_limit, self.upper_limit])
        ax.set_ylim([self.lower_limit, self.upper_limit])
        h = plt.hist2d(X, Y,
                       bins=(self.n,self.n), range=[[self.lower_limit, self.upper_limit],
                       [self.lower_limit, self.upper_limit]],cmap='Spectral', cmin = 1)
        plt.colorbar(h[3], ax=ax)
        ax.scatter(self.robot_pos[0], self.robot_pos[1], c='black', marker='s', s=50)
        ax.set_xlabel('x (m)',fontdict={'fontsize':15})
        ax.set_ylabel('y (m)',fontdict={'fontsize':15})
        # ax.set_xticks(list(np.arange(self.lower_limit,self.upper_limit, self.n)))
        # ax.set_yticks(list(np.arange(self.lower_limit,self.upper_limit, self.n)))
        ax.grid(True)
        plt.savefig(path)
        plt.close(fig)

    def make_the_distribution(self, data, path):
        X,Y = self.get_collision_coordinates(data)
        self.plot_the_dist(X, Y, path)
