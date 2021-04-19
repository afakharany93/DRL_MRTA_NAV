#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import glob
import argparse
# sns.set()

class ActionTimeReport:
    def __init__ (self, path):
        self.expr_path = path

        self.file_names = glob.glob(self.expr_path+'/*env_0_robot_actions.csv')
        self.file_names.sort()

    def get_comboned_pd(self, file_names):
        data = pd.read_csv(self.file_names[0])
        if len(self.file_names) > 1:
            for i in range(1, len(self.file_names)):
                # pd.concat([s1, s2], ignore_index=True)
                temp_pd = pd.read_csv(self.file_names[i])
                data = pd.concat([data, temp_pd], ignore_index=True)
        return data

    def update (self):
        if self.file_names != []:
            time_data = self.get_comboned_pd(self.file_names)
            time_data_1 = time_data[time_data['category']!='episode_start']
            time_data_1 = time_data_1[time_data_1['duration']>=0.0]
            time_data_1 = time_data_1[time_data_1['duration']<0.5]

            with sns.axes_style("darkgrid"):
                sns_plot = sns.relplot(x="steps", y="duration",
                            hue="category",
                            facet_kws={ 'sharex':False},
                            kind="line", legend="full", data=time_data_1, height=5, aspect=3,
                            alpha = 0.7);
                sns_plot.savefig(self.expr_path+"/action_time.png")
            names = list(time_data_1.category.unique())
            for i in range(len(names)):
                with sns.axes_style("darkgrid"):
                    sns_plot = sns.relplot(x="steps", y="duration",
                                hue="category",
                                facet_kws={ 'sharex':False},
                                kind="line", legend="full", data=time_data_1[time_data_1['category']==names[i]], height=5, aspect=3,
                                alpha = 0.7);
                    sns_plot.savefig(self.expr_path+"/action_time_{}.png".format(names[i]))


if __name__ == '__main__':
    import os
    path = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--path', type=str,default='None',
                            help='experiment directory path')


    args = parser.parse_args()

    if args.path != 'None':
        path = args.path
    print(path)
    report = ActionTimeReport(path)
    report.update()
