import numpy as np
import tensorflow as tf
# tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(3)
import matplotlib
matplotlib.use('Agg')
import argparse
import pandas as pd
from datetime import datetime
import time, random, threading
import csv

import rospy
from std_srvs.srv import Empty

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

from goals_handler import GoalHandler
from turt_robot import TurtBot
from env import Env
from env_handler import EnvHandler
from multi_states import MultiStates
# from hindsight import Hindsight
import os, sys, gc

from data_dumper_reset import ResetDataDumper
from data_dumper_step import StepDataDumper
from data_dumper_action import ActionDataDumper

from train_queue import TrainQ

from models_scripts.base_class import BaseBrain
from models_scripts.Ppo_1d_conv_fc import Ppo1DConvFCDisc


from results_report import CreateReport

class SingleRobot():
    '''
    Used for interfacing a single robot (gets its state, sends action, recieves reward)
    and logging it's data
    '''
    def __init__(self, env_h, index, goal_h, hyper_params, expr_path, start_time,
    test, threads_n, HZ, SEC_PAST, MAX_BATCH, brain, CHANGE_OF_VELOCITY):
        self.env = env_h.envs[index]
        self.env_h = env_h
        self.name = self.env.robot_name
        self.index = index
        self.goalh = goal_h
        self.hyper_params = hyper_params
        self.expr_path = expr_path
        self.threads_n = threads_n
        self.requested_hz = HZ
        self.sec_past = SEC_PAST
        self.max_batch = MAX_BATCH
        self.start_time = start_time
        self.test = test
        self.brain = brain
        self.CHANGE_OF_VELOCITY = CHANGE_OF_VELOCITY

        rospy.loginfo("creating robot {}".format(self.name))
        self.steps = 0
        self.stacked_states = MultiStates(self.hyper_params['red_states'], n_robots=self.threads_n, hz=self.requested_hz, sec_past=self.sec_past)
        self.stacked_states_ = MultiStates(self.hyper_params['red_states'], n_robots=self.threads_n, hz=self.requested_hz, sec_past=self.sec_past)
        self.train_q = TrainQ(max_batch_size=self.max_batch)
        self.data_dumper = StepDataDumper(self.expr_path, name=self.name, threads=self.threads_n, h=self.goalh, start_time=self.start_time)
        self.vl = self.vr = 0.0
        self.counter=0
        self.hz = 0
        self.change_goal = False
        # self.local_n_episode = Environment.n_episode
        self.min_distance = 0
        # self.action_data_dumper = ActionDataDumper(self.expr_path, name=self.name, start_time=start_time)
        self.done = False
        self.collision = self.env.collision
        self.reach_a_goal = self.env.get_goalbox
        self.tock = 0
        self.R = 0

    def get_done_flag(self):
        collision = self.env.collision
        reach_a_goal = self.env.get_goalbox
        reach_all_goals, _ = self.env_h.check_for_ending_conditions()
        return self.done, collision, reach_a_goal, reach_all_goals

    def get_state(self):
        # robot_info = [self.local_n_episode, self.steps]
        # self.action_data_dumper.write_csv_row(category='acting_start', robot_info=robot_info, stop_signal=Environment.stop_signal)
        self.s, done = self.env.generate_state()
        self.stacked_s = self.stacked_states.insert(self.s)
        # print(self.stacked_s[1].shape)
        self.min_distance = np.min(self.s[2])
        # self.action_data_dumper.write_csv_row(category='stacked_states', robot_info=robot_info, stop_signal=Environment.stop_signal)
        return self.stacked_s

    def send_action(self, a, v):
        self.v = v[0]
        # print(self.v)
        self.a = a
        if self.CHANGE_OF_VELOCITY:
            self.vl = min(max(a[0]+self.vl, -1), 1)
            self.vr = min(max(a[1]+self.vr, -1), 1)
        else:
            self.vl = a[0]
            self.vr = a[1]
        self.env.take_action((self.vl,self.vr))
        self.hz = 1/abs(rospy.get_time() - self.tock)
        self.hz = np.clip(self.hz, 0, self.requested_hz)
        self.tock = rospy.get_time()
        # self.action_data_dumper.write_csv_row(category='took_the_action', robot_info=robot_info, stop_signal=Environment.stop_signal)

    def get_reward_and_next_state(self):
        self.s_, self.r, self.done = self.env.generate_reward_and_next_state()
        # self.action_data_dumper.write_csv_row(category='recieved_the_reward', robot_info=robot_info, stop_signal=Environment.stop_signal)

        self.stacked_s_ = self.stacked_states_.insert(self.s_)

        self.min_distance = np.min(self.s_[2])
        self.R += self.r
        self.steps += 1
        if not self.test:
            self.train_q.train_push(self.stacked_s, self.a, self.r, self.stacked_s_, self.v, self.done)

        _, self.collision, self.reach_a_goal, self.reach_all_goals = self.get_done_flag()

        if self.reach_a_goal:
            rospy.loginfo("{} has reached a goal !".format(self.name))

        return self.r, self.stacked_s_

    def reset(self):
        rospy.loginfo("Resetting by {} : reach a goal = {}, collision = {}, HZ = {}".format(self.name, self.reach_a_goal, self.env.collision, self.hz))
        self.tock = 0
        self.stacked_states.reset()
        self.stacked_states_.reset()
        self.R = 0
        self.done = self.collision = self.reach_a_goal = self.reach_all_goals = False
        if not self.test:
            self.train_q.push_to_other_list(self.brain.train_push_list, self.env.name)

    def report_step(self, duration_done, episode_n):
        network_op = [self.vl, self.vr, self.v, self.done] #'action_l', 'action_r', 'value', 'done'
        episode_state = [self.collision, self.reach_a_goal, self.reach_all_goals, duration_done, self.hz]
        robot_info = [episode_n, self.steps, self.name, self.env.robot_state.odom, self.env.get_logging_data(), self.min_distance]
        self.data_dumper.write_csv_row(network_op, episode_state, robot_info)


class Environment():
    def __init__(self, env_h, goal_h, threads_n, test, episode_duration,
    start_time, n_episodes_respawn_goals, thread_delay, run_time, hyper_params, expr_path,
    requested_hz, SEC_PAST, MAX_BATCH, brain, CHANGE_OF_VELOCITY, multi_env, **kwargs):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.env_h = env_h
        self.goal_h = goal_h
        self.test = test
        self.episode_duration = episode_duration
        self.start_time = start_time
        self.n_robots = threads_n
        self.n_episodes_respawn_goals = n_episodes_respawn_goals
        self.thread_delay = thread_delay
        self.runtime = run_time
        self.hyper_params = hyper_params
        self.expr_path = expr_path
        self.requested_hz = requested_hz
        self.SEC_PAST = SEC_PAST
        self.MAX_BATCH = MAX_BATCH
        self.brain = brain
        self.CHANGE_OF_VELOCITY = CHANGE_OF_VELOCITY
        self.multi_env = multi_env
        if 'main_expr_path' in kwargs:
            self.main_expr_path = kwargs.get("main_expr_path")


        self.reset_dumper = ResetDataDumper(self.expr_path, self.start_time)

        self.change_goal = False
        self.episode_n = 0
        self.last_episode_to_change_goals = 0
        self.steps = 0
        self.elapsed_time = 0
        self.robots = [
        SingleRobot(env_h=self.env_h, index=i, goal_h=self.goal_h, hyper_params = self.hyper_params,
        expr_path=self.expr_path, start_time=self.start_time, test=self.test, threads_n=self.n_robots,
        HZ=self.requested_hz, SEC_PAST=self.SEC_PAST, MAX_BATCH=self.MAX_BATCH, brain=self.brain,
        CHANGE_OF_VELOCITY=self.CHANGE_OF_VELOCITY) for i in range(self.n_robots)]
        self.tock = self.tick = rospy.get_time()
        self.reach_all_goals = self.collision = self.duration_done = False
        self.pauseSim()
        # self.done_robot_list = [robot.get_done_flag()[0] for i, robot in enumerate(self.robots)]

    def collect_states(self):
        s0 = []
        s1 = []
        s2 = []
        s3 = []
        done = []
        # (np.array(self.l_current_robot_rel_dist), np.array(self.l_other_robots_rel_dist), np.array(self.l_laser), s[3])
        for i, robot in enumerate(self.robots):
            robot_done = robot.get_done_flag()[0]
            if not robot_done:
                # self.done_robot_list[i] = robot_done
                robot_state = robot.get_state()
                s0.append(robot_state[0])
                s1.append(robot_state[1])
                s2.append(robot_state[2])
                s3.append(robot_state[3])
                done.append(robot_done)
        s = [s0, s1, s2, s3]
        return s, done

    def perform_prediction(self, s):
        s0, s1, s2, s3 = s
        s0 = np.array(s0)
        s1 = np.array(s1)
        s2 = np.array(s2)
        s3 = np.array(s3).reshape((-1,3))
        # print(s3.shape)
        should_sample = False
        if not self.test:
            should_sample = True
        a, v = self.brain.predict([s0, s1, s2, s3], should_sample)
        # print(a)
        return a, v

    def send_actions(self, actions, v):
        i = 0
        for robot in self.robots:
            robot_done = robot.get_done_flag()[0]
            if not robot_done:
                a, val = actions[i], v[i]
                i = i+1
            else:
                a, val = np.array([0,0]), np.array([0])
            robot.send_action(a, val)

    def collect_next_states_and_rewards(self):
        for robot in self.robots:
            robot_done = robot.get_done_flag()[0]
            if not robot_done:
                robot.get_reward_and_next_state()

    def check_terminal_conditions(self):
        reach_all_goals, collision = self.env_h.check_for_ending_conditions()
        # self.tock =
        self.duration = rospy.get_time() - self.tick
        duration_done = self.duration >= self.episode_duration
        return reach_all_goals, collision, duration_done

    def report_step(self, duration_done, episode_n):
        for robot in self.robots:
            # robot_done = robot.get_done_flag()[0]
            # if not robot_done:
            robot.report_step(duration_done, episode_n)

    def reset(self):
        # network_op = [0, 0, 0, self.done] #'action_l', 'action_r', 'value', 'done'
        episode_state = [self.collision, 0, self.reach_all_goals, self.duration_done, 0]
        robot_info = [self.episode_n, self.steps, '']
        # self.data_dumper.write_csv_row(network_op, episode_state, robot_info, Environment.stop_signal)
        self.reset_dumper.write_csv_row(episode_state, robot_info)

        self.elapsed_time = time.time() - self.start_time
        self.total_time = time.strftime("%H:%M:%S", time.gmtime(self.elapsed_time))
        rospy.loginfo("Resetting for episode {}, after total duration of {} for: collision = {}, reach_all_goals = {}, duration done = {}, Episode time = {}".format( self.episode_n, self.total_time,
                                                                                                                                                    self.collision, self.reach_all_goals,
                                                                                                                                                    self.duration_done, self.duration))
        self.env_h.reset()
        for robot in self.robots:
            robot.reset()
        self.reach_all_goals = self.collision = self.duration_done = False

    def runEpisode(self):
        self.unpauseSim()
        while rospy.get_time() == 0:
            pass
        terminate = False
        self.tock = self.tick = rospy.get_time()
        if self.episode_n - self.last_episode_to_change_goals >= self.n_episodes_respawn_goals:
            self.change_goal = True
        if self.change_goal:
            self.last_episode_to_change_goals = self.episode_n
            self.change_goal = False
            self.goal_h.respawn_all_goals()
        while not (terminate or rospy.is_shutdown()):
            self.tock = rospy.get_time()
            s, done = self.collect_states()
            a, v = self.perform_prediction(s)
            self.send_actions(a, v)
            while abs(rospy.get_time() - self.tock) < self.thread_delay:
                pass
            self.collect_next_states_and_rewards()
            self.reach_all_goals, self.collision, self.duration_done = self.check_terminal_conditions()
            self.report_step(self.duration_done, self.episode_n)
            if self.reach_all_goals or self.collision or self.duration_done:
                terminate = True
                if self.reach_all_goals :
                    self.change_goal = True

        self.reset()
        self.episode_n += 1
        self.pauseSim()

    def run(self):
        while self.elapsed_time < self.runtime:
            self.runEpisode()
            if not self.test:
                if self.multi_env:
                    opt = self.brain.me_save_data()
                    if opt:
                        # rospy.loginfo('saved collected data')
                        ld_mdl = False
                        while not ld_mdl:
                            ld_mdl = self.brain.load_latest_model(self.main_expr_path)
                else:
                    opt = self.brain.optimize()
            self.elapsed_time = time.time() - self.start_time


    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.loginfo ("/gazebo/pause_physics service call failed")

    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.loginfo ("/gazebo/unpause_physics service call failed")
