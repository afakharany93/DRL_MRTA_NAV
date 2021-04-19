#!/usr/bin/env python3

'''
Built upon work done by:
1)
    https://github.com/Hyeokreal/Actor-Critic-Continuous-Keras/blob/master/a2c_continuous.py
2)
    https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py
3)
    https://gist.github.com/ByungSunBae/563a0d554fa4657a5adefb1a9c985626
4)
    https://github.com/jaromiru/AI-blog/blob/master/CartPole-A3C.py
5)
    https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
6)
    https://github.com/clwainwright/proximal_policy_optimization/blob/master/src/ppo.py
7)
    https://github.com/uidilr/ppo_tf/blob/master/ppo.py
8)
    https://github.com/google-research/batch-ppo/blob/master/agents/algorithms/ppo/ppo.py
9)
    https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py
10)
    https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb
'''

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
import os, sys, gc
from high_level_environment import Environment

from data_dumper_reset import ResetDataDumper
from data_dumper_step import StepDataDumper
from data_dumper_action import ActionDataDumper

from train_queue import TrainQ

from models_scripts.base_class import BaseBrain

from models_scripts.Ppo_1d_conv_fc import Ppo1DConvFCDisc


from results_report import CreateReport

# # seed = 1432
# seed = 730
# random.seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)

dirPath0 = os.path.dirname(os.path.realpath(__file__))
# dirPath = dirPath0.replace('multi_robot/scripts', 'multi_robot/models')
expr_path = dirPath0.replace('multi_robot/scripts', 'multi_robot/experiments')

# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#-- CLI args
parser = argparse.ArgumentParser()
parser.add_argument('-rt', '--run_time', type=float, default=1.0,
                    help='Total script run time in hours')
parser.add_argument('-th', '--threads', type=int, default=3,
                    help='number of threads')
parser.add_argument('-dur', '--episode_duration', type=int, default=120,
                    help='max duration of a single episode in seconds')
parser.add_argument('-hz', '--HZ', type=float, default=30,
                    help='number of times the network is called per second')
parser.add_argument('-g', '--gamma', type=float, default=0.95,
                    help='reward discount factor')

parser.add_argument('-bs', '--batch_size', type=int, default=2084,
                    help='batch size')
parser.add_argument('-mbs', '--max_batch_size', type=int, default=4096,
                    help='max batch size')
parser.add_argument('-bsi', '--batch_size_inc', type=float, default=0.1,
                    help='batch size increase ration each optimiziation')

parser.add_argument('-her_p', '--her_prop', type=float, default=0.0,
                    help='start propability of her')
parser.add_argument('-her_d', '--her_dec_rate', type=float, default=0.0,
                    help='her probability decrease rate')

parser.add_argument('-rs', '--seed', type=int, default=730,
                    help='random seed')

parser.add_argument('-ep', '--epochs', type=int, default=1,
                    help='number of epochs')

parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,
                    help='starting learning rate')
parser.add_argument('-lrd', '--learning_rate_decay', type=float, default=0.96,
                    help='learning rate decay rate')
parser.add_argument('-lrds', '--learning_rate_decay_steps', type=int, default=10000,
                    help='learning rate decay steps')

parser.add_argument('-s_past', '--seconds_past_for_stacking', type=float, default=3,
                    help='number of seconds to stack for states')

parser.add_argument('-dis_size', '--discrete_size', type=int, default=17,
                    help='size of discrete output of the NN')
parser.add_argument('-e', '--epsilon', type=float, default=0.2,
                    help='epsilon for ppo')
parser.add_argument('-ent_f', '--entropy_factor', type=float, default=0.01,
                    help='entropy factor for ppo')
parser.add_argument('-v_f', '--value_factor', type=float, default=0.5,
                    help='value loss factor for ppo')
parser.add_argument('--ret_norm', action='store_true', help='normalize returns')
parser.add_argument('--red_state', action='store_true', help='reduced states')

parser.add_argument('--ch_vel', action='store_true', help='output of network is change in velocity')

parser.add_argument('--vis', action='store_true', help='visualize the goals in gazebo')
parser.add_argument('-erg', '--episodes_respawn_goal', type=int, default=100000000,
                    help='number of episodes that a goal is respawned')

parser.add_argument('--debug',  action='store_true', help='Print and log debugging data')
parser.add_argument('-mp','--model_path', type=str, default='None',
                        help='Path of pretrained .h5 model file')
parser.add_argument('-mw','--model_weights', type=str, default='None', help='Path of pretrained .h5 model weights file')
parser.add_argument('--test', action='store_true', help='Testing the network')

parser.add_argument('-c','--comment', type=str, default='None',
                        help='any comment for file')

parser.add_argument('--gae', action='store_true', help='use GAE advanatage')
parser.add_argument('-mini_bs', '--mini_batch_size', type=int, default=2084,
                    help='mini batch size')

args = parser.parse_args()


RUN_TIME = args.run_time*60*60
THREADS = args.threads

EPISODE_DURATION = args.episode_duration #secs
HZ = args.HZ
THREAD_DELAY = 1/HZ

SEC_PAST = args.seconds_past_for_stacking

hyper_params = {
'gamma':args.gamma,
'lr':args.learning_rate,
'lrd':args.learning_rate_decay,
'lrd_steps':args.learning_rate_decay_steps,
'dis_size':args.discrete_size,
'epsilon':args.epsilon,
'entropy_factor':args.entropy_factor,
'value_factor':args.value_factor,
'seconds_past':SEC_PAST,
'norm_returns':args.ret_norm,
'red_states':args.red_state,
'model_weights':args.model_weights,
'epochs':args.epochs,
'use_gae': args.gae,
'mini_batch_size': args.mini_batch_size
                }

MIN_BATCH = args.batch_size
MAX_BATCH = args.max_batch_size
BATCH_RATIO = 1+args.batch_size_inc

CHANGE_OF_VELOCITY = args.ch_vel
VIS_GOALS = args.vis
EPISODES_TO_RESPAWN_GOAL = args.episodes_respawn_goal

MODEL_NAME = 'Ppo_1d_conv_fc'
MODEL_PATH = args.model_path

COMMENT = args.comment

DEBUG = args.debug
TEST = args.test

HER_P = args.her_prop
HER_D = args.her_dec_rate

tf.config.set_soft_device_placement(True)

Brain = Ppo1DConvFCDisc

if DEBUG:
    node_log_level = rospy.DEBUG
else:
    node_log_level = rospy.INFO

now = datetime.now() # current date and time
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
expr_time = now.strftime("%d-%m_%H-%M-%S")
if TEST:
    expr_id = 'expr_'+expr_time+'_'+MODEL_NAME+'_test'
else:
    expr_id = 'expr_'+expr_time+'_'+MODEL_NAME+'_train'
expr_path = expr_path+'/'+expr_id

os.mkdir(expr_path)


start_time = time.time()

brain = Brain(threads=THREADS, min_batch = MIN_BATCH, batch_ratio=BATCH_RATIO,
    max_batch=MAX_BATCH, hyper_params = hyper_params, model_path=MODEL_PATH,
    expr_path = expr_path, start_time = start_time, hz=HZ)    # brain is global

reset_dumper = ResetDataDumper(expr_path, start_time)

if __name__ == '__main__':
    # try:
    # initializing node
    rospy.init_node('a2c', anonymous=True, log_level=node_log_level)
    rospy.loginfo(rospy.get_name())
    rospy.loginfo(args)
    rospy.loginfo(expr_path)

    # logging meta data
    exp_meta_file_handle = open(expr_path+'/meta_data.txt', 'w+')
    exp_meta_file_handle.write('Experiment ID : {} \n'.format(expr_id))
    exp_meta_file_handle.write('Experiment time : {} \n'.format(date_time))
    exp_meta_file_handle.write('Node name : {} \n'.format(rospy.get_name()))
    exp_meta_file_handle.write(str(args)+'\n')
    exp_meta_file_handle.write(COMMENT+'\n')
    exp_meta_file_handle.write('arguments: '+' '.join(sys.argv[1:])+'\n')
    exp_meta_file_handle.close()

    h = GoalHandler(THREADS, visualize = VIS_GOALS, defined_init = True, test = TEST)
    env_h = EnvHandler(THREADS, h, defined_init = True, red_state=hyper_params['red_states'], hz=HZ)

    envs = Environment(env_h=env_h, goal_h=h, threads_n=THREADS, test=TEST, episode_duration=EPISODE_DURATION,
    start_time=start_time, n_episodes_respawn_goals=EPISODES_TO_RESPAWN_GOAL, thread_delay=THREAD_DELAY,
    run_time=RUN_TIME, hyper_params=hyper_params, expr_path=expr_path, requested_hz=HZ,
    SEC_PAST=SEC_PAST, MAX_BATCH=MAX_BATCH, brain=brain, CHANGE_OF_VELOCITY=CHANGE_OF_VELOCITY, multi_env=False)
    # env_h.reset()

    envs.run()

    brain.model.save(expr_path+'/model.h5')
    # report.update(-1)
    # reset_dumper.close()
    rospy.loginfo("Training finished")
