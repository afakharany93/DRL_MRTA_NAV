#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import rospy
import argparse
import csv
import time, random, threading, logging, pickle, glob
import os, sys

# import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
tf.keras.backend.set_floatx('float64')

from sklearn.utils import shuffle

from goals_handler import GoalHandler
from turt_robot import TurtBot
from env import Env
from env_handler import EnvHandler

from models_scripts.utils.brain_train_q import BrainTrainQ

class BaseBrain():
    brain_q = BrainTrainQ()

    def __init__(self, threads=3, min_batch = 4096, batch_ratio=1.1, max_batch=16392,
    hyper_params={}, model_path='None', expr_path = 'None', start_time=0, hz=25,
    name='None', worker=False):
        # tf.autograph.set_verbosity(3)
        # self.random_identifier = np.random.rand(10)
        # print(self.random_identifier)
        BaseBrain.brain_q.set_threads(threads)
        self.worker = worker
        critic_path = (model_path.replace('checkpoint', 'checkpoint_critic')).replace('model', 'critic_model')
        self.expr_path = expr_path
        if start_time == 0 :
            self.start_time = time.time()
        else:
            self.start_time = start_time

        self.action_size = 2

        self.threads = threads
        self.name = name

        if hyper_params == {}:
            hyper_params = {
            'gamma':0.95,
            'lr':1e-4,
            'lrd':0.96,
            'lrd_steps':10000,
            'dis_size':17,
            'epsilon':0.2,
            'entropy_factor':0.01,
            'value_factor':1.0,
            'seconds_past':3,
            'norm_returns':False,
            'red_states':True,
            'model_weights':'None',
            'epochs':1,
            'use_gae':False,
            'mini_batch_size': 2048
                            }

        self.entropy_factor = hyper_params['entropy_factor']
        self.value_factor = hyper_params['value_factor']
        self.epsilon = hyper_params['epsilon']
        self.gamma = hyper_params['gamma']
        # self.lmbda=hyper_params['lambda']
        self.seconds_past = hyper_params['seconds_past']
        self.starter_learning_rate = hyper_params['lr']
        self.lr_decay_steps = hyper_params['lrd_steps']
        self.lr_decay_rate = hyper_params['lrd']
        self.epochs = hyper_params['epochs']

        self.discrete_size = hyper_params['dis_size']
        self.mini_batch_size = hyper_params['mini_batch_size']

        self.norm_ret = hyper_params['norm_returns']
        self.use_gae = hyper_params['use_gae']
        self.red_states = hyper_params['red_states']
        self.model_weights = hyper_params['model_weights']
        self.critic_weights = (self.model_weights.replace('checkpoint', 'checkpoint_critic')).replace('model', 'critic_model')
        self.row = 2 if self.red_states else 4

        self.min_batch = min_batch
        self.batch_ratio = batch_ratio
        self.max_batch = max_batch
        self.continous_output = True
        self.hz=hz
        self.lock_acting = threading.Lock()

        # self.put_debug_file('0')


        # self.put_debug_file('1')

        self.model, self.model_critic = self._build_model()

        # self.put_debug_file('2')


        if model_path != 'None':
            temp_model = load_model(model_path)
            temp_critic = load_model(critic_path)
            self.model.set_weights(temp_model.get_weights())
            self.model_critic.set_weights(temp_critic.get_weights())
        elif self.model_weights != 'None':
            self.model.load_weights(self.model_weights)
            self.model_critic.load_weights(self.critic_weights)

        #rospy.logdebug('end building model')

        self.update_number = 0

        # self.default_graph = tf.get_default_graph()

        self.past_model_path = ''
        self.current_model_path = ''
        self.timer_snap = time.time()
        # self.act_out, self.vout = self.get_action_and_value()
        # in case of this is the training network
        if self.worker == False:
            self.old_model, self.old_model_critic = self._build_model()
            self.old_model.set_weights(self.model.get_weights())
            self.old_model_critic.set_weights(self.model_critic.get_weights())

            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.compat.v1.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                       self.lr_decay_steps, self.lr_decay_rate, staircase=False)
            self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.critic_learning_rate = tf.compat.v1.train.exponential_decay(1e-3, self.global_step,
                                                       self.lr_decay_steps, self.lr_decay_rate, staircase=False)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)

            self.model.summary()
            self.model_critic.summary()
            self.ckpt_path = expr_path+'/checkpoint'
            os.mkdir(self.ckpt_path)
            os.chmod(self.ckpt_path, 0o777)
            self.ckpt_path_critic = expr_path+'/checkpoint_critic'
            os.mkdir(self.ckpt_path_critic)
            os.chmod(self.ckpt_path_critic, 0o777)
            self.f = open(expr_path+"/{}.csv".format('brain'), "w")
            self.f.flush()
            self.writer = csv.writer(self.f)
            self.prepare_csv_header()
            plot_model(self.model, to_file=self.expr_path+'/model.png', show_shapes=True)
            plot_model(self.model_critic, to_file=self.expr_path+'/model_critic.png', show_shapes=True)
            # self.put_debug_file('3')

    def put_debug_file(self, name=''):
        name = 'debug_nn_'+self.name+'_'+name
        os.mknod(self.expr_path+"/"+name)
        os.chmod(self.expr_path+"/"+name, 0o777)

    def ortho_relu_init(self):
        return tf.keras.initializers.Orthogonal(np.sqrt(2))

    def ortho_LinearOrSigmoid_init(self):
        return tf.keras.initializers.Orthogonal(1)

    def ortho_leaky_relu_init(self, alpha=0.3):
        return tf.keras.initializers.Orthogonal(np.sqrt(2/(1+alpha**2)))

    def ortho_softmax_init(self):
        return tf.keras.initializers.Orthogonal(0.01)

    def _build_model(self):
        # with tf.device(self.device):
        actor = self.build_actor()
        actor._make_predict_function()    # have to initialize before threading


        critic = self.build_critic()
        critic._make_predict_function()    # have to initialize before threading

        return actor, critic

    def build_actor (self):
        # with tf.device(self.device):
        model = Model()
        return model

    def build_critic(self):
        # with tf.device(self.device):
        model = Model()
        return model

    def make_model_copy(self):
        model_copy= tf.keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())
        return model_copy

    def update_model_copy_weights(self, model_copy):
        model_copy.set_weights(self.model.get_weights())
        return model_copy

    def copy_weights(self, model_copy):
        self.model.set_weights(model_copy.get_weights())
        # return model_copy

    @tf.function(experimental_relax_shapes=True)
    def calc_action (self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t):
        act_out = self.model([s_current_robot_t, s_goals_t, s_laser_t, s_time_t])
        v = self.model_critic([s_current_robot_t, s_goals_t, s_laser_t, s_time_t])
        # act_out = tf.stack([v_l_onehot, v_r_onehot], axis=1)
        return act_out, v

    # @tf.function
    def calc_old_action (self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t):
        old_act_out = self.old_model([s_current_robot_t, s_goals_t, s_laser_t, s_time_t])
        # old_act_out = tf.stack([old_v_l_onehot, old_v_r_onehot], axis=1)
        return old_act_out

    def calc_old_value (self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t):
        old_value = self.old_model_critic([s_current_robot_t, s_goals_t, s_laser_t, s_time_t])
        return old_value

    def calc_policy_loss(self, act_out, old_act_out, a_t, advantages_t):
        a_t_onehot = tf.one_hot(a_t, self.discrete_size, dtype=tf.float64)

        log_prob = tf.math.log( tf.reduce_sum(act_out * a_t_onehot, axis=2) )

        # old_log_prob = tf.math.log( tf.clip_by_value(tf.reduce_sum(old_act_out * a_t_onehot, axis=2, keep_dims=True), 1e-10, 1.0) )
        old_log_prob = tf.math.log( tf.reduce_sum(old_act_out * a_t_onehot, axis=2))
        ratio = tf.exp(log_prob - tf.stop_gradient(old_log_prob))

        # with tf.control_dependencies(clone_ops):
        surr1 =  ratio * advantages_t
        surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)* advantages_t
        policy_loss = tf.reduce_mean(tf.minimum(surr1, surr2))
        # value_loss = -1*self.value_factor*tf.reduce_mean(tf.square(r_t - v))
        entropy_all = -1*self.entropy_factor * tf.reduce_sum(act_out * tf.math.log(tf.clip_by_value(act_out, 1e-10, 1.0)), axis=-1)
        entropy = tf.reduce_mean(entropy_all)

        return policy_loss, entropy

    # def calc_value_loss(self, v, r_t):
    #     value_loss = tf.reduce_mean(tf.square(r_t - v))
    #     return value_loss

    def calc_value_loss(self, v, old_v, r_t):
        surr_1 = tf.square(v - r_t)
        clip = tf.clip_by_value(v, old_v - self.epsilon, old_v + self.epsilon)
        surr_2 = tf.square(clip - r_t)
        value_loss = tf.reduce_mean(tf.minimum(surr_1, surr_2))
        return value_loss

    # @tf.function
    def calc_loss (self, v, old_v, act_out, old_act_out, a_t, r_t, advantages_t):
        policy_loss, entropy = self.calc_policy_loss(act_out, old_act_out, a_t, advantages_t)
        value_loss = self.calc_value_loss(v, old_v, r_t)
        value_loss_1 = -1*self.value_factor*value_loss
        loss = tf.reduce_mean(-1*(policy_loss+value_loss_1+entropy))

        return loss, policy_loss, value_loss, entropy

    def train_policy_step(self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, advantages_t):
        # train the policy
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            act_out, v = self.calc_action (s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            old_act_out = self.calc_old_action (s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            # loss, policy_loss, value_loss, entropy = self.calc_loss (v, act_out, old_act_out, a_t, r_t, advantages_t)
            policy_loss, entropy = self.calc_policy_loss(act_out, old_act_out, a_t, advantages_t)
            loss = tf.reduce_mean(-1*(policy_loss + entropy))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, policy_loss, entropy

    def train_critic_step(self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t, r_t):
        # train the critic
        with tf.GradientTape() as tape:
            _, v = self.calc_action (s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            value_loss = self.calc_value_loss(v, r_t)
        gradients = tape.gradient(value_loss, self.model_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.model_critic.trainable_variables))
        return value_loss

    def train_combined_step(self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, r_t, advantages_t):
        with tf.GradientTape() as tape:
            act_out, v = self.calc_action (s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            old_act_out = self.calc_old_action (s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            old_v = self.calc_old_value(s_current_robot_t, s_goals_t, s_laser_t, s_time_t)
            loss, policy_loss, value_loss, entropy = self.calc_loss (v, old_v, act_out, old_act_out, a_t, r_t, advantages_t)
        sources = self.model.trainable_variables + self.model_critic.trainable_variables
        gradients = tape.gradient(loss, sources)
        self.policy_optimizer.apply_gradients(zip(gradients, sources))
        # value_loss = -1*value_loss
        return loss, policy_loss, value_loss, entropy

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, r_t, advantages_t):
        # loss, policy_loss, entropy = self.train_policy_step(s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, advantages_t)
        # value_loss = self.train_critic_step(s_current_robot_t, s_goals_t, s_laser_t, s_time_t, r_t)
        loss, policy_loss, value_loss, entropy = self.train_combined_step(s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, r_t, advantages_t)

        return loss, policy_loss, value_loss, entropy

    def calc_returns(self, r, v,s_mask):
        returns = np.zeros_like(r)
        rolling = 0
        for i in reversed(range(len(r))):
            if s_mask[i]:
                rolling = 0
            rolling = rolling * self.gamma + r[i]
            returns[i] = rolling
        if self.norm_ret:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10) # usual normalization
        adv = returns - v
        return adv, returns

    def calc_gae_adv(self, rwrds, v, s_mask):
        rewards = rwrds.copy()
        gae = np.zeros_like(rewards)
        ret = np.zeros_like(rewards)
        gamma = self.gamma
        lmbda = 0.95
        rolling = 0
        masks = 1 - s_mask.astype('float32')
        v = np.append(v,v[-1])
        v = v.reshape((-1,1))
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * v[i + 1] * masks[i] - v[i]
            rolling = delta + gamma * lmbda * masks[i] * rolling
            gae[i] = rolling
            ret[i] = rolling+v[i]
        if self.norm_ret:
            ret = adv + v[:-1]
        return gae, ret

    def optimize(self):
        # global self.min_batch
        if BaseBrain.brain_q.check_memory_size(self.min_batch):
            time.sleep(0)    # yield
            return False

        with BaseBrain.brain_q.lock_queue:
            if BaseBrain.brain_q.check_memory_size(self.min_batch) :    # more thread could have passed without lock
                return False                                     # we can't yield inside lock

            arrays = BaseBrain.brain_q.get_train_data()
            # print('before empty', len(arrays[1]))

            self._optimization_procedures(arrays)
            self.model.save(self.ckpt_path+'/model_ckpt_{}.h5'.format(str(self.update_number).zfill(5)))
            self.model_critic.save(self.ckpt_path_critic+'/critic_model_ckpt_{}.h5'.format(str(self.update_number).zfill(5)))
            BaseBrain.brain_q.empty_train_queue()
            # print('after empty', len(arrays[1]))
        return True

    def me_optimize(self, arrays):
        self._optimization_procedures(arrays)
        self.save_model()
        return True

    def _optimization_procedures(self, arrays):
        # print(self.random_identifier)
        s1, s2, s3, s4, a, r, s1_, s2_, s3_, s4_, v, s_mask = arrays

        s1 = np.array(s1)
        s2 = np.array(s2)
        s3 = np.array(s3)
        s4 = np.array(s4).reshape((-1,3))
        a = np.array(a).squeeze()
        r = np.array(r).reshape((-1,1))
        s1_ = np.array(s1_)
        s2_ = np.array(s2_)
        s3_ = np.array(s3_)
        s4_ = np.array(s4_).reshape((-1,3))
        v = np.array(v).reshape((-1,1))
        s_mask = np.array(s_mask)

        a = self.cont_to_disc(a)
        # print(s2.shape)
        s2 = np.transpose(s2, (0, 2, 1, 3, 4))
        self.old_model.set_weights(self.model.get_weights())
        self.old_model_critic.set_weights(self.model_critic.get_weights())
        rospy.logdebug("Optimizer Minimizing batch of {}".format(len(s1)))

        if self.use_gae:
            advantages, returns = self.calc_gae_adv(r, v, s_mask)
        else:
            advantages, returns = self.calc_returns(r, v, s_mask)
        if self.norm_ret:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-10) # usual normalization
        # s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, r_t, advantages_t, minimize, act_out, v, loss,policy_loss, value_loss, entropy  = self.graph
        self.batch_size = len(s1)
        mini_batch_size = self.mini_batch_size
        rospy.loginfo('optimizing on data size: {}'.format(self.batch_size))
        for i in range(self.epochs):
            s1, s2, s3, s4, a, returns, advantages = shuffle(s1, s2, s3, s4, a, returns, advantages)
            self.l = self.policy_l = self.value_l = self.entropy_l = counter =0
            for j in range(0, self.batch_size, mini_batch_size):
                s1_m, s2_m , s3_m, s4_m,  a_m, returns_m, advantages_m = s1[j:j+mini_batch_size], s2[j:j+mini_batch_size] , s3[j:j+mini_batch_size], s4[j:j+mini_batch_size],  a[j:j+mini_batch_size], returns[j:j+mini_batch_size], advantages[j:j+mini_batch_size]
                advantages_m = (advantages_m-np.mean(advantages_m))/(np.std(advantages_m) + 1e-8)
                s1_m, s2_m , s3_m, s4_m = tf.convert_to_tensor(s1_m), tf.convert_to_tensor(s2_m) , tf.convert_to_tensor(s3_m), tf.convert_to_tensor(s4_m)
                a_m, returns_m, advantages_m = tf.convert_to_tensor(a_m), tf.convert_to_tensor(returns_m), tf.convert_to_tensor(advantages_m)
                l_m, policy_l_m, value_l_m, entropy_l_m = self.train_step(s1_m, s2_m , s3_m, s4_m, a_m, returns_m, advantages_m)
                counter += 1
                self.l += l_m.numpy()
                self.policy_l += policy_l_m.numpy()
                self.value_l += value_l_m.numpy()
                self.entropy_l += entropy_l_m.numpy()
            self.l = self.l/counter
            self.policy_l = self.policy_l/counter
            self.value_l = self.value_l/counter
            self.entropy_l = self.entropy_l/counter
            if self.worker == False:
                self.write_csv_row(i)
            self.global_step = self.global_step + counter
        self.update_number += 1
        self.min_batch = int(min(self.min_batch*self.batch_ratio, self.max_batch))
        rospy.loginfo('Done optimizing on data size: {}, loss = {}'.format(len(s1), self.l))


    def predict(self, s, sample=False):
        # print(self.random_identifier)
        # with self.lock_acting:
        # with tf.device(self.device):
        s1 = tf.convert_to_tensor(s[0])
        s2 = s[1]
        s3 = tf.convert_to_tensor(s[2])
        s4 = tf.convert_to_tensor(s[3])
        s2 = tf.convert_to_tensor(np.transpose(s2, (0, 2, 1, 3, 4)))
        # s_current_robot_t, s_goals_t, s_laser_t, s_time_t, a_t, r_t, advantages_t, minimize, act_out, v, loss,policy_loss, value_loss, entropy  = self.graph
        action_prop_dist, value = self.calc_action(s1, s2, s3, s4)
        if sample:
            action = self.softmax_sampling(action_prop_dist)
        else:
            action = np.argmax(action_prop_dist, -1)
        action = self.disc_to_cont(action)
        return action, value

    def save_model(self):
        self.model.save_weights(self.ckpt_path+'/model_ckpt_{}.h5'.format( str(self.update_number).zfill(5) ))
        self.model_critic.save_weights(self.ckpt_path_critic+'/critic_model_ckpt_{}.h5'.format(str(self.update_number).zfill(5)))

    def me_save_data(self):
        if time.time() - self.timer_snap < 300 :
            if BaseBrain.brain_q.check_memory_size(self.min_batch):
                time.sleep(0)    # yield
                return False

        with BaseBrain.brain_q.lock_queue:
            if time.time() - self.timer_snap < 300 :
                if BaseBrain.brain_q.check_memory_size(self.min_batch) :    # more thread could have passed without lock
                    return False                                     # we can't yield inside lock

            l = BaseBrain.brain_q.get_train_data()

            if self.worker == False:
                if not os.path.isfile(self.expr_path+'/arrays.pickle') :
                    with open(self.expr_path+'/start_pickle.txt', 'wb') as fp1:
                        pass
                    with open(self.expr_path+'/arrays.pickle', 'wb') as fp:
                        pickle.dump(l, fp)
                    with open(self.expr_path+'/true.txt', 'wb') as fp2:
                        pass
                    self.min_batch = int(min(self.min_batch*self.batch_ratio, self.max_batch))
                    BaseBrain.brain_q.empty_train_queue()
                    self.timer_snap = time.time()
                    return True
                self.timer_snap = time.time()
            # BaseBrain.brain_q.empty_train_queue()
            return False

    def load_latest_model(self, main_experiment_path):
        '''
        Used by workers dockers to load latest trained model
        '''
        # self.put_debug_file('load_model_0')
        self.timer_snap = time.time()
        models_paths = glob.glob(main_experiment_path+'/checkpoint/*.h5')
        models_paths.sort()
        critic_models_paths = glob.glob(main_experiment_path+'/checkpoint_critic/*.h5')
        critic_models_paths.sort()
        # self.put_debug_file('load_model_1')
        if models_paths[-1] == self.current_model_path:
            # self.put_debug_file('load_model_4')
            return False
        else:
            try:
                # self.put_debug_file('load_model_2')
                self.model.load_weights(models_paths[-1])
                self.model_critic.load_weights(critic_models_paths[-1])
                self.current_model_path = models_paths[-1]
                # self.put_debug_file('load_model_3')
                return True
            except:
                # self.put_debug_file('load_model_5')
                return False
        # self.put_debug_file('load_model_6')
        return False

    def prepare_csv_header(self):
        print('preparing header')
        columnTitleRow = ['absolute_time', 'ros_time', 'update_number', 'epoch', 'min_batch_size', 'actual_batch_size', 'loss', 'policy_loss', 'value_loss', 'entropy']
        self.writer.writerow(columnTitleRow)
        self.f.flush()

    def write_csv_row(self,i=0):
        row = [time.strftime("%H:%M:%S", time.gmtime(time.time()-self.start_time)), rospy.get_time(), self.update_number, i, self.min_batch, self.batch_size, self.l, np.mean(self.policy_l), self.value_l, np.mean(self.entropy_l)]
        self.writer.writerow(row)
        self.f.flush()

    def cont_to_disc(self, a):
        quanta = 2/self.discrete_size
        d = ((a+1)//quanta).astype(np.int32)
        d = np.clip(d, 0, self.discrete_size-1)
        return d

    def disc_to_cont (self, a):
        quanta = 2/self.discrete_size
        a = np.clip(a, 0, self.discrete_size-1)
        c = (a.astype(np.float32)*quanta)-1+(quanta/2)
        c = np.round(c,3)
        return c

    def softmax_sampling(self, soft):
        matrix_U = np.random.rand(*soft.shape)
        final_num = np.argmax(soft - matrix_U, axis=-1)
        # final_num = np.argmax((soft-soft.max(axis=-1, keepdims=True)) - matrix_U, axis=-1)
        # print('softmax: ',soft, '\n sample',(soft-soft.max(axis=-1, keepdims=True)) - matrix_U, '\n',np.argmax(soft, axis=-1),final_num )
        #the following part is to avoid choosing actions with low probabilities
        for i in range(len(final_num)):
            for j in range(final_num.shape[1]):
                if soft[i,j,final_num[i,j]] < 0.01:
                    # print('changing prop from {} to {}'.format(final_num[i,j], np.argmax(soft[i,j,:])))
                    final_num[i,j] = np.argmax(soft[i,j,:])
        return final_num

    def train_push(self, s, a, r, s_, v,done):
        BaseBrain.brain_q.train_push(s, a, r, s_, v,done)

    def train_push_list(self, queue_data, name):
        BaseBrain.brain_q.train_push_list(queue_data, name)

    def exit(self):
        if self.worker == False:
            self.expr_path = 'None'
            self.f.close()
