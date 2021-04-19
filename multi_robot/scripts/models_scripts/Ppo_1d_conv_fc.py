#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
tf.keras.backend.set_floatx('float64')

from models_scripts.base_class import BaseBrain

class Ppo1DConvFCDisc(BaseBrain):
    def common_architecture_part(self):
        goals_current_robot_rel_state = Input(shape=(int(self.hz*self.seconds_past),self.row,self.threads))
        goals_other_robots_rel_state = Input(batch_shape=(None,self.threads-1, int(self.hz*self.seconds_past), self.row,self.threads))
        laser_state = Input(shape=(int(self.hz*self.seconds_past), self.row//2, 360))
        time_state = Input(batch_shape=(None,3))
        # print(goals_current_robot_rel_state)
        reshape_goals_current_robot = Flatten()(goals_current_robot_rel_state)
        reshape_goals_other_robots = Reshape((self.threads-1, int(self.hz*self.seconds_past)*self.row*self.threads))(goals_other_robots_rel_state)
        laser_reshape = Reshape(((self.row//2)*360, int(self.hz*self.seconds_past)))(laser_state)

        conv_laser = Conv1D(32, kernel_size=5, strides=2, activation='relu',
        data_format="channels_last", kernel_initializer=self.ortho_relu_init())(laser_reshape)
        conv_laser_1 = Conv1D(32, kernel_size=3, strides=2, activation='relu',
        data_format="channels_last", kernel_initializer=self.ortho_relu_init())(conv_laser)
        # conv_laser_2 = Conv1D(32, kernel_size=7, strides=2, activation='relu', data_format="channels_first")(conv_laser_1)
        # print(reshape_goals_other_robots)
        # dense_current_robot = Dense(32, activation='sigmoid', kernel_initializer=self.ortho_LinearOrSigmoid_init())(reshape_goals_current_robot)
        dense_other_robots = Dense(32, activation='relu', kernel_initializer=self.ortho_relu_init())(reshape_goals_other_robots)
        flatten_other_robots = Flatten()(dense_other_robots)
        flatten_conv_laser_scan = Flatten()(conv_laser_1)

        dense_laser = Dense(256, activation='relu', kernel_initializer=self.ortho_relu_init())(flatten_conv_laser_scan)

        concat_layer = Concatenate()([reshape_goals_current_robot, flatten_other_robots, dense_laser, time_state])

        last_common_layer = Dense(256, activation='relu', kernel_initializer=self.ortho_relu_init())(concat_layer)

        return goals_current_robot_rel_state, goals_other_robots_rel_state, laser_state ,time_state, last_common_layer

    def build_actor (self):
        # with tf.device(self.device):
        goals_current_robot_rel_state, goals_other_robots_rel_state, laser_state ,time_state, last_common_layer = self.common_architecture_part()
        v_l = Dense(self.discrete_size, activation='softmax', kernel_initializer=self.ortho_softmax_init())(last_common_layer)
        v_r = Dense(self.discrete_size, activation='softmax', kernel_initializer=self.ortho_softmax_init())(last_common_layer)

        layer1 = Lambda(lambda x: K.expand_dims(x, axis=1))(v_l)
        layer2 = Lambda(lambda x: K.expand_dims(x, axis=1))(v_r)
        act_out = Concatenate(axis=1)([layer1, layer2])

        model = Model(inputs=[goals_current_robot_rel_state, goals_other_robots_rel_state, laser_state ,time_state], outputs=[act_out])
        return model

    def build_critic(self):
        # with tf.device(self.device):
        goals_current_robot_rel_state, goals_other_robots_rel_state, laser_state ,time_state, last_common_layer = self.common_architecture_part()
        v = Dense(1, activation='linear', kernel_initializer=self.ortho_LinearOrSigmoid_init())(last_common_layer)
        model = Model(inputs=[goals_current_robot_rel_state, goals_other_robots_rel_state, laser_state ,time_state], outputs=[v])
        return model
