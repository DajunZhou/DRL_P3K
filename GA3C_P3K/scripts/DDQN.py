import tensorflow as tf
import numpy as np
import random
from replay_buffer import ReplayBuffer
from configure import *
import re
from tensorflow.contrib.framework import get_variables

GAMMA = configure.GAMMA
OBSERVE = configure.OBSERVE
ANNELING_STEPS = configure.ANNELING_STEPS
INITIAL_EPSILON = configure.INITIAL_EPSILON
FINAL_EPSILON = configure.FINAL_EPSILON
REPLAY_MEMORY = configure.REPLAY_MEMORY
BATCH_SIZE = configure.BATCH_SIZE
EXPLORE = configure.EXPLORE

class DDQN:
    def __init__(self, model_name, action_dim):
        self.device = configure.DEVICE
        self.model_name = model_name
        self.action_dim = action_dim
        self.episode = 0
        # self.timeStep = 0
        self.STARTtrain = False
        self.epsilon = INITIAL_EPSILON

        self.img_width = configure.IMAGE_WIDTH
        self.img_height = configure.IMAGE_HEIGHT
        self.img_channels = configure.STACKED_FRAMES * 4

        self.learning_rate = configure.LEARNING_RATE_START
        self.tau = configure.TargetNet_Tau

        self.replaybuffer = ReplayBuffer(REPLAY_MEMORY)

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                with tf.variable_scope('Main_net'):
                    self.imageIn, self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, \
                    self.Advantage, self.Value, self.Qout, self.predict \
                        = self.__create_graph()

                with tf.variable_scope('Target_net'):
                    self.imageInT, _,_,_,_,_,_,_, self.QoutT, _ = self.__create_graph()

                self.MainNet_vars = get_variables('Main_net')
                self.TargetNet_vars = get_variables('Target_net')
                self.createTrainingMethod()
                self.createupdateTargetNetOp()

                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)
                    )
                )
                self.sess.run(tf.global_variables_initializer())

                if configure.TENSORBOARD:
                    self._create_tensor_board()
                # if configure.LOAD_CHECKPOINT or configure.SAVE_MODELS:
                #     vars = tf.global_variables()
                #     self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

                self.saver = tf.train.Saver()

                checkpoint = tf.train.get_checkpoint_state(self.model_name)
                if checkpoint and checkpoint.model_checkpoint_path:
                    self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                    print "Successfully loaded:", checkpoint.model_checkpoint_path
                    mypath = str(checkpoint.model_checkpoint_path)
                    stepmatch = re.split('-', mypath)[2]
                    self.episode = int(stepmatch)
                # pass
                else:
                    print "Could not find old network weights"

    # def __create_main_graph(self):
    #     self.imageIn = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='imgIn')
    #
    #     self.conv1 = self.conv2d_layer(self.imageIn, 8, 32, 'conv1', strides=[1, 4, 4, 1])
    #     self.conv2 = self.conv2d_layer(self.conv1, 4, 64, 'conv2', strides=[1, 2, 2, 1])
    #     self.conv3 = self.conv2d_layer(self.conv2, 3, 128, 'conv3', strides=[1, 1, 1, 1])
    #     self.conv4 = self.conv2d_layer(self.conv3, self.conv3.get_shape()[1].value, 512, 'conv4', strides=[1,1,1,1])
    #     with tf.variable_scope('A_V'):
    #         self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
    #         self.streamA = tf.contrib.layers.flatten(self.streamAC)
    #         self.streamV = tf.contrib.layers.flatten(self.streamVC)
    #
    #         self.AW = tf.Variable(tf.random_normal([self.streamA, self.action_dim]), name='AW')
    #         self.VW = tf.Variable(tf.random_normal([self.streamV, 1]), name='VW')
    #         self.Advantage = tf.matmul(self.streamA, self.AW, name='Advantage')
    #         self.Value = tf.matmul(self.streamV, self.VW, name='Value')
    #
    #     with tf.variable_scope('Qout'):
    #         self.Qout = self.Value + tf.subtract(
    #             self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
    #
    #     with tf.variable_scope('Predict'):
    #         self.predict = tf.argmax(self.Qout, 1)

    def __create_graph(self):
        imageIn = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='imgIn')

        conv1 = self.conv2d_layer(imageIn, 8, 128, 'conv1', strides=[1, 4, 4, 1])
        conv2 = self.conv2d_layer(conv1, 4, 128, 'conv2', strides=[1, 2, 2, 1])
        conv3 = self.conv2d_layer(conv2, 3, 128, 'conv3', strides=[1, 1, 1, 1])
        pool1 = self.mpool_layer(conv3, 2, [1,2,2,1], name='pool1')
        conv4 = self.conv2d_layer(pool1, pool1.get_shape()[1].value, 1024, 'conv4', strides=[1,1,1,1], padding='VALID')

        streamAC, streamVC = tf.split(conv4, 2, 3)
        streamA = tf.contrib.layers.flatten(streamAC)
        streamV = tf.contrib.layers.flatten(streamVC)

        Advantage = self.fc_layer(streamA, self.action_dim, 'Advantage', func=None)
        Value = self.fc_layer(streamV, 1, 'Value', func=None)

        # AW = tf.Variable(tf.random_normal([streamA.get_shape()[1].value, self.action_dim]), name='AW')
        # VW = tf.Variable(tf.random_normal([streamV.get_shape()[1].value, 1]), name='VW')
        # Advantage = tf.matmul(streamA, AW, name='Advantage')
        # Value = tf.matmul(streamV, VW, name='Value')
        with tf.variable_scope('Qout'):
            Qout = Value + tf.subtract(
                Advantage, tf.reduce_mean(Advantage, reduction_indices=1, keep_dims=True))
        with tf.variable_scope('Predict'):
            predict = tf.argmax(Qout, 1)

        return imageIn, conv1, conv2, conv3, pool1, conv4, Advantage, Value, Qout, predict

    # def __create_target_graph(self):
    #     self.target_imageIn = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels],
    #                                   name='imgIn')
    #     self.target_conv1 = self.conv2d_layer(self.target_imageIn, 8, 32, 'conv1', strides=[1, 4, 4, 1])
    #     self.target_conv2 = self.conv2d_layer(self.target_conv1, 4, 64, 'conv2', strides=[1, 2, 2, 1])
    #     self.target_conv3 = self.conv2d_layer(self.target_conv2, 3, 128, 'conv3', strides=[1, 1, 1, 1])
    #     self.target_conv4 = self.conv2d_layer(self.target_conv3, self.target_conv3.get_shape()[1].value, 512, 'conv4', strides=[1, 1, 1, 1])
    #     with tf.variable_scope('A_V'):
    #         self.target_streamAC, self.target_streamVC = tf.split(self.target_conv4, 2, 3)
    #         self.target_streamA = tf.contrib.layers.flatten(self.target_streamAC)
    #         self.target_streamV = tf.contrib.layers.flatten(self.target_streamVC)
    #
    #         self.target_AW = tf.Variable(tf.random_normal([self.target_streamA, self.action_dim]), name='AW')
    #         self.target_VW = tf.Variable(tf.random_normal([self.target_streamV, 1]), name='VW')
    #         self.target_Advantage = tf.matmul(self.target_streamA, self.target_AW, name='Advantage')
    #         self.target_Value = tf.matmul(self.target_streamV, self.target_VW, name='Value')
    #
    #     with tf.variable_scope('Qout'):
    #         self.Qout = self.target_Value + tf.subtract(
    #             self.target_Advantage, tf.reduce_mean(self.target_Advantage, reduction_indices=1, keep_dims=True))


    def createTrainingMethod(self):
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='targetQ')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.actions_onehot = tf.one_hot(self.actions, self.action_dim, dtype=tf.float32, name='act_onehot')
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1, name='Q')
        self.td_error = tf.square(self.targetQ - self.Q, name='td_error')
        self.loss = tf.reduce_mean(self.td_error, name='loss')
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.train_op = self.trainer.minimize(self.loss, global_step=self.global_step, name='train_update')

    def createupdateTargetNetOp(self):
        self.assign_op = {}
        for from_, to_ in zip(self.MainNet_vars, self.TargetNet_vars):
            self.assign_op[to_.name] = to_.assign(self.tau * from_ + (1 - self.tau) * to_)

    def updateTargetNet(self):
        for var in self.TargetNet_vars:
            self.sess.run(self.assign_op[var.name])

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu, padding='SAME'):
        in_dim = input.get_shape()[-1].value
        # in_dim = input.get_shape()[-1].value
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding=padding) + b
            if func is not None:
                output = func(output)

        return output

    def mpool_layer(self, input_op, mpool_size, strides, name):
        with tf.variable_scope(name):
            output = tf.nn.max_pool(input_op, ksize=[1, mpool_size, mpool_size, 1],
                                    strides=strides,
                                    padding="SAME")
        return output

    def fc_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape()[-1].value
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', dtype=tf.float32, shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Loss", self.loss))
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram("W_%s" % var.name, var))

        summaries.append(tf.summary.histogram("conv1", self.conv1))
        summaries.append(tf.summary.histogram("conv2", self.conv2))
        summaries.append(tf.summary.histogram("conv3", self.conv3))
        summaries.append(tf.summary.histogram("pool1", self.pool1))
        summaries.append(tf.summary.histogram("conv4", self.conv4))
        summaries.append(tf.summary.histogram("Advantage", self.Advantage))
        summaries.append(tf.summary.histogram("Value", self.Value))
        summaries.append(tf.summary.histogram("Qout", self.Qout))
        summaries.append(tf.summary.histogram("Q", self.Q))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def log(self, y_batch, action_batch, state_batch):
        feed_dict = {self.targetQ: y_batch,
                     self.actions: action_batch,
                     self.imageIn: state_batch,
                     self.var_learning_rate: self.learning_rate}
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def trainQNetwork(self):
        minibatch = self.replaybuffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        action_batch = np.resize(action_batch, [BATCH_SIZE])

        A = self.sess.run(self.predict,feed_dict={self.imageIn:next_state_batch})
        Q = self.sess.run(self.QoutT, feed_dict={self.imageInT:next_state_batch})
        doubleQ = Q[range(BATCH_SIZE), A]
        targetQ = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                targetQ.append(reward_batch[i])
            else:
                targetQ.append(reward_batch[i] + GAMMA * doubleQ[i])
        # targetQ = np.resize(targetQ, [BATCH_SIZE, 1])
        self.sess.run(self.train_op, feed_dict={self.imageIn:state_batch,
                                                self.targetQ:targetQ, self.actions:action_batch,
                                                self.var_learning_rate:self.learning_rate})

        self.updateTargetNet()

        if self.episode % configure.SAVE_NET == 0 and self.episode != 0:
            self.saver.save(self.sess, self.model_name + '/network' + '-dqn',
                            global_step=self.episode)

        if configure.TENSORBOARD and self.episode % configure.TENSORBOARD_UPDATE_FREQUENCY == 0 and self.episode != 0:
            self.log(targetQ, action_batch, state_batch)

        self.episode += 1
        self.STARTtrain = True

    def setPerception(self, nextObservation, action, reward, terminal):
        newState = np.concatenate((self.currentState[:, :, 4:], nextObservation), axis=2)
        self.replaybuffer.add(self.currentState, action, reward, newState, terminal)
        # self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if self.episode <= OBSERVE:
            state = "observe"
        elif self.episode > OBSERVE and self.episode <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.episode % 100 == 0 and self.STARTtrain:
            print "episode", self.episode , "/ STATE", state, \
                "/ EPSILON", self.epsilon

        self.currentState = newState

    def Perce_Train(self):
        if self.replaybuffer.count() > configure.REPLAY_START_SIZE:
            self.trainQNetwork()

    def getAction(self):
        if np.random.rand(1) < self.epsilon:
            action_get = np.random.randint(0, self.action_dim)
        else:
            action_get = self.sess.run(self.predict, feed_dict={self.imageIn:[self.currentState]})

        if self.epsilon > FINAL_EPSILON and self.episode > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action_get

    def setInitState_rgb(self, observation):
        self.currentState = observation
        for i in xrange(configure.STACKED_FRAMES-1):
            self.currentState = np.concatenate((self.currentState, observation), axis=2)