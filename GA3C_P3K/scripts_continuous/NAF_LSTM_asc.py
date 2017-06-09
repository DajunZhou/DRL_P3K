import tensorflow as tf
import numpy as np
from replay_buffer import ReplayBuffer
from configure import *
import re
from tensorflow.contrib.framework import get_variables

GAMMA = configure.GAMMA
OBSERVE = configure.OBSERVE
# ANNELING_STEPS = configure.ANNELING_STEPS
INITIAL_EPSILON = configure.INITIAL_EPSILON
FINAL_EPSILON = configure.FINAL_EPSILON
REPLAY_MEMORY = configure.REPLAY_MEMORY
BATCH_SIZE = configure.BATCH_SIZE
EXPLORE = configure.EXPLORE
LSTM_layer = configure.LSTM_layer

class NAF():
    def __init__(self, model_name, action_dim, rebuffer, exp_queue=None, Trainer=False, Graph=None, Sess=None, ID=None):
        self.device = configure.DEVICE
        self.model_name = model_name
        self.action_dim = action_dim
        self.episode = 0
        self.timeStep = 0

        self.epsilon = INITIAL_EPSILON
        self.id = ID

        self.img_width = configure.IMAGE_WIDTH
        self.img_height = configure.IMAGE_HEIGHT
        self.img_channels = configure.STACKED_FRAMES * 4

        self.learning_rate = configure.LEARNING_RATE_START
        self.tau = configure.TargetNet_Tau

        self.replaybuffer = rebuffer
        self.exp_queue = exp_queue

        self.graph = Graph
        self.sess = Sess
        if Trainer:
            self.Trainer_Graph()
        else:
            self.Agent_Graph()

    def Agent_Graph(self):
        # with tf.variable_scope(self.model_name):
        with self.graph.as_default() as g:
            with tf.device(self.device):
                # with tf.variable_scope('Main_net'):
                with tf.variable_scope(self.model_name):
                    self.imageIn, self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, \
                    self.Advantage, self.Value, self.Qout, self.predict, self.actions,\
                    self.LSTM_state, self. init_lstim_state, self.next_lstm_state \
                        = self.__create_graph()

                self.MainNet_vars = get_variables(self.model_name)
                self.sess.run(tf.variables_initializer(self.MainNet_vars))


    def Trainer_Graph(self):
        # with tf.variable_scope(self.model_name):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                with tf.variable_scope('Main_net'):
                    self.imageIn, self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, \
                    self.Advantage, self.Value, self.Qout, self.predict, self.actions, \
                    self.LSTM_state, self. init_lstim_state, _\
                        = self.__create_graph()

                with tf.variable_scope('Target_net'):
                    self.imageInT, _,_,_,_,_,_, self.ValueT,_,_,_, \
                    self.LSTM_stateT, self.init_lstim_stateT, _ = self.__create_graph()

                self.MainNet_vars = get_variables('Main_net')
                self.TargetNet_vars = get_variables('Target_net')
                # var = tf.global_variables()
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


    def __create_graph(self):
        imageIn = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='imgIn')
        actions = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')

        conv1 = self.conv2d_layer(imageIn, 3, 256, 'conv1', strides=[1, 2, 2, 1])
        conv1_1 = self.conv2d_layer(conv1, 3, 256, 'conv1_1', strides=[1, 1, 1, 1])
        conv1_2 = self.conv2d_layer(conv1_1, 3, 256, 'conv1_2', strides=[1, 1, 1, 1])
        pool0 = self.mpool_layer(conv1_2, 2, [1,2,2,1], name='pool0')
        conv2 = self.conv2d_layer(pool0, 4, 256, 'conv2', strides=[1, 2, 2, 1])
        conv3 = self.conv2d_layer(conv2, 3, 256, 'conv3', strides=[1, 1, 1, 1])
        conv3_1 = self.conv2d_layer(conv3, 3, 512, 'conv3_1', strides=[1, 1, 1, 1])
        pool1 = self.mpool_layer(conv3_1, 2, [1,2,2,1], name='pool1')
        conv4 = self.conv2d_layer(pool1, pool1.get_shape()[1].value, 1024, 'conv4', strides=[1,1,1,1], padding='VALID')
        conv4_flatten = tf.contrib.layers.flatten(conv4)
        with tf.variable_scope('LSTM_layer'):
            LSTM_state = (tf.placeholder(tf.float32, [1, LSTM_layer]),
                          tf.placeholder(tf.float32, [1, LSTM_layer]))
            initial_lstm_state = (np.zeros([1, LSTM_layer], np.float32),
                                  np.zeros([1, LSTM_layer], np.float32))
            lstm_state = tf.contrib.rnn.LSTMStateTuple(*LSTM_state)
            lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_layer, forget_bias=0.0, state_is_tuple=True)

            batch_size = tf.shape(conv4_flatten)[:1]
            lstm_input = tf.expand_dims(conv4_flatten, [0])
            lstm_output, new_lstm_state = tf.nn.dynamic_rnn(lstm,
                                                            lstm_input,
                                                            batch_size,
                                                            lstm_state)
            lstm_output = tf.squeeze(lstm_output, [0])


        # fc_1 = self.fc_layer(conv4_flatten, 1024, 'fc_1')
        # fc_v = self.fc_layer(conv4_flatten, 300, 'fc_v')
        # fc_l = self.fc_layer(conv4_flatten, 300, 'fc_l')
        # fc_mu = self.fc_layer(conv4_flatten,300, 'fc_mu')
        V = self.fc_layer(lstm_output, 1, 'Value', func=None)
        with tf.variable_scope('advantage'):
            l = self.fc_layer(lstm_output, (self.action_dim*(self.action_dim + 1))/2, 'l', func=None)
            mu = self.fc_layer(lstm_output, self.action_dim, 'mu', func=tf.nn.tanh)

            pivot = 0
            rows = []
            for idx in xrange(self.action_dim):
                count = self.action_dim - idx

                diag_elem = tf.exp(tf.slice(l, (0, pivot), (-1, 1)))
                non_diag_elems = tf.slice(l, (0, pivot + 1), (-1, count - 1))
                Diag_non = tf.concat((diag_elem, non_diag_elems), 1)
                row = tf.pad(Diag_non, ((0, 0), (idx, 0))) # up,down, left,right  -> add 0
                rows.append(row)

                pivot += count

            L = tf.transpose(tf.stack(rows, axis=1), (0, 2, 1))
            P = tf.matmul(L, tf.transpose(L, (0, 2, 1)))
            tmp = tf.expand_dims(actions - mu, -1) # [?, act_dim] -> [?, act_dim, 1]
            A = -tf.matmul(tf.transpose(tmp, [0, 2, 1]), tf.matmul(P, tmp)) / 2
            A = tf.reshape(A, [-1, 1]) # [?, 1, 1] -> [?, 1, 1]

        with tf.variable_scope('Q'):
            Q = A + V

        return imageIn, conv1, conv2, conv3, pool1, conv4, A, V, Q, mu, actions, \
               LSTM_state, initial_lstm_state, new_lstm_state


    def createTrainingMethod(self):
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='targetQ')
        self.loss = tf.reduce_mean(tf.squared_difference(self.targetQ, tf.squeeze(self.Qout)), name='loss')
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.var_learning_rate)
        self.train_op = self.trainer.minimize(self.loss, global_step=self.global_step, name='train_update')

    def createupdateTargetNetOp(self):
        self.assign_op = {}
        for from_, to_ in zip(self.MainNet_vars, self.TargetNet_vars):
            self.assign_op[to_.name] = to_.assign(self.tau * from_ + (1 - self.tau) * to_)

    def updateTargetNet(self):
        for var in self.TargetNet_vars:
            self.sess.run(self.assign_op[var.name])

    def Copy_Net_Var_OP(self, net):
        self.copy_var_op = {}
        for from_, to_ in zip(net.MainNet_vars, self.MainNet_vars):
            self.copy_var_op[to_.name] = to_.assign(from_)

    def Copy_Net_to_Net(self):
        for var in self.MainNet_vars:
            self.sess.run(self.copy_var_op[var.name])

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
        # d = 1.0 / np.sqrt(in_dim)
        d = 3e-4
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
        for var in self.MainNet_vars:
            summaries.append(tf.summary.histogram("W_%s" % var.name, var))
        for var in self.TargetNet_vars:
            summaries.append(tf.summary.histogram("W_%s" % var.name, var))

        summaries.append(tf.summary.histogram("conv1", self.conv1))
        summaries.append(tf.summary.histogram("conv2", self.conv2))
        summaries.append(tf.summary.histogram("conv3", self.conv3))
        summaries.append(tf.summary.histogram("pool1", self.pool1))
        summaries.append(tf.summary.histogram("conv4", self.conv4))
        summaries.append(tf.summary.histogram("Advantage", self.Advantage))
        summaries.append(tf.summary.histogram("Value", self.Value))
        summaries.append(tf.summary.histogram("Qout", self.Qout))
        summaries.append(tf.summary.histogram("Predict", self.predict))

        self.summary_op = tf.summary.merge(summaries)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def log(self, y_batch, action_batch, state_batch):
        feed_dict = {self.targetQ: y_batch,
                     self.actions: action_batch,
                     self.imageIn: state_batch,
                     self.LSTM_state: self.init_lstim_state,
                     self.var_learning_rate: self.learning_rate}
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def trainQNetwork(self):
        if not self.exp_queue.empty():
            reward_sum = 0
            experiences = self.exp_queue.get()
            step_t = len(experiences)-1
            if experiences[step_t].terminal != True:
                End_V = self.sess.run(self.ValueT,
                                           feed_dict={self.imageInT:[experiences[step_t].nextstate],
                                                      self.LSTM_stateT: self.init_lstim_stateT})
                reward_sum = End_V[0][0]
            for t in reversed(range(0, step_t+1)):
                reward_sum = GAMMA*reward_sum + experiences[t].reward
                experiences[t].reward = reward_sum
            state_batch = np.asarray([data.state for data in experiences])
            action_batch = np.asarray([data.action for data in experiences])
            targetQ = np.asarray([data.reward for data in experiences])
            self.sess.run(self.train_op, feed_dict={self.imageIn: state_batch,
                                                    self.targetQ: targetQ, self.actions: action_batch,
                                                    self.LSTM_state: self.init_lstim_state,
                                                    self.var_learning_rate: self.learning_rate})
        else:
            minibatch = self.replaybuffer.get_batch(BATCH_SIZE)
            state_batch = np.asarray([data[0] for data in minibatch])
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            next_state_batch = np.asarray([data[3] for data in minibatch])
            done_batch = np.asarray([data[4] for data in minibatch])

            # action_batch = np.resize(action_batch, [BATCH_SIZE])

            V_target = self.sess.run(self.ValueT, feed_dict={self.imageInT:next_state_batch,
                                                             self.LSTM_stateT: self.init_lstim_stateT})

            targetQ = []
            for i in range(len(minibatch)):
                if done_batch[i]:
                    targetQ.append(reward_batch[i])
                else:
                    targetQ.append(reward_batch[i] + GAMMA * V_target[i])
            targetQ = np.resize(targetQ, [BATCH_SIZE])
            self.sess.run(self.train_op, feed_dict={self.imageIn:state_batch,
                                                    self.targetQ:targetQ, self.actions:action_batch,
                                                    self.LSTM_state: self.init_lstim_state,
                                                    self.var_learning_rate:self.learning_rate})

        self.updateTargetNet()

        if self.episode % configure.SAVE_NET == 0 and self.episode != 0:
            self.saver.save(self.sess, self.model_name + '/network' + '-naf',
                            global_step=self.episode)

        if configure.TENSORBOARD and self.episode % configure.TENSORBOARD_UPDATE_FREQUENCY == 0 and self.episode != 0:
            self.log(targetQ, action_batch, state_batch)

        if self.episode % 100 == 0:
            print "Training Episode:  ", self.episode

        self.episode += 1


    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.concatenate((self.currentState[:, :, 4:], nextObservation), axis=2)
        newState = nextObservation
        self.replaybuffer.add(self.currentState, action, reward, newState, terminal)
        # self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        if self.timeStep % 100 == 0:
            print self.model_name, "/ steptime", self.timeStep , "/ STATE", state, \
                "/ EPSILON", self.epsilon

        self.currentState = newState

    def Perce_Train(self):
        if self.replaybuffer.count() > configure.REPLAY_START_SIZE or self.exp_queue.qsize()>=configure.Exp_que_START_SIZE:
            self.trainQNetwork()

    def getAction(self, lstm_state):
        if np.random.rand(1) < self.epsilon:
            action_get = np.float32(np.random.uniform(-1, 1, self.action_dim))
            # action_get = np.array([1.0,0.0])
            # print "1: ", action_get
            # action_get[1] = np.random.choice([action_get[1], abs(action_get[1])],p=[0.5,0.5])
            # print "2:             ", action_get
            next_lstm_state = self.sess.run(self.next_lstm_state, feed_dict={self.imageIn:[self.currentState],
                                                                self.LSTM_state: lstm_state})
        else:
            action_get, next_lstm_state = self.sess.run((self.predict, self.next_lstm_state), feed_dict={self.imageIn:[self.currentState],
                                                                self.LSTM_state: lstm_state})
            action_get = np.resize(action_get, (self.action_dim))
            # print action_get

        if self.epsilon > (FINAL_EPSILON+(self.id)*0.23) and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - (FINAL_EPSILON+(self.id)*0.23)) / EXPLORE

        self.timeStep += 1
        return action_get, next_lstm_state

    def setInitState_rgb(self, observation):
        self.currentState = observation
        # for i in xrange(configure.STACKED_FRAMES-1):
        #     self.currentState = np.concatenate((self.currentState, observation), axis=2)