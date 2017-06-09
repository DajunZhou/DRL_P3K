from threading import Thread
from configure import *
from NAF_asc import NAF
from A3C import A3C
import Gazebo_Env_asc as game
import cv2
import numpy as np
from Experience import Experience

img_h = configure.IMAGE_HEIGHT
img_w = configure.IMAGE_WIDTH

class ThreadAgent(Thread):
    def __init__(self, id, rebuffer, network, exp_queue):
        super(ThreadAgent, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.Agent = NAF('Agent'+str(id), configure.actions_dim, rebuffer, exp_queue=exp_queue, Graph=network.graph, Sess=network.sess, ID=id)
        self.Agent.Copy_Net_Var_OP(network)
        self.env = game.GazeboEnv(self.id)
        self.stepMax = configure.Step
        self.exp_que = exp_queue

    def RGBimg_preprocess(self, rgb, depth):
        # rgb = cv2.resize(rgb, (img_h, img_w))
        rgb = rgb.astype(np.float32) / 128.0 - 1.0
        # depth = cv2.resize(depth, (img_h, img_w))
        # depth = np.reshape(depth, (img_h, img_w, 1))
        depth = depth / 4.5 - 1.0
        w_nan = np.isnan(depth)
        depth[w_nan] = 1.0
        return np.concatenate((rgb, depth), axis=2)

    def run(self):
        
        for episode in xrange(configure.EPISODES):

            self.Agent.Copy_Net_to_Net()

            #copy NET
            state_rgb0, state_depth0 = self.env.reset()
            # state_rgb0 = cv2.resize(state_rgb0, (img_h, img_w))
            state_rgb0 = state_rgb0.astype(np.float32) / 128.0 - 1.0

            # state_depth0 = cv2.resize(state_depth0, (img_h, img_w))
            # state_depth0 = np.reshape(state_depth0, (img_h, img_w, 1))
            state_depth0 = state_depth0 / 4.5 - 1.0
            w_nan = np.isnan(state_depth0)
            state_depth0[w_nan] = 1.0

            Obser0 = np.concatenate((state_rgb0, state_depth0), axis=2)

            self.Agent.setInitState_rgb(Obser0)
            self.currentState = Obser0
            total_reward = 0.0
            episode_steptime = 0
            experiences = []
            step_num = 0
            while 1 != 0 and episode_steptime<900:
                action = self.Agent.getAction()
                state_rgb, state_depth, reward, terminal = self.env.step(action)

                step_num += 1

                total_reward += reward
                next0state = self.RGBimg_preprocess(state_rgb, state_depth)
                self.Agent.setPerception(next0state, action, reward, terminal)
                episode_steptime += 1

                exp = Experience(self.currentState, action, reward, next0state, terminal)
                experiences.append(exp)
                if step_num == self.stepMax or terminal:
                    self.exp_que.put(experiences)
                    step_num = 0
                    experiences = []

                if terminal or episode_steptime >= 900:
                    print 'Agent'+str(self.id), "  Total: ", total_reward, "  Rebuffer: ", self.Agent.replaybuffer.count(),
                    print '  Exp_Que_size: ', self.exp_que.qsize()
                    break
                self.currentState = next0state