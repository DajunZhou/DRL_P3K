from threading import Thread
from configure import *
from DDQN_asc import DDQN
import Gazebo_Env_asc as game
import cv2
import numpy as np

img_h = configure.IMAGE_HEIGHT
img_w = configure.IMAGE_WIDTH

class ThreadAgent(Thread):
    def __init__(self, id, rebuffer, network):
        super(ThreadAgent, self).__init__()
        self.setDaemon(True)

        self.id = id
        self.Agent = DDQN('Agent'+str(id), configure.actions_dim, rebuffer, Graph=network.graph, Sess=network.sess, ID=id)
        self.Agent.Copy_Net_Var_OP(network)
        self.env = game.GazeboEnv(self.id)

    def RGBimg_preprocess(self, rgb, depth):
        rgb = cv2.resize(rgb, (img_h, img_w))
        rgb = rgb.astype(np.float32) / 128.0 - 1.0
        depth = cv2.resize(depth, (img_h, img_w))
        depth = np.reshape(depth, (img_h, img_w, 1))
        depth = depth / 4.5
        w_nan = np.isnan(depth)
        depth[w_nan] = 2.0
        return np.concatenate((rgb, depth), axis=2)

    def run(self):
        
        for episode in xrange(configure.EPISODES):

            self.Agent.Copy_Net_to_Net()

            #copy NET
            state_rgb0, state_depth0 = self.env.reset()
            state_rgb0 = cv2.resize(state_rgb0, (img_h, img_w))
            state_rgb0 = state_rgb0.astype(np.float32) / 128.0 - 1.0

            state_depth0 = cv2.resize(state_depth0, (img_h, img_w))
            state_depth0 = np.reshape(state_depth0, (img_h, img_w, 1))
            state_depth0 = state_depth0 / 4.5
            w_nan = np.isnan(state_depth0)
            state_depth0[w_nan] = 2.0

            Obser0 = np.concatenate((state_rgb0, state_depth0), axis=2)

            self.Agent.setInitState_rgb(Obser0)

            action = self.Agent.getAction()
            self.env.step(action)
            total_reward = 0.0
            episode_steptime = 0
            while 1 != 0 and episode_steptime<800:
                state_rgb, state_depth, reward, terminal = self.env.Oberved()
                total_reward += reward

                next0state = self.RGBimg_preprocess(state_rgb, state_depth)
                self.Agent.setPerception(next0state, action, reward, terminal)
                episode_steptime += 1
                if terminal or episode_steptime >= 800:
                    print 'Agent'+str(self.id), "  Total: ", total_reward
                    break
                action = self.Agent.getAction()
                self.env.step(action)