import cv2
import GazeboDQN_Env as game
from BrainDQN_RGB import *
import numpy as np
from configure import *
import time
from DDQN import *
import rospy
from GA3C_P3K.srv import *


# def img_preprocess(observation):
#     # cv2.imshow('depth_img', observation)
#     # cv2.waitKey(2)
#     observation = cv2.resize(observation, (80, 80))
#     observation = np.reshape(observation, (80,80,1))
#     observation = observation / 4.5
#     w_nan = np.isnan(observation)
#     observation[w_nan] = 2.0
#     return observation
img_h = configure.IMAGE_HEIGHT
img_w = configure.IMAGE_WIDTH


def RGBimg_preprocess(rgb, depth):
    rgb = cv2.resize(rgb, (img_h,img_w))
    rgb = rgb.astype(np.float32) / 128.0 - 1.0
    depth = cv2.resize(depth, (img_h,img_w))
    depth = np.reshape(depth,(img_h,img_w,1))
    depth = depth / 4.5
    w_nan = np.isnan(depth)
    depth[w_nan] = 2.0
    return np.concatenate((rgb, depth), axis=2)


# def playPioKinect():
#
#     env = game.GazeboEnv()
#     brain = BrainDQN(env.anctions_dim)
#     for episode in xrange(configure.EPISODES):
#         state_rgb0, state0 = env.reset()  ##_
#         # state0 = img_preprocess(state0)
#         state_rgb0 = cv2.resize(state_rgb0,(80, 80))
#         state_rgb0 = state_rgb0.astype(np.float32) / 128.0 - 1.0
#
#         state0 = cv2.resize(state0, (80, 80))
#         # state0 = np.reshape(state0, (80, 80))
#         state0 = np.reshape(state0, (80, 80, 1))
#         state0 = state0 / 4.5
#         w_nan = np.isnan(state0)
#         state0[w_nan] = 2.0
#
#         Obser0 = np.concatenate((state_rgb0, state0), axis=2)
#
#         brain.setInitState_rgb(Obser0)
#
#         action_vec, action = brain.getAction()
#         env.step(action)
#         total_reward = 0.0
#         while 1 != 0:
#
#             # start = time.time()
#             brain.Perce_Train() # 0.01s
#             # end = time.time()
#             # print end - start
#             ##perception
#             state_rgb, state_depth, reward, terminal = env.Oberved() ##_
#             total_reward += reward
#
#             next0state = RGBimg_preprocess(state_rgb, state_depth)
#             # next0state = img_preprocess(next0state)
#             brain.setPerception(next0state, action_vec, reward, terminal)
#             if terminal:
#                 print "Total: ", total_reward
#                 break
#             action_vec, action = brain.getAction()
#             env.step(action)
#
#             # action_vec, action = brain.getAction()
#             # _, next0state, reward, terminal = env.step(action)
#             # next0state = img_preprocess(next0state)
#             # brain.setPerception(next0state, action_vec, reward, terminal)
#             # if terminal:
#             #     break

def playPioKinect():
    rospy.wait_for_service('/food_relocate_request')
    food_relocate_request_service = rospy.ServiceProxy('/food_relocate_request', Data_request)
    food_numb = configure.number_of_food

    env = game.GazeboEnv()
    brain = DDQN('DDQN2',env.anctions_dim)
    for episode in xrange(configure.EPISODES):
        for id in xrange(0, food_numb):
            _ = food_relocate_request_service(id)
            rospy.sleep(0.05)

        state_rgb0, state0 = env.reset()  ##_
        # state0 = img_preprocess(state0)
        state_rgb0 = cv2.resize(state_rgb0,(img_h,img_w))
        state_rgb0 = state_rgb0.astype(np.float32) / 128.0 - 1.0

        state0 = cv2.resize(state0, (img_h,img_w))
        # state0 = np.reshape(state0, (80, 80))
        state0 = np.reshape(state0, (img_h,img_w, 1))
        state0 = state0 / 4.5
        w_nan = np.isnan(state0)
        state0[w_nan] = 2.0

        Obser0 = np.concatenate((state_rgb0, state0), axis=2)

        brain.setInitState_rgb(Obser0)

        action = brain.getAction()
        env.step(action)
        total_reward = 0.0
        episode_steptime = 0
        while 1 != 0 and episode_steptime<500:

            # start = time.time()
            brain.Perce_Train() # 0.01s

            state_rgb, state_depth, reward, terminal = env.Oberved() ##_
            total_reward += reward

            next0state = RGBimg_preprocess(state_rgb, state_depth)
            # next0state = img_preprocess(next0state)
            brain.setPerception(next0state, action, reward, terminal)
            episode_steptime += 1
            if terminal or episode_steptime>=500:
                print "Total: ", total_reward
                break
            action = brain.getAction()
            env.step(action)


def main():
    playPioKinect()

if __name__ == '__main__':
    main()