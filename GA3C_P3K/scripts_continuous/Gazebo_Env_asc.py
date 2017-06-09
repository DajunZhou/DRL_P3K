import rospy
from configure import *
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from GA3C_P3K.srv import *
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, LaserScan
import numpy as np
from cv_bridge import CvBridge
import cv2
import time

img_h = configure.IMAGE_HEIGHT
img_w = configure.IMAGE_WIDTH
count = 0

class GazeboEnv:
    def __init__(self, id):
        self.id = id
        self.type_bot = self.type_agent()
        self.model_name = self.type_bot + str(self.id)
        self.episode_time = configure.episode_time
        self.arena_width = configure.arena_width
        self.energy = configure.energy
        self.AllEnergy = configure.allenergy
        self.forward_speed_max = configure.forward_speed_max
        self.forward_speed_min = configure.forward_speed_min
        self.angular_speed_max = configure.angular_speed_max
        self.angular_speed_min = configure.angular_speed_min
        self.actions_dim = configure.actions_dim
        self.command_message = Twist()
        self.step_count = 0
        self.initialised = False
        # self.command_sent = False
        self.current_time = 0
        self.bridge = CvBridge()
        self.Terminal = False
        self.have_relocate = Bool()
        self.have_relocate.data = True
        self.not_relocate = Bool()
        self.not_relocate.data = False
        self.act_reward = -0.01
        self.RGB_array = np.zeros([img_h, img_w, 9])
        self.Dep_array = np.zeros([img_h, img_w, 3])
        self.x = 0.0001/self.forward_speed_max

        # rospy.init_node("GazeboDQN_Env",anonymous=True) ###


    def type_agent(self):
        return "piokinect"

    def clock_callback(self, msg):
        self.current_time = msg.clock.to_sec()

    # def comsent_callback(self, msg):
    #     self.command_sent = msg.data
        # print "Sent"

    def img_rgb_callback(self, ros_img):
        # bridge = CvBridge()
        # time_s = time.time()
        img_rgb = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        img_rgb = np.array(img_rgb, dtype=np.uint8)
        img_rgb0 = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])
        img_rgb1 = cv2.resize(img_rgb0,(img_h, img_w))
        img_rgb1 = np.resize(img_rgb1, (img_rgb1.shape[0], img_rgb1.shape[1], 1))
        # self.RGB_array = np.concatenate((self.RGB_array[:,:,3:], img_rgb1), axis=2)
        self.img_rgb = img_rgb1
        # time_e = time.time()
        # print time_e - time_s

        if self.id == 0:
            cv2.imshow('rgb_img', img_rgb)
            cv2.waitKey(2)
        # global count
        # count += 1

    def img_dep_callback(self, ros_img):
        img_dep = self.bridge.imgmsg_to_cv2(ros_img, "32FC1")
        img_dep = np.array(img_dep, dtype=np.float32)
        img_dep1 = cv2.resize(img_dep, (img_h, img_w))
        img_dep1 = np.resize(img_dep1, (img_dep1.shape[0], img_dep1.shape[1], 1))
        # self.Dep_array = np.concatenate((self.Dep_array[:,:,1:], img_dep1), axis=2)
        # global count
        # count+=1
        self.img_dep = img_dep1
        # cv2.imshow('dep_img', self.img_dep)
        # cv2.waitKey(2)

    def terminal_callback(self, msg):
        self.Terminal = msg.data
        # print 'Terminal!'



    def reset(self):
        if not self.initialised:
            self.initialised = True
            # self.model_name = self.type_bot + str(self.id)

            # self.command_publisher = rospy.Publisher('/' + self.model_name + '/network_command', Twist, queue_size=100)
            self.command_publisher = rospy.Publisher('/' + self.model_name + '/cmd_vel', Twist, queue_size=100)

            self.Have_Relocate = rospy.Publisher('/' + self.model_name + '/is_relocate', Bool, tcp_nodelay=True, queue_size=100)

            self.clock_subscriber = rospy.Subscriber('/clock', Clock, self.clock_callback, queue_size=100)
            self.is_Terminal = rospy.Subscriber('/' + self.model_name + '/Terminal', Bool, self.terminal_callback, queue_size=100)

            rospy.wait_for_service('/random_relocate_request')
            self.random_relocate_request_service = rospy.ServiceProxy('/random_relocate_request',Data_request)
            rospy.wait_for_service('/energy_request')
            self.energy_request_service = rospy.ServiceProxy('/energy_request', Data_request)
            rospy.wait_for_service('/speed_request')
            self.speed_request_service = rospy.ServiceProxy('/speed_request', Data_request)

            # self.command_sent_subscriber = rospy.Subscriber('/' + self.model_name + '/commands_sent', Bool,
            #                                                 self.comsent_callback, queue_size=100)

            bot_image_color = "/" + self.model_name + "/kinect2/hd/image_color"  ### ### ### ###
            self.camera_input_subscribers = rospy.Subscriber(bot_image_color, Image, self.img_rgb_callback, queue_size=100)

            bot_image_depth = "/" + self.model_name + "/kinect2/hd/image_depth"
            self.cdepth_input_subscribers = rospy.Subscriber(bot_image_depth, Image, self.img_dep_callback, queue_size=100)

            rospy.sleep(0.5)

        self.command_message.linear.x = 0.0
        self.command_message.angular.z = 0.0
        # while not self.command_sent:
        self.command_publisher.publish(self.command_message)
        rospy.sleep(0.001)
        # self.command_sent = False

        # rospy.sleep(0.5)
        self.Terminal = True
        while self.Terminal:
            self.Terminal = False
            self.random_relocate_message = self.random_relocate_request_service(self.id)
            rospy.sleep(0.3)
            self.Have_Relocate.publish(self.have_relocate)
            rospy.sleep(1.2)

        self.energy = configure.energy
        self.AllEnergy = configure.allenergy
        self.start_time = self.current_time

        self.RGB_array = np.concatenate((self.img_rgb,self.img_rgb,self.img_rgb), axis=2)
        self.Dep_array = np.concatenate((self.img_dep,self.img_dep,self.img_dep), axis=2)
        return self.RGB_array, self.Dep_array

    def isValidationAgent(self):
        return self.id == 0

    # def parse_action(self, action):
    #     action_taken = np.array([0.0,0.0])
    #     if action == 0:
    #         action_taken[0] = self.forward_speed
    #         action_taken[1] = 0
    #         self.act_reward = 0.000#0.005
    #     elif action == 1:
    #         action_taken[0] = self.forward_speed/1.5
    #         action_taken[1] = self.angular_speed
    #         self.act_reward = -0.0005#-0.0001
    #     elif action == 2:
    #         action_taken[0] = self.forward_speed/1.5
    #         action_taken[1] = -self.angular_speed
    #         self.act_reward = -0.0005#-0.0001
    #     return action_taken

    def check_received(self, received, n):
        for i in xrange(1,n+1):
            if not received[i]:
                return False
        return True

    def step(self, action):
        self.step_count = self.step_count + 1
        # action_taken = self.parse_action(action)
        # print "action:                    ", action

        action[0] = (action[0] + 1.0)*(self.forward_speed_max-self.forward_speed_min)/2.0 + self.forward_speed_min
        action[1] = (action[1] + 1.0)*(self.angular_speed_max-self.angular_speed_min)/2.0 + self.angular_speed_min
        # print "action:                    ", action
        # if action[1] > 0:
        #     pass

        self.command_message.linear.x = action[0]
        self.command_message.angular.z = action[1]

        # self.command_sent = False
        # r = rospy.Rate(100)
        # while not self.command_sent:
        # global count

        self.command_publisher.publish(self.command_message)
        # count = 0
        # rospy.sleep(0.001)
            # r.sleep()
        # s = time.time()
        rospy.sleep(0.111)
        self.RGB_array = np.concatenate((self.RGB_array[:,:,1:], self.img_rgb), axis=2)
        self.Dep_array = np.concatenate((self.Dep_array[:,:,1:], self.img_dep), axis=2)
        rospy.sleep(0.111)
        self.RGB_array = np.concatenate((self.RGB_array[:, :, 1:], self.img_rgb), axis=2)
        self.Dep_array = np.concatenate((self.Dep_array[:, :, 1:], self.img_dep), axis=2)
        rospy.sleep(0.111)
        # self.command_sent = False

        # self.command_message.linear.x = 0.0
        # self.command_message.angular.z = 0.0
        # # while not self.command_sent:
        # self.command_publisher.publish(self.command_message)
        # rospy.sleep(0.01)
        # # self.command_sent = False

        old_energy = self.energy

        # print 'count: ', count, "  ", time.time() - s


        self.energy_request_message = self.energy_request_service(self.id)
        self.energy = self.energy_request_message.data
        wd = (action[0]*self.x/(abs(action[1])+0.01))
        # print action, "    ",wd
        reward = self.energy - old_energy + self.act_reward + wd

        # if reward > 0:
        #     self.AllEnergy += reward
        # else:
        #     self.AllEnergy += self.act_reward + wd

        self.RGB_array = np.concatenate((self.RGB_array[:, :, 1:], self.img_rgb), axis=2)
        self.Dep_array = np.concatenate((self.Dep_array[:, :, 1:], self.img_dep), axis=2)

        # Img_rgb = self.RGB_array
        # Img_dep = self.Dep_array
        # print 'count: ', count, "  ", time.time() - s
        Terminal = self.Terminal

        # if self.AllEnergy <= 0.0:
        #     Terminal = True
        #     self.Have_Relocate.publish(self.not_relocate)
        #     reward = reward + self.act_reward + wd

        if Terminal:
            reward = reward - self.act_reward - wd
            print "Agent"+str(self.id), "  event: ", reward
            self.command_message.linear.x = 0.0
            self.command_message.angular.z = 0.0
            # while not self.command_sent:
            self.command_publisher.publish(self.command_message)
            # rospy.sleep(0.001)
            # self.command_sent = False

        return self.RGB_array, self.Dep_array, reward, Terminal



    def Get_CurrentState(self):
        return self.RGB_array, self.Dep_array





