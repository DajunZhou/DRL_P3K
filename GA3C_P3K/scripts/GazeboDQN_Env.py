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
# import genpy
#
# c = genpy.Time()
# c.to_sec()


class GazeboEnv:
    def __init__(self):
        self.type_bot = self.type_agent()
        self.episode_time = configure.episode_time
        self.arena_width = configure.arena_width
        self.number_colour_channels = configure.number_colour_channels
        self.number_channels = configure.number_channels
        self.laser_scan_range = configure.laser_scan_range
        self.number_of_cameras = configure.number_of_cameras
        self.camera_size = configure.camera_size
        self.min_reward = configure.min_reward
        self.max_reward = configure.max_reward
        self.energy = configure.energy
        self.forward_speed = configure.forward_speed
        self.angular_speed = configure.angular_speed
        self.anctions_dim = configure.actions_dim
        self.command_message = Twist()
        self.current_observation = 'torch.Tensor(self.number_channels, self.camera_size, self.number_of_cameras):zero()'
        self.step_count = 0
        self.number_steps_in_episode = 'opts.valSteps --assuming 150 step/s for 180 seconds'
        self.number_of_sensors = 2
        self.updated = {False, False, False}
        self.initialised = False
        self.command_sent = False
        self.current_time = 0
        self.latest_sensor_updates = {0, 0, 0}
        self.scan_sensor_index = 2
        self.bridge = CvBridge()
        self.Terminal = False
        # self.use_net = Bool()
        # self.use_net.data = True
        # self.have_Obser = Bool()
        # self.have_Obser.data = True
        self.have_relocate = Bool()
        self.have_relocate.data = True
        self.act_reward = 0.0

        rospy.init_node("GazeboDQN_Env",anonymous=True) ###


    def type_agent(self):
        return "piokinect"

    def getStateSpec(self):
        return {'real', {self.number_channels, self.camera_size, self.number_of_cameras}, {0, 1}}

    def getActionSpec(self):
        return {'int', 1, {0, 2}}

    def getRewardSpec(self):
        return self.min_reward, self.max_reward

    def clock_callback(self, msg):
        self.current_time = msg.clock.to_sec()

    def comsent_callback(self, msg):
        self.command_sent = msg.data
        # print "Sent"

    def img_rgb_callback(self, ros_img):
        # bridge = CvBridge()
        # time_s = time.time()
        img_rgb = self.bridge.imgmsg_to_cv2(ros_img, "bgr8")
        img_rgb = np.array(img_rgb, dtype=np.uint8)
        self.img_rgb = img_rgb
        # time_e = time.time()
        # print time_e - time_s
        cv2.imshow('rgb_img', self.img_rgb)
        cv2.waitKey(2)

    def img_dep_callback(self, ros_img):
        img_dep = self.bridge.imgmsg_to_cv2(ros_img, "32FC1")
        img_dep = np.array(img_dep, dtype=np.float32)
        self.img_dep = img_dep
        # cv2.imshow('dep_img', self.img_dep)
        # cv2.waitKey(2)

    def terminal_callback(self, msg):
        self.Terminal = msg.data
        # print 'Terminal!'


    # def laser_callback(self, msg):
    #     self.current_observation  #### RGB + laser
    #     pass


    def reset(self):
        if not self.initialised:
            self.initialised = True
            # rospy.sleep(4)
            self.id = 0 ######
            self.model_name = self.type_bot + str(self.id)

            self.command_publisher = rospy.Publisher('/' + self.model_name + '/network_command', Twist, queue_size=100)
            # self.USE_NET = rospy.Publisher('/' + self.model_name + '/use_net', Bool, tcp_nodelay=True, queue_size=100)
            # self.Have_Obser = rospy.Publisher('/' + self.model_name + '/have_obser', Bool, tcp_nodelay=True, queue_size=100)
            self.Have_Relocate = rospy.Publisher('/' + self.model_name + '/is_relocate', Bool, tcp_nodelay=True, queue_size=100)

            self.clock_subscriber = rospy.Subscriber('/clock', Clock, self.clock_callback, queue_size=100)
            self.is_Terminal = rospy.Subscriber('/' + self.model_name + '/Terminal', Bool, self.terminal_callback, queue_size=100)

            rospy.wait_for_service('/random_relocate_request')
            self.random_relocate_request_service = rospy.ServiceProxy('/random_relocate_request',Data_request)
            rospy.wait_for_service('/energy_request')
            self.energy_request_service = rospy.ServiceProxy('/energy_request', Data_request)
            rospy.wait_for_service('/speed_request')
            self.speed_request_service = rospy.ServiceProxy('/speed_request', Data_request)

            self.command_sent_subscriber = rospy.Subscriber('/' + self.model_name + '/commands_sent', Bool,
                                                            self.comsent_callback, queue_size=100)

            bot_image_color = "/" + self.model_name + "/kinect2/hd/image_color"  ### ### ### ###
            self.camera_input_subscribers = rospy.Subscriber(bot_image_color, Image, self.img_rgb_callback, queue_size=100)

            bot_image_depth = "/" + self.model_name + "/kinect2/hd/image_depth"
            self.cdepth_input_subscribers = rospy.Subscriber(bot_image_depth, Image, self.img_dep_callback, queue_size=100)

        # self.random_relocate_message = self.random_relocate_request_service(self.id)


        self.command_message.linear.x = 0.0
        self.command_message.angular.z = 0.0
        while not self.command_sent:
            self.command_publisher.publish(self.command_message)
            rospy.sleep(0.001)
        self.command_sent = False

        # rospy.sleep(0.5)
        self.Terminal = True
        while self.Terminal:
            self.Terminal = False
            self.random_relocate_message = self.random_relocate_request_service(self.id)
            rospy.sleep(0.3)
            self.Have_Relocate.publish(self.have_relocate)
            rospy.sleep(1.2)

        # rospy.sleep(0.5)

        # self.Terminal = False
        # self.Have_Relocate.publish(self.have_relocate)
        self.energy = 0.0
        self.start_time = self.current_time
        return self.img_rgb, self.img_dep


    # def start(self):
    #
    #     # random_relocate_request_service = 0
    #
    #     if not self.initialised:
    #         self.initialised = True
    #         rospy.sleep(4)
    #         self.id = __threadid or 0
    #         self.model_name = self.type_bot + str(self.id)
    #
    #         # command_destination = "/network_command"
    #         self.command_publisher = rospy.Publisher('/'+self.model_name+'/network_command', Twist, queue_size=100)
    #         self.clock_subscriber = rospy.Subscriber('/clock', Clock, self.clock_callback, queue_size=100)
    #
    #         random_relocate_request_service = rospy.ServiceProxy('/random_relocate_request',Data_request)
    #         energy_request_service = rospy.ServiceProxy('/energy_request', Data_request)
    #         speed_request_service = rospy.ServiceProxy('/speed_request', Data_request)
    #
    #         self.command_sent_subscriber = rospy.Subscriber('/'+self.model_name+'/commands_sent',Bool, self.comsent_callback, queue_size=100)
    #
    #         # self.camera_input_subscribers = {}
    #
    #         bot_imageraw = "/"+self.model_name+"/image_raw" ### ### ### ###
    #         self.camera_input_subscribers = rospy.Subscriber(bot_imageraw, Image, self.img_callback, queue_size=100)
    #
    #         bot_scan = "/"+self.model_name+"/scan_sensor"
    #         self.laser_input_subscriber = rospy.Subscriber(bot_scan, LaserScan, self.laser_callback, queue_size=100)
    #
    #         if not ros.isStarted():
    #             self.spinner = ros.AsyncSpinner()
    #             self.spinner:start()
    #
    #     print('[Robot '+self.model_name+' finished episode with '+str(self.energy)+' energy]')
    #     energy_request_message = random_relocate_request_service(self.id)
    #
    #     self.start_time = self.current_time
    #     return self.current_observation


    def isValidationAgent(self):
        return self.id == 0

    def parse_action(self, action):
        action_taken = np.array([0.0,0.0])
        if action == 0:
            action_taken[0] = self.forward_speed
            action_taken[1] = 0
            self.act_reward = 0.000#0.005
        elif action == 1:
            action_taken[0] = self.forward_speed/1.5
            action_taken[1] = self.angular_speed
            self.act_reward = -0.0005#-0.0001
        elif action == 2:
            action_taken[0] = self.forward_speed/1.5
            action_taken[1] = -self.angular_speed
            self.act_reward = -0.0005#-0.0001
        return action_taken

    def check_received(self, received, n):
        for i in xrange(1,n+1):
            if not received[i]:
                return False
        return True

    def step(self, action):
        self.step_count = self.step_count + 1
        action_taken = self.parse_action(action)
        self.command_message.linear.x = action_taken[0]
        self.command_message.angular.z = action_taken[1]

        self.command_sent = False
        # r = rospy.Rate(100)
        while not self.command_sent:
            self.command_publisher.publish(self.command_message)
            rospy.sleep(0.002)
            # r.sleep()

        rospy.sleep(0.0333)
        self.command_sent = False


        # self.command_message.linear.x = 0.0
        # self.command_message.angular.z = 0.0
        # while not self.command_sent:
        #     self.command_publisher.publish(self.command_message)
        #     rospy.sleep(0.01)
        # self.command_sent = False

    def Oberved(self):
        old_energy = self.energy
        self.energy_request_message = self.energy_request_service(self.id)
        self.energy = self.energy_request_message.data
        reward = self.energy - old_energy
        # if reward > 10:
        #     terminal = True

        reward = reward + self.act_reward
        # reward = reward + self.current_observation[1]:sum()/self.camera_size
        Img_rgb = self.img_rgb
        Img_dep = self.img_dep
        Terminal = self.Terminal
        if Terminal:
            reward = reward - self.act_reward
            print "event: ", reward
            self.command_message.linear.x = 0.0
            self.command_message.angular.z = 0.0
            while not self.command_sent:
                self.command_publisher.publish(self.command_message)
                rospy.sleep(0.001)
            self.command_sent = False
        # else:
        #     Terminal = self.current_time - self.start_time > self.episode_time
            # print "TimeOut!"
            # pass

            # self.Have_Obser.publish(self.have_Obser)
        # print "reward: ", reward
        return Img_rgb, Img_dep, reward, Terminal


    # def step(self, action):
    #     self.step_count = self.step_count + 1
    #     # terminal = False
    #
    #     # while not self.check_received(self.updated, self.number_of_sensors):
    #     #     rospy.sleep(0.01)
    #     #
    #     # for i in xrange(0, self.number_of_sensors):
    #     #     self.updated[i] = False
    #
    #     action_taken = self.parse_action(action)
    #     self.command_message.linear.x = action_taken[0]
    #     self.command_message.angular.z = action_taken[1]
    #
    #     # terminal = self.current_time - self.start_time > self.episode_time
    #     self.command_sent = False
    #     # r = rospy.Rate(100)
    #     while not self.command_sent:
    #         self.command_publisher.publish(self.command_message)
    #         rospy.sleep(0.01)
    #         # r.sleep()
    #
    #       ###
    #     rospy.sleep(0.02)
    #     self.command_sent = False
    #
    #     self.command_message.linear.x = 0.0
    #     self.command_message.angular.z = 0.0
    #     while not self.command_sent:
    #         self.command_publisher.publish(self.command_message)
    #         rospy.sleep(0.01)
    #     self.command_sent = False
    #
    #     old_energy = self.energy
    #     self.energy_request_message = self.energy_request_service(self.id)
    #     self.energy = self.energy_request_message.data
    #     reward = self.energy - old_energy
    #     # if reward > 10:
    #     #     terminal = True
    #
    #     reward = reward - 0.001
    #     # reward = reward + self.current_observation[1]:sum()/self.camera_size
    #
    #     return self.img_rgb, self.img_dep, reward, self.Terminal






