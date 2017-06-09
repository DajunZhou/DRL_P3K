import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ContactsState
import re
from configure import configure
from std_msgs.msg import Bool
from GA3C_P3K.srv import *
#
# class piokinect():
#     def __init__(self):
#         pub = rospy.Publisher("/Ts1", ModelState, queue_size=100)
#         rospy.Publisher("/Ts2", ModelState, queue_size=100)

arena_width = configure.arena_width

class piokinect:
    def __init__(self, id, x, y, z, value = 0.0, typebot = 'piokinect'):
        if typebot == "predator":
            if value >= 0.0:
                value = -10.0
        self.id = id
        self.model_id = 0
        self.energy = configure.energy
        self.position = np.array([0.0,0.0,0.0,0.0])
        self.speed = 1.0
        self.previous_speed = 1.0
        self.velocity = np.array([0.0,0.0,0.0])
        self.typebot = typebot
        self.model_name  = typebot + str(id)
        self.position_updated = False
        self.collision_updated = False
        self.value = value
        self.have_terminal = Bool()
        self.have_terminal.data = True
        # self.is_use_net = False
        # self.is_obser = False
        self.is_relocate = False

        self.relocation_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
        self.have_Terminal = rospy.Publisher("/"+self.model_name+"/Terminal", Bool, queue_size=100)
        rospy.Subscriber("/"+self.model_name+"/collision_indicator", ContactsState, self.Contacts_callback, queue_size=100)
        self.relocation_message = ModelState()
        # rospy.Subscriber("/"+self.model_name+"/use_net", Bool, self.use_net_callback, tcp_nodelay=True, queue_size=100)
        # rospy.Subscriber("/" + self.model_name + "/have_obser", Bool, self.have_obser_callback, tcp_nodelay=True, queue_size=100)
        rospy.Subscriber("/" + self.model_name + "/is_relocate", Bool, self.is_relocate_callback, tcp_nodelay=True,queue_size=100)

        # self.have_consume_food_service = rospy.ServiceProxy('/Have_Consume_Food', Data_request)
        self.food_relocate_service = rospy.ServiceProxy('/food_relocate_request', Data_request)
    # def use_net_callback(self, msg):
    #     self.is_use_net = msg.data
    #
    # def have_obser_callback(self, msg):
    #     self.is_obser = msg.data

    def is_relocate_callback(self, msg):
        self.is_relocate = msg.data
        # print "recevive_relo"

    def Contacts_callback(self, msg):
        # xx = msg._connection_header['topic']
        # mymatch = re.match('/([a-zA-Z]*)([0-9]+)/([^,]+)',xx)
        # t1 = mymatch.group(1)
        # t2 = mymatch.group(2)
        # t3 = mymatch.group(3)
        # T2 = int(t2)

        colliding_object = {}
        link_name = {}
        type_obj = {}
        id_obj = {}
        # print len(msg.states)

        if not self.is_relocate:
            return

        if len(msg.states) == 0:
            if self.position[0] < -12.0:
                self.GoOut()
                self.have_Terminal.publish(self.have_terminal)

        if len(msg.states):
            # print "states"
            for key in xrange(0,1):

                collision1_name = msg.states[key].collision1_name
                collision2_name = msg.states[key].collision2_name
                # print collision2_name


                match1 = re.match('([^,]+)::([^,]+)::([^,]+)',collision1_name)
                colliding_object[1] = match1.group(1)
                link_name[1] = match1.group(2)

                match2 = re.match('([^,]+)::([^,]+)::([^,]+)',collision2_name)
                colliding_object[2] = match2.group(1)
                link_name[2] = match2.group(2)
                print colliding_object

                for i in [1,2]:
                    match_ = re.match("([a-zA-Z]*)([0-9]*)", colliding_object[i])
                    type_obj[i] = match_.group(1)
                    id_obj[i] = match_.group(2)
                    if type_obj[i] == 'food' :
                        self.consume(int(id_obj[i]))
                    elif type_obj[i] == 'predator' :
                        self.touched_predator(predators[id_obj[i]])
                        self.have_Terminal.publish(self.have_terminal)

                    elif type_obj[i] != 'piokinect':
                        self.collision_something()
                        self.have_Terminal.publish(self.have_terminal)
                    elif i == 2 and type_obj[1] == type_obj[2] and type_obj[1] == 'piokinect':
                        self.collision_bots()
                        self.have_Terminal.publish(self.have_terminal)
        # else:
        #     self.energy -= 0.1
        #     if self.energy <= 0:
        #         self.is_relocate = False
        #         self.have_Terminal.publish(self.have_terminal)
        self.collision_updated = True

    def random_relocate(self, distance):
        new_position = np.random.uniform([-arena_width,-arena_width,0.00, -3.1416],[arena_width,arena_width,0.01,3.1416],size=4)
        self.relocate(new_position)

    def upright(self):
        self.relocate(self.position)

    def relocate(self, new_positon):
        m = self.relocation_message
        m.model_name = self.model_name
        m.pose.position.x = new_positon[0]
        m.pose.position.y = new_positon[1]
        # m.pose.position.z = new_positon[2]
        m.pose.orientation.z = new_positon[3]
        m.pose.orientation.w = 1

        self.relocation_publisher.publish(m)
        self.position = new_positon
        self.energy = configure.energy
        # print "relocate"

    def touched_predator(self, predator):
        self.add_energy(predator.value)
        # self.collision_updated = True
        self.is_relocate = False

    def consume(self, id):
        # self.have_consume_food_service(id)
        self.food_relocate_service(id)
        self.add_energy(1.0)
            # self.collision_updated = True
    def collision_bots(self):
        self.add_energy(-15.0)
        self.is_relocate = False

    def collision_something(self):
        self.add_energy(-10.0)
        self.is_relocate = False
        # print "coll"

        # self.collision_updated = True

        # rospy.sleep(0.1)
        # if self.is_use_net:
        #     self.is_obser = False
        #     while not self.is_obser:
        #         rospy.sleep(0.001)
        #     self.is_use_net = False

        # self.random_relocate(arena_width)

    def GoOut(self):
        self.add_energy(10.0)
        self.is_relocate = False

    def add_energy(self, value):
        self.energy = self.energy + value