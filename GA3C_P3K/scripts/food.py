import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ContactsState
import re
from configure import configure
from std_msgs.msg import Bool

arena_width = configure.arena_width + 1

class food():
    def __init__(self, id, x, y, z, value):
        self.id = id
        self.model_name = 'food' + str(id)
        self.value = value
        self.position = np.array([0.0,0.0,0.0,0.0])
        self.position[0] = x
        self.position[1] = y
        self.position[2] = z
        self.relocation_message = ModelState()
        self.immunity_duration = 0.03
        self.immunity_start_time = 0
        self.edible = True

        self.relocation_publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=100)
        # self.relocation_service = rospy.ServiceProxy()

    def random_relocate(self, distance):
        new_position = np.random.uniform([-arena_width-2, -arena_width-1, 0.00, -3.1416],
                                         [arena_width, arena_width+1, 0.01, 3.1416], size=4)
        self.relocate(new_position)

    def have_consumed(self):
        new_position = np.array([12.0, float(self.id*0.25), 0, 0])
        self.relocate(new_position)

    def relocate(self, new_position):
        m = self.relocation_message
        m.model_name = self.model_name
        m.pose.position.x = new_position[0]
        m.pose.position.y = new_position[1]
        m.pose.position.z = 0.05
        m.pose.orientation.z = new_position[3]
        m.pose.orientation.w = 1

        self.relocation_publisher.publish(m)
        self.position = new_position