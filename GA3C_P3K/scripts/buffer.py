import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
import re


class buffer:
    def __init__(self, type_agent, n):
        self.type_agent = type_agent
        self.n = n
        self.commands_sent_msg = Bool()

        self.command_publishers = {}
        self.command_sent_publishers = {}
        self.commands = {}
        self.network_commands_received = {}

        for i in xrange(0, n):
            self.network_commands_received[i] = False
            self.commands[i] = Twist

        self.subsc_publish()
        # if self.type_agent == 'piokinect':
        self.create_command_sent_publishers()

    def twist_callback(self, msg):
        topic_msg = msg._connection_header['topic']
        mymatch = re.match('/([a-zA-Z]*)([0-9]*)/([^,]+)', topic_msg)
        i = int(mymatch.group(2))
        self.commands[i] = msg
        self.network_commands_received[i] = True
        # print "twist_cb"


    def subsc_publish(self):
        for i in xrange(0, self.n):
            # self.commands[i] = Twist
            rospy.Subscriber('/'+self.type_agent+str(i)+'/network_command', Twist, self.twist_callback, queue_size=100)
            self.command_publishers[i] = rospy.Publisher('/'+self.type_agent+str(i)+'/cmd_vel', Twist, queue_size=100)

    def create_command_sent_publishers(self):
        for i in xrange(0, self.n):
            self.command_sent_publishers[i] = rospy.Publisher('/'+self.type_agent+str(i)+'/commands_sent', Bool, queue_size=100)

    def check_commands_received(self):
        return self.check_received(self.network_commands_received, len(self.network_commands_received))

    def check_received(self, received, n):
        for i in xrange(0,n):
            if not received[i]:
                return False
        # print "True"
        return True

    def publish_commands(self):
        for i in xrange(0, self.n):
            self.command_publishers[i].publish(self.commands[i])
            self.network_commands_received[i] = False
        # if len(self.command_publishers) > 0 :
        #     self.command_publishers[0].publish(self.commands[0])
        #
        # for i in xrange(1,len(self.command_publishers)+1):
        #     self.command_publishers[i].publish(self.commands[i])
        #     self.network_commands_received[i] = False

    def notify_agents_commands_sent(self):
        self.commands_sent_msg.data = True

        for i in xrange(0, self.n):
            self.command_sent_publishers[i].publish(self.commands_sent_msg)
            # print "Have Send"
        #
        # if self.n > 0 :
        #     self.command_sent_publishers[0].publish(self.commands_sent_msg)
        #
        # for i in xrange(1, len(self.command_publishers)+1):
        #     self.command_sent_publishers[i].publish(self.commands_sent_msg)