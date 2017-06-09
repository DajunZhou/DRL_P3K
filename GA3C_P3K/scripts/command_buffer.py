import rospy
from configure import configure
from buffer import *


number_of_bots = configure.number_of_bots
number_of_pred = configure.number_of_pred

def main():
    rospy.init_node('command_buffer', anonymous=True)
    piokinect_buffer = buffer('piokinect', number_of_bots)
    predator_buffer = buffer('predator', number_of_pred)

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        while not (piokinect_buffer.check_commands_received() and predator_buffer.check_commands_received()):
            r.sleep()

        piokinect_buffer.publish_commands()
        predator_buffer.publish_commands()

        # print "Go !"

        piokinect_buffer.notify_agents_commands_sent()
        predator_buffer.notify_agents_commands_sent()


if __name__ == '__main__':
    main()