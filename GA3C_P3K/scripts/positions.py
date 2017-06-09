#!/usr/bin/env python

import rospy
# from std_msgs.msg import String
# from gazebo_msgs.msg import ContactsState
# from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates


# import re

initialised = False
latest_message = ModelStates()
def callback(msg):
    global latest_message
    latest_message = msg
    # print latest_message
    # rospy.sleep(10)
    global initialised
    initialised = True



def main():

    rospy.init_node('positions',anonymous=True)

    position_publisher = rospy.Publisher("/throttled_model_states", ModelStates, queue_size=100)
    # latest_message = ModelStates()
    rospy.Subscriber("/gazebo/model_states", ModelStates, callback, queue_size=100)

    frequency = 2
    r = rospy.Rate(frequency)
    while not rospy.is_shutdown():
        if initialised:
            position_publisher.publish(latest_message)
            # print latest_message.name[31]
            # rospy.sleep(1/frequency)
            r.sleep()

    # rospy.Subscriber("collision_indicator", ContactsState, callback)

    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()



if __name__ == '__main__':
    main()

    #'MyKinectV2::base_link::base_link_fixed_joint_lump__KinectCo_collision_2'
    #'bookshelf::link::left_side'