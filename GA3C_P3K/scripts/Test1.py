import rospy
from geometry_msgs.msg import Twist
import random
from GA3C_P3K.srv import *

def main():
    rospy.init_node('Test1')
    # Test1Pub = rospy.Publisher('/piokinect0/network_command', Twist, queue_size=100)
    rospy.wait_for_service('/energy_request')
    random_relocate_request_service = rospy.ServiceProxy('/energy_request', Data_request)
    relocate = random_relocate_request_service(0)
    print relocate.success, relocate.data,

    twistpub = Twist()

    # rate = rospy.Rate(1)
    # while not rospy.is_shutdown():
    #     twistpub.linear.x = random.uniform(0.0,0.5)
    #     twistpub.angular.z = random.uniform(-0.5,0.5)
    #     Test1Pub.publish(twistpub)
    #     rate.sleep()

if __name__ == '__main__':
    main()