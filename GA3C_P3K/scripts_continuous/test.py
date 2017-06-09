import rospy
from geometry_msgs.msg import Twist

def main():
    rospy.init_node('speed_test')
    command_publisher = rospy.Publisher('/piokinect0/cmd_vel', Twist, queue_size=1)
    twist = Twist()
    twist.linear.x = 0.5
    twist.angular.z = 5.0
    rospy.sleep(0.2)
    command_publisher.publish(twist)
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     command_publisher.publish(twist)
    #     rate.sleep()

if __name__ == '__main__':
    main()