#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def relay_cb(msg):
    pub.publish(msg)

rospy.init_node("rviz_goal_relay")
pub = rospy.Publisher("/goal", PoseStamped, queue_size=1)
rospy.Subscriber("/move_base_simple/goal", PoseStamped, relay_cb)
rospy.loginfo("✅ Relaying RViz 2D Nav Goal → /goal")
rospy.spin()
