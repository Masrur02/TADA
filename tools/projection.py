#!/usr/bin/env python3
import rospy
import ros_numpy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import message_filters

class TravProjector:
    def __init__(self):
        self.bridge = CvBridge()

        # Subscribers
        trav_sub = message_filters.Subscriber("/trav_map", Image)
        pc_sub   = message_filters.Subscriber("/D435i/depth/color/points", PointCloud2)

        ats = message_filters.ApproximateTimeSynchronizer(
            [trav_sub, pc_sub], queue_size=5, slop=0.5
        )
        ats.registerCallback(self.callback)

        # Publisher: single combined cloud
        self.pub_trav_cloud = rospy.Publisher("/trav_cloud", PointCloud2, queue_size=1)
        rospy.loginfo("✅ Traversability projector initialized (x,y,z,trav).")

    def callback(self, trav_msg, pc_msg):
        # --- 1️⃣ Convert traversability map to NumPy ---
        trav_img = self.bridge.imgmsg_to_cv2(trav_msg, desired_encoding="32FC1")
        H, W = trav_img.shape
        trav_flat = trav_img.flatten().astype(np.float32)

        # --- 2️⃣ Convert PointCloud2 → structured NumPy array ---
        pc_arr = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        pc_flat = pc_arr.reshape(-1)

        if pc_flat.shape[0] != H * W:
            rospy.logwarn(f"⚠️ Size mismatch: trav_map={H}x{W}, pointcloud={pc_flat.shape[0]}")
            return

        # --- 3️⃣ Build new structured array with x,y,z,trav ---
        new_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('confidence_ratio', np.float32)
        ])

        pc_combined = np.empty(pc_flat.shape, dtype=new_dtype)
        pc_combined['x'] = pc_flat['x'].astype(np.float32)
        pc_combined['y'] = pc_flat['y'].astype(np.float32)
        pc_combined['z'] = pc_flat['z'].astype(np.float32)
        pc_combined['confidence_ratio'] = trav_flat

        # --- 4️⃣ Convert back to PointCloud2 ---
        out_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            pc_combined,
            stamp=pc_msg.header.stamp,
            frame_id=pc_msg.header.frame_id
        )
        out_msg.is_dense = pc_msg.is_dense

        self.pub_trav_cloud.publish(out_msg)
        rospy.loginfo_throttle(5.0, "📤 Published /trav_cloud (x,y,z,trav)")

if __name__ == "__main__":
    rospy.init_node("trav_projector")
    TravProjector()
    rospy.spin()
