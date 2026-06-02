'''
Author: BIT807s
LastEditors: 解博炜 8894423+xie-bowei@user.noreply.gitee.com
LastEditTime: 2025-02-25
FilePath: hci_topic_vis.py
Description: 
Copyright (c) 2025 by BIT807s, All Rights Reserved. 
'''


import rospy
import numpy as np
import threading
from piper_msgs.msg import PosCmd
from show_traj import show_trajectory3d_web, show_trajectory3d_comp_web


class C_PiperIK:
    def __init__(self):
        rospy.init_node("topic_vis", anonymous=True)

        sub_pos_th = threading.Thread(target=self.SubPosThread)
        sub_pos_th.daemon = True
        sub_pos_th.start()
        self.pose3d = np.empty((0, 3))

        print("Show Pose3d")
        self.catch_pose3d = False

        while not rospy.is_shutdown():
            rospy.spin()
        self.show_joints()

    def show_joints(self):
        # show_trajectory3d_web([temp_joint], save_path="/home/aloha-pc/interbotix_ws/src/piper_hci/temp_imag/joint_trajectory_f.png")
        show_trajectory3d_web([self.pose3d], 7026)

    def SubPosThread(self):
        # 创建订阅者，监听PosCmd类型的消息
        rospy.Subscriber("/piper_right/pin_pos_cmd", PosCmd, self.pos_cmd_callback)
        rospy.spin()

    def pos_cmd_callback(self, msg):
        # 获取PosCmd类型消息中的数据
        x = msg.x
        y = msg.y
        z = msg.z
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        if not self.catch_pose3d:
            print("Catch Pose3d")
            self.catch_pose3d = True
        self.pose3d = np.vstack((self.pose3d, np.array([x, y, z])))


if __name__ == "__main__":
    # piper_ik = C_PiperIK()
    # while True:
    #     pass
    try:
        piper_ik = C_PiperIK()
    except rospy.ROSInterruptException:
        pass
