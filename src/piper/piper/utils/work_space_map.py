'''
Author: 解博炜
LastEditors: 解博炜
LastEditTime: 2025-02-25
FilePath: work_space_map.py
Description: 处理动捕数据和hci数据逆解算，并生成训练数据
Version: 
Notion: 用root作为固定的坐标原点
Copyright (c) 2025 by BIT807s, All Rights Reserved. 
'''

import numpy as np
import pathad


def work_space_map(root: np.array, data: np.array):
    '''_summary_

    Args:
        root (np.array): 映射变换稳定root
        data (np.array): 3D数据 (3,0) xyz

    Returns:
        arm: np.array(6,0),x, y, z, roll, pitch, yaw
    '''
    pub_pose = np.zeros((6, 1))
    pub_pose[0] = (data[0]-0.25)*0.15/0.5 + 0.325
    pub_pose[1] = (data[1]+0.3)*2
    pub_pose[2] = (data[2]-0.25)*0.48/0.5 + 0.36
    # tag 映射不限制范围
    # pub_pose[0] = max(0.25, min(0.4, pub_pose[0]))  # 限制范围 
    # pub_pose[1] = max(-0.3, min(0.3, pub_pose[1]))
    # pub_pose[2] = max(0.12, min(0.7, pub_pose[2]))
    pub_pose[3] = 0
    pub_pose[5] = np.arctan2(pub_pose[1], pub_pose[0])
    pub_pose[4] = 1.57
    if pub_pose[2] < 0.3:
        pub_pose[4] = 2.34+(pub_pose[2]-0.12)*(1.57-2.34)/(0.3-0.12)
    return pub_pose
