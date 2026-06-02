'''
Author       : station geyuanliu@gmail.com
Date         : 2025-02-24 11:24:45
LastEditors: 解博炜
LastEditTime: 2025-02-27
FilePath: mc_hci_process.py
Description  : 处理动捕数据和hci数据逆解算，并生成训练数据
version      : 
notion       : 
'''

import numpy as np
import pathad
import pdb
from show_traj import show_trajectory3d_web, show_trajectory3d_comp_web,show_joints_comp_web
import pandas as pd
from tqdm import tqdm
from scripts.piper_pinocchio_f import Arm_IK
from work_space_map import work_space_map


if __name__ == "__main__":
    try:
        gt_data = np.load(
            '/home/aloha-pc/interbotix_ws/src/hcipose_lite/save_data/correct_data/hand_root_sc1v2_train_gt.npz',
            # '/home/aloha-pc/interbotix_ws/src/hcipose_lite/save_data/correct_data/hand_root_sc1v2_val_gt.npz',
            allow_pickle=True,
        )
        pred_data = np.load(
            '/home/aloha-pc/interbotix_ws/src/hcipose_lite/save_data/correct_data/hand_root_sc1v2_train.npz',
            # '/home/aloha-pc/interbotix_ws/src/hcipose_lite/save_data/correct_data/hand_root_sc1v2_val.npz',
            allow_pickle=True,
        )
    except:
        raise ImportError("Detected data do not exist!")

    ground_truth_data_name = gt_data["name"]
    ground_truth_data_hand_3d = gt_data["hand_3d"]
    # detected_data_imgname = detected_data["name"]
    # detected_data_hand_3d = detected_data["hand_3d"]
    pred_data_hand_3d = pred_data["hand_3d"]
    sequence_num = len(ground_truth_data_name)
    # sequence_num = 1  # tem 仅用于测试
    data_pred_all = []
    data_gt_all = []
    error_all = np.empty((1, 0))
    log1=tqdm(range(sequence_num))
    for i in log1:
        ik_pred = Arm_IK()
        ik_gt = Arm_IK()
        temp_pred = pred_data_hand_3d[i]
        temp_gt = ground_truth_data_hand_3d[i]
        root = temp_pred[0]
        joint_pred = np.zeros((len(temp_pred), 6))
        joint_gt = np.zeros((len(temp_gt), 6))
        log1.set_description(f"Sequence-{ground_truth_data_name[i][0]}")
        log = tqdm(range(len(temp_pred)))
        for j in log:
            armpose_pred = work_space_map(root, temp_pred[j])
            joint_pred[j] = ik_pred.cal_ik_solution(
                armpose_pred[0], armpose_pred[1], armpose_pred[2], armpose_pred[3], armpose_pred[4], armpose_pred[5])
            armpose_gt = work_space_map(root, temp_gt[j])
            joint_gt[j] = ik_gt.cal_ik_solution(
                armpose_gt[0], armpose_gt[1], armpose_gt[2], armpose_gt[3], armpose_gt[4], armpose_gt[5])
            pose_error = np.linalg.norm(armpose_pred - armpose_gt)
            joint_error = np.linalg.norm(joint_pred[j] - joint_gt[j])
            log.set_description(f"Pose Error: {pose_error:.6f}, Joint Error: {joint_error:.6f}")
        error_seq = np.linalg.norm(joint_pred - joint_gt, axis=1)
        if error_all.size == 0:  # 如果 error_all 是空的
            error_all = error_seq  # 直接将 error_seq 赋值给 error_all
        else:
            error_all = np.vstack((error_all, error_seq))
        # error_all = np.vstack((error_all, error_seq))
        data_pred_all.append(joint_pred)
        data_gt_all.append(joint_gt)
    print(f"Mean Error: {np.mean(error_all)}")
    show_joints_comp_web([joint_pred, joint_gt])
    # show_trajectory3d_comp_web([temp_pred, temp_gt])
    

    # pred = {
    #     "name": ground_truth_data_name,
    #     "hand_3d": np.array(data_pred_all, dtype=object),
    # }
    # gt_d = {
    #     "name": ground_truth_data_name,
    #     "hand_3d": np.array(data_gt_all, dtype=object),
    # }


    # np.savez("/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_train_IK.npz", **pred)
    # np.savez("/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_train_gt_IK.npz", **gt_d)
