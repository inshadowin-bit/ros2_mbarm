"""
Author       : 解博炜 8894423+xie-bowei@user.noreply.gitee.com
Date         : 2025-02-26 17:20:12
LastEditors  : 解博炜 8894423+xie-bowei@user.noreply.gitee.com
LastEditTime : 2025-02-27 10:05:07
FilePath     : /piper_hci/utils/data_test.py
Description  : 测试生成的mc ik 数据
version      : 
notion       : 
"""

import numpy as np
import pdb
from show_traj import show_trajectory3d_web, show_trajectory3d_comp_web, show_joints_comp_web
from filter import Filter
from tqdm import tqdm

try:
    gt_data = np.load(
        "/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_val_gt_IK.npz",
        # "/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_train_gt_IK.npz",
        allow_pickle=True,
    )
    pred_data = np.load(
        "/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_val_IK.npz",
        # "/home/aloha-pc/interbotix_ws/src/piper_hci/dataset/hand_root_sc1v2_train_IK.npz",
        allow_pickle=True,
    )
except:
    raise ImportError("Detected data do not exist!")

error_all = np.empty((1, 0))
error_all_f = np.empty((1, 0))
seq_sum = len(gt_data["name"])
# seq_sum = 1  # tem 仅用于测试
for j in tqdm(range(seq_sum)):
    joint_pred = pred_data["hand_3d"][j]
    joint_gt = gt_data["hand_3d"][j]
    joint_pred_f = np.zeros_like(joint_pred)

    f = Filter(s=16, n=6, r=0.3)
    f_start = False

    def filter_set6(sol_q, f_start):
        if not f_start:
            # f = KalmanFilter_1d_da1_set6(x_init=sol_q, var_np=np.array([2] * 6),con_a=np.array([0.00] * 6))
            f.init_history(sol_q)
            f_start = True
            return sol_q,f_start
        else:
            # f.predict()
            sol_q_f = f.update(sol_q)
            return sol_q_f,f_start


    for i in range(len(joint_pred)):
        joint_pred_f[i],f_start = filter_set6(joint_pred[i],f_start)
    joint_pred_f = joint_pred_f.astype(np.float64)
    joint_gt = joint_gt.astype(np.float64)
    joint_pred = joint_pred.astype(np.float64)

    error_pred = np.linalg.norm(joint_pred - joint_gt, axis=1)
    error_pred_f = np.linalg.norm(joint_pred_f - joint_gt, axis=1)
    # print("error_pred:", error_pred.mean())
    # print("error_pred_f:", error_pred_f.mean())
    if error_all.size == 0:
        error_all = error_pred
        error_all_f = error_pred_f
    else:
        error_all = np.vstack((error_all, error_pred))
        error_all_f = np.vstack((error_all_f, error_pred_f))

print(f"Mean Error: {np.mean(error_all)}")
print(f"Mean Error_f: {np.mean(error_all_f)}")

show_joints_comp_web([joint_pred, joint_pred_f, joint_gt])

# pdb.set_trace()
