"""
Author: station geyuanliu@gmail.com
Date: 2025-02-13 11:44:35
LastEditors: station geyuanliu@gmail.com
LastEditTime: 2025-02-13 14:17:36
FilePath: /piper_hci/utils/kalman_filter.py
Description: 基础kalman
version: 
notion: 
"""

import re
import numpy as np


class KalmanFilter_1d_da1:
    """
    一维的p、v、a的kf
    假设加速度导数为1
    0.03s采样间隔
    """

    def __init__(self, dim_x=3, dim_z=1, var=0.1, con_a:float=1, x_init: np.array = None):
        self.dim_x = dim_x  # 状态向量维度
        self.dim_z = dim_z  # 观测向量维度
        if x_init is not None:
            self.x = x_init
        else:
            self.x = np.zeros((dim_x, 1))  # 状态向量       
        self.P = np.eye(dim_x)  # 状态协方差矩阵
        self.A = np.array([[1, 0.03, 0], [0, 1, 0.03], [0, 0, 1]])  # 状态转移矩阵
        self.E = np.array([[0], [0], [1]])  # 状态矩阵常数项，加速度导数为1
        self.H = np.array([[1, 0, 0]])  # 观测矩阵 3x1
        self.R = np.eye(dim_z) * var  # 观测噪声协方差矩阵
        self.Q = np.eye(dim_x)  # 过程噪声协方差矩阵

    def predict(self):
        self.x = np.dot(self.A, self.x) + self.E  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),BU(k) = 0
        self.P = (
            np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        )  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=[[1,1],[0,1]]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(
            np.dot(self.P, self.H.T), np.linalg.inv(S)
        )  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        self.x = self.x + np.dot(K, y)  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)]
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # P(k|k) = (1 - Kg(k)H)P(k|k-1)
        return self.x[0, 0]

class KalmanFilter_1d_da1_set6:
    '''
    6关节独立kf,假设加速度导数为1
    '''
    def __init__(self, var_np:np.array=None , x_init: np.array = None, con_a:np.array= None):
        if var_np is not None:
            self.var = var_np
        else:
            self.var = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        
        if x_init is not None:
            self.x_init = x_init
        else:
            self.x_init = np.zeros((6, 1))

        if con_a is not None:
            self.con_a = con_a
        else:
            self.con_a = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        self.kf0 = KalmanFilter_1d_da1(var=self.var[0], x_init=self.x_init[0],con_a=self.con_a[0])
        self.kf1 = KalmanFilter_1d_da1(var=self.var[1], x_init=self.x_init[1],con_a=self.con_a[1])
        self.kf2 = KalmanFilter_1d_da1(var=self.var[2], x_init=self.x_init[2],con_a=self.con_a[2])
        self.kf3 = KalmanFilter_1d_da1(var=self.var[3], x_init=self.x_init[3],con_a=self.con_a[3])
        self.kf4 = KalmanFilter_1d_da1(var=self.var[4], x_init=self.x_init[4],con_a=self.con_a[4])
        self.kf5 = KalmanFilter_1d_da1(var=self.var[5], x_init=self.x_init[5],con_a=self.con_a[5])
    
    def predict(self):
        self.kf0.predict()
        self.kf1.predict()
        self.kf2.predict()
        self.kf3.predict()
        self.kf4.predict()
        self.kf5.predict()
    
    def update(self, z:np.array):
        x0 = self.kf0.update(z[0])
        x1 = self.kf1.update(z[1])
        x2 = self.kf2.update(z[2])
        x3 = self.kf3.update(z[3])
        x4 = self.kf4.update(z[4])
        x5 = self.kf5.update(z[5])
        return np.array([x0, x1, x2, x3, x4, x5])