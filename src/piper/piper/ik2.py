'''
Author       : station geyuanliu@gmail.com
Date         : 2025-02-14 17:28:44
LastEditors: 解博炜 8894423+xie-bowei@user.noreply.gitee.com
LastEditTime: 2025-12-19 13:35:29
FilePath: piper_pinocchio_f.py
Description  : 用常规滑动滤波处理关节角度 (ROS 2 Foxy版本)
version      : 
notion       : 
'''
#!/home/saber/anaconda3/envs/ros2_arm/bin python3
import sys
sys.path.append('/home/saber/anaconda3/envs/pin/lib/python3.8/site-packages')
#sys.path = [p for p in sys.path if "python3.8" not in p or "ros" in p]
import pathad
import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
import os

#import cv2
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from piper_ctrl_single_node import PiperRosNode
from piper_msgs.msg import PosCmd  # 假设消息类型在ROS 2中保持不变
# from utils.show_traj import show_joints, show_joints_comp, show_joints_comp_web
from utils.kalman_filter import KalmanFilter_1d_da1_set6
from utils.filter import Filter

piper_control = None

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
pp_dir=os.path.dirname(parent_dir)   
sys.path.append(parent_dir)


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # URDF路径设置
        urdf_path = pp_dir+'/piper_description/urdf/piper_description.urdf'
        #urdf_path = "/home/saber/ros2_mbarm/src/piper_description/urdf/piper_description.urdf"
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7", "joint8"]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # 末端执行器坐标系设置
        q = quaternion_from_euler(0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame(
                "ee",
                self.reduced_robot.model.getJointId("joint6"),
                pin.SE3(
                    pin.Quaternion(q[3], q[0], q[1], q[2]),
                    np.array([0.0, 0.0, 0.0]),
                ),
                pin.FrameType.OP_FRAME,
            )
        )

        # 碰撞检测设置
        self.geom_model = pin.buildGeomFromUrdf(
            self.robot.model, urdf_path, pin.GeometryType.COLLISION
        )
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        # 初始化数据
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)
        self.temp_joints = []
        self.temp_joints_f = []
        self.f = Filter(s=16, n=6, r=0.1)
        self.f_start = False
        self.last_q = np.zeros(self.reduced_robot.model.nq)
        self.last_2q = np.zeros(self.reduced_robot.model.nq)
        self.last_3q = np.zeros(self.reduced_robot.model.nq)
        self.gripper = 0.0

        # 创建Casadi模型和数据用于符号计算
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # 获取手部关节ID并定义误差函数
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # 定义优化问题
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # 用于平滑处理
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)  # 用于平滑处理

        # 设置优化约束和目标
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        
        # 优化目标函数
        self.opti.minimize(
            20 * self.totalcost + 0.001 * self.regularization + 0.001 * self.smooth_cost
        )

        # 求解器设置
        opts = {"ipopt": {"print_level": 0, "max_iter": 30, "tol": 2e-4}, "print_time": False}
        self.opti.solver("ipopt", opts)

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper / 2.0, -gripper / 2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, target_pose)
        self.opti.set_value(self.var_q_last, self.init_data)  # 用于平滑处理

        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

        except Exception as e:
            print("No solution found")
            sol_q = self.last_q + (self.last_q - self.last_2q) * 0.5 + (self.last_q - self.last_3q) * 0.25

        if self.init_data is not None:
            max_diff = max(abs(self.history_data - sol_q))
            self.init_data = sol_q
            if max_diff > 45.0 / 180.0 * 3.1415:
                self.init_data = np.zeros(self.reduced_robot.model.nq)
        else:
            self.init_data = sol_q
            
        self.history_data = sol_q
        self.last_3q = self.last_2q
        self.last_2q = self.last_q
        self.last_q = sol_q
        is_collision = self.check_self_collision(sol_q, gripper)
        return sol_q, None, not is_collision

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(
            self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0)
        )
        pin.updateGeometryPlacements(
            self.robot.model, self.robot.data, self.geom_model, self.geometry_data
        )
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        return collision

    def get_ik_solution(self, x, y, z, roll, pitch, yaw):
        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous, 0)

        if get_result:
            self.temp_joints.append(sol_q)
            sol_q_f = self.filter_set6(sol_q)
            self.temp_joints_f.append(sol_q_f)
            piper_control.joint_control_piper(
                sol_q_f[0], sol_q_f[1], sol_q_f[2], sol_q_f[3], sol_q_f[4], sol_q_f[5], self.gripper
            )
        else:
            print("collision!!!")

    def cal_ik_solution(self, x, y, z, roll, pitch, yaw):
        '''
        用于计算逆解，不能控制，只用于数据集生成
        Args:
            x (_type_): _description_
            y (_type_): _description_
            z (_type_): _description_
            roll (_type_): _description_
            pitch (_type_): _description_
            yaw (_type_): _description_

        Returns:
            _type_: _description_
        '''

        q = quaternion_from_euler(roll, pitch, yaw)
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous, 0)
        
        if get_result:
            return sol_q  
        else:
            print("collision!!!")
            return sol_q

    def filter_set6(self, sol_q):
        if not self.f_start:
            self.f.init_history(sol_q)
            self.f_start = True
            return sol_q
        else:
            sol_q_f = self.f.update(sol_q)
            return sol_q_f


class C_PiperIK(Node):
    def __init__(self):
        super().__init__("inverse_solution_node")

        # 创建Arm_IK实例
        self.arm_ik = Arm_IK()

        # 创建订阅者，监听PosCmd类型的消息
        self.subscription = self.create_subscription(
            PosCmd,
            "pin_pos_cmd",
            self.pos_cmd_callback,
            10  # QoS配置
        )
        self.subscription  # 防止未使用变量警告

        self.get_logger().info("Inverse Solution Node Start := piper_pinocchio")

    def destroy_node(self):
        self.show_joints()
        super().destroy_node()

    def show_joints(self):
        temp_joint = np.array(self.arm_ik.temp_joints)
        temp_joint_f = np.array(self.arm_ik.temp_joints_f)
        # 注释：ROS 2版本中可视化代码可能需要调整
        # show_joints_comp([temp_joint, temp_joint_f],
        #                  save_path="/home/aloha-pc/interbotix_ws/src/piper_hci/temp_imag/joint_trajectory_f.png")
        # show_joints_comp_web([temp_joint, temp_joint_f], 7026)

    def pos_cmd_callback(self, msg):
        # 获取PosCmd类型消息中的数据
        x = msg.x
        y = msg.y
        z = msg.z
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        self.arm_ik.gripper = msg.gripper
        # 调用Arm_IK类的逆解函数
        self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw)


def main(args=None):
    rclpy.init(args=args)
    piper_control = PiperRosNode()
    try:
        piper_ik = C_PiperIK()
        
        # 使用多线程执行器以提高性能
        executor = MultiThreadedExecutor()
        executor.add_node(piper_ik)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            piper_ik.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()