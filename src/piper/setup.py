'''
Author: 解博炜 8894423+xie-bowei@user.noreply.gitee.com
Date: 2025-12-15 14:28:11
LastEditors: 解博炜 8894423+xie-bowei@user.noreply.gitee.com
LastEditTime: 2025-12-17 20:44:49
FilePath: /ros2_mbarm/src/piper/setup.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from setuptools import find_packages, setup
import glob
import sys
import os
from glob import glob

package_name = 'piper'

python_version = f'{sys.version_info.major}.{sys.version_info.minor}'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 安装 launch 文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # 安装其他文件，如参数文件
        # (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'piper_single_ctrl = piper.piper_ctrl_single_node:main',
            'ik2 = piper.ik2:main',
        ],
    },
)
