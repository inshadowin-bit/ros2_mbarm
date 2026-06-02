'''
Author       : 解博炜 8894423+xie-bowei@user.noreply.gitee.com
Date         : 2025-03-01 15:27:25
LastEditors  : 解博炜 8894423+xie-bowei@user.noreply.gitee.com
LastEditTime : 2025-03-01 15:28:32
FilePath     : /piper_hci/utils/Rbf_test.py
Description  : csdn介绍径向基插值函数的例程
version      : 
notion       : 
'''

import numpy as np
import matplotlib.pyplot as plt

def gen_data(x1,x2):
    y_sample = np.sin(np.pi*x1/2)+np.cos(np.pi*x1/3)
    y_all = np.sin(np.pi*x2/2)+np.cos(np.pi*x2/3)
    return y_sample, y_all

def kernel_interpolation(y_sample,x1,sig):
    gaussian_kernel = lambda x,c,h: np.exp(-(x-x[c])**2/(2*(h**2)))
    num = len(y_sample)
    w = np.zeros(num)
    int_matrix = np.asmatrix(np.zeros((num,num)))
    for i in range(num):
        int_matrix[i,:] = gaussian_kernel(x1,i,sig)  
    w = int_matrix.I * np.asmatrix(y_sample).T      
    return w

def kernel_interpolation_rec(w,x1,x2,sig):
    gkernel = lambda x,xc,h: np.exp(-(x-xc)**2/(2*(h**2)))
    num = len(x2)
    y_rec = np.zeros(num)
    for i in range(num):
        for k in range(len(w)):
            y_rec[i] = y_rec[i] + w[k]*gkernel(x2[i],x1[k],sig)
    return y_rec

if __name__ == '__main__':
    snum = 20   # control point数量
    ratio = 20  # 总数据点数量：snum*ratio
    sig = 1		# 核函数宽度
    xs = -8
    xe = 8
    x1 = np.linspace(xs,xe,snum)
    x2 = np.linspace(xs,xe,(snum-1)*ratio+1)
    y_sample, y_all = gen_data(x1,x2)
    plt.figure(1)
    w = kernel_interpolation(y_sample,x1,sig)   
    y_rec = kernel_interpolation_rec(w,x1,x2,sig)
    plt.plot(x2,y_rec,'k')
    plt.plot(x2,y_all,'r:')        
    plt.ylabel('y')
    plt.xlabel('x')
    for i in range(len(x1)):        
        plt.plot(x1[i],y_sample[i],'go',markerfacecolor='none')        
        
    plt.legend(labels=['reconstruction','original','control point'],loc='lower left')
    plt.title('kernel interpolation:$y=sin(\pi x/2)+cos(\pi x/3)$')       
    plt.savefig('temp_imag/kernel_interpolation.png')
