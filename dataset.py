import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import os
from PIL import Image
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import torch.nn.functional as F
import torch

# def create_custom_cmap():
#     colors = plt.cm.viridis.colors  # 红、黄、蓝
#     cmap_name = 'custom_rgb'
#     return LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# 创建自定义的颜色映射
def adjust_saturation(cmap):
    # 获取颜色映射的颜色列表
    colors = cmap(np.linspace(0, 1, cmap.N))

    # 调整每个颜色的饱和度
    for i in range(len(colors)):
        r, g, b= colors[i][:3]
        #将颜色转换为HSV颜色空间
        h, s, v = matplotlib.colors.rgb_to_hsv([r, g, b])
        # 调整饱和度
       
        # 将颜色转换回RGB颜色空间
        colors[i][:3] = matplotlib.colors.hsv_to_rgb([h, s, v])

    # 创建新的颜色映射
    adjusted_cmap = LinearSegmentedColormap.from_list(f"adjusted_{cmap.name}", colors, N=cmap.N)
    
    return adjusted_cmap


def plot_3d_from_data(data_file, output_path,output_path1,imgx,imgy,imgz):
    # 读取data文件的前16行数据
    with open(data_file, 'r') as file:
        lines = file.readlines()[16:]

    # 将数据转换为NumPy数组
    data = np.array([[float(value) for value in line.split()] for line in lines])

    # 将数据转化为30x20x20的形状
    data = torch.Tensor(data).view(imgx, imgy, imgz)
    print(data.shape)
    interpolated_data = F.interpolate(data.unsqueeze(0).unsqueeze(0), 
                                      size=(64,64,128), mode='trilinear', align_corners=False)
    print(interpolated_data.shape)
    interpolated_data = interpolated_data.squeeze(0).squeeze(0)
    print(interpolated_data.shape)
    # 创建绘图
    my_colormap = plt.get_cmap('jet')

    plt.figure(facecolor='white')
    adjusted_jet_cmap = adjust_saturation(my_colormap)

    min_value = data.min()
    max_value = data.max()
    
    # 计算每个数值的相对强度
    relative_value = (data-min_value) / (max_value-min_value)
    value = (interpolated_data-interpolated_data.min())/(interpolated_data.max()-interpolated_data.min())
    # 创建颜色数组
    colors_values = np.empty(relative_value.shape, dtype=object)
    color = np.empty(value.shape, dtype=object)
    alpha = 0.8
    

    colors_values = adjusted_jet_cmap(relative_value)               
    color = adjusted_jet_cmap(value)  
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    #ax.set_axis_off()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1]))
    ax.set_xticks([0, 16, 32])
    ax.set_yticks([0,16,32])
    ax.set_zticks([0,32,64])
    ax.invert_zaxis()

    ax.voxels(relative_value, facecolors=colors_values, edgecolor=None, shade=False)
    # plt.imshow(colors_values, cmap=adjusted_jet_cmap, origin='auto')
    
    plt.savefig(output_path, dpi=600)
    plt.show()

    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    # ax.set_axis_off()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1]))
    ax.set_xticks([0, 16, 32])
    ax.set_yticks([0,16,32])
    ax.set_zticks([0,32,64])
    ax.invert_zaxis()

    ax.voxels(value, facecolors=color, edgecolor=None, shade=False)
    # plt.imshow(colors_values, cmap=adjusted_jet_cmap, origin='auto')
    plt.savefig(output_path1, dpi=600)
    plt.show()



def plot_2d_slice_from_data(imgx,imgy,imgz,data_file, output_path, slice_index, slice_axis=0,):
    # 读取data文件的前16行数据
    with open(data_file, 'r') as file:
        lines = file.readlines()[16:]

    # 将数据转换为NumPy数组
    data = np.array([[float(value) for value in line.split()] for line in lines])
    data = data.reshape(imgx, imgy, imgz)
    print(data.shape)
    # 选择切片轴
    if slice_axis == 0:
        slice_data = data[slice_index, :, :]
        xlabel, ylabel = 'Z', 'Y'
    elif slice_axis == 1:
        slice_data = data[:, slice_index, :]
        xlabel, ylabel = 'Z', 'X'
    elif slice_axis == 2:
        slice_data = data[:, :, slice_index]
        xlabel, ylabel = 'X', 'Y'
    else:
        raise ValueError("Invalid slice_axis. Choose from 0, 1, or 2.")

    # 创建绘图
    my_colormap = plt.get_cmap('jet')
    # colors = plt.cm.viridis.colors
    plt.figure(facecolor='white')
    adjusted_jet_cmap = adjust_saturation(my_colormap)
    # tecplot_small_jet_cmap = LinearSegmentedColormap.from_list("tecplot_small_jet", colors, N=256)
    # 计算最大和最小值
    min_value = slice_data.min()
    max_value = slice_data.max()

    # 计算每个数值的相对强度
    relative_value = (slice_data - min_value) / (max_value - min_value)

    # 创建颜色数组
    colors_values = adjusted_jet_cmap(relative_value)

    # colors_values = np.flipud(np.fliplr(colors_values))
    # 绘制图像
    plt.imshow(colors_values,  origin='lower')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path, dpi=600)
    plt.show()

def plot_2d_from_tensor(data_file, output_path,  slice_axis=0):
    data = data_file.squeeze().cpu().numpy() 
     # 去除多余的维度
    if len(data.shape) >=3:
        h,w,d = data.shape
        slice_index = int(w/2)
        # 选择切片轴
        if slice_axis == 0:
            data = data[slice_index, :, :]
            xlabel, ylabel = 'Z', 'Y'
        elif slice_axis == 1:
            data = data[:, slice_index, :]
            xlabel, ylabel = 'Z', 'X'
        elif slice_axis == 2:
            data = data[:, :, slice_index]
            xlabel, ylabel = 'X', 'Y'
        else:
            raise ValueError("Invalid slice_axis. Choose from 0, 1, or 2.")
    else:
        xlabel, ylabel = 'Z', 'X'
        
    slice_data = data
    # 创建绘图
    my_colormap = plt.get_cmap('jet')
    plt.figure(facecolor='white')
    adjusted_jet_cmap = adjust_saturation(my_colormap)
    # 计算最大和最小值
    min_value = slice_data.min()
    max_value = slice_data.max()

    # 计算每个数值的相对强度
    relative_value = (slice_data - min_value) / (max_value - min_value)

    colors_values = adjusted_jet_cmap(relative_value)

    df = pd.DataFrame(slice_data)
    # df_transposed = df.T
    # 保存 DataFrame 到 Excel 文件
    path = output_path+".xlsx"
    df.to_excel(path, index=False)


    plt.imshow(colors_values,  origin='lower')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(output_path, dpi=600)
    plt.show()


def plot_3d_from_tensor(tensor_data, output_path):
    # print(len(tensor_data.shape))
    # c, h, w, d = tensor_data.shape
    # for i in range(batch_size):
        # 获取单个样本的数据
    #print(tensor_data.squeeze().cpu().shape)
    h,w,d = tensor_data.squeeze().cpu().shape
    data = tensor_data.squeeze().cpu().numpy()  # 去除多余的维度

    # 创建绘图
    my_colormap = plt.get_cmap('jet')
    plt.figure(facecolor='white')
    adjusted_jet_cmap = adjust_saturation(my_colormap)
        # 计算最大和最小值
    min_value = data.min()
    max_value = data.max()

        # 计算每个数值的相对强度
    relative_value = (data - min_value) / (max_value - min_value)

        # 创建颜色数组
    # colors_values = my_colormap(relative_value)
    colors_values = adjusted_jet_cmap(relative_value)
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    # ax.set_axis_off()
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 1, 1]))
    ax.set_xticks([0, h/2,h])
    ax.set_yticks([0, w/2,w])
    ax.set_zticks([0,d/2,d])
    ax.invert_zaxis()
        # 设置坐标轴标签和刻度
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

        # 绘制图像
    ax.voxels(relative_value, facecolors=colors_values, edgecolor=None, shade=False)

        # 保存图像
    output_path = f"{output_path}.png"
    plt.savefig(output_path, dpi=600)
    plt.close()


if __name__ == '__main__':

    data_file = '/mnt/HDD2/hlz/sr3_drnn/all_test/32_64_test/20down32-32/fluent-1031-LES-0.120250-32-32-64.dat'
    output_path = '/mnt/HDD2/hlz/sr3_drnn/output12_31.png'
    output_path1 = '/mnt/HDD2/hlz/sr3_drnn/output12_32.png'
    plot_3d_from_data(data_file, output_path,output_path1,32,32,64)
    plot_2d_slice_from_data(32,32,64,data_file, 
                             '/mnt/HDD2/hlz/sr3_drnn/output12_33.png', slice_index=16,slice_axis=1)