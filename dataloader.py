from plyfile import PlyData
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, random_split
import os
import torchvision.transforms as transforms
import torch
import math
import pandas as pd
import time
from scipy.spatial import distance_matrix

labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101), (129, 81, 28), (235, 104, 163), (0, 158, 150),
          (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0),
          (255, 255, 0), (212, 22, 26), (255, 255, 255), (255, 134, 55))


# def get_data_v2(path=""):
#     labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101), (129, 81, 28), (235, 104, 163), (0, 158, 150),
#               (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0),
#               (255, 255, 0), (212, 22, 26), (255, 255, 255), (255, 134, 55))
#     row_data = PlyData.read(path)  # read ply file
#     points = np.array(pd.DataFrame(row_data.elements[0].data))
#     faces = np.array(pd.DataFrame(row_data.elements[1].data))
#     n_face = faces.shape[0]  # number of faces
#     n_points = points.shape[0]
#     xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
#     label_point = np.zeros([n_points, 1]).astype('int32')
#     label_point_onehot = np.zeros([14]).astype(('int32'))
#     """ index of faces shape=[N, 3] """
#     index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
#     norm = points[:, 3:6]
#
#     points_cn = np.concatenate((xyz, norm), axis=1).astype('float32')
#
#     """ RGB of faces shape=[N, 3] """
#     RGB_face = points[:, 6:9]
#     """ get label of each face """
#     for i, label in enumerate(labels):
#         label_point[(RGB_face == label).all(axis=1)] = i
#         # label_point_onehot[(RGB_face == label).all(axis=1), i] = 1
#         if (i in label_point) and i != 0 and i <= 14:
#             label_point_onehot[i-1] = 1
#     print(points.shape)
#     return index_face, points_cn, label_point, label_point_onehot, points

def get_data_v2(path=""):
    labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101), (129, 81, 28), (235, 104, 163), (0, 158, 150),
                     (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0),
                     (255, 255, 0), (212, 22, 26))
    # labels = ((160, 160, 160), (161, 0, 36), (129, 0, 76), (100, 0, 127), (44, 1, 138), (12, 0, 187),
    #           (144, 203, 251), (0, 67, 152), (0, 114, 114), (0, 136, 59), (193, 228, 108), (12, 226, 1), (255, 255, 0),
    #           (90, 114, 1), (121, 72, 2))
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    faces = np.array(pd.DataFrame(row_data.elements[1].data))
    n_face = faces.shape[0]  # number of faces
    n_points = points.shape[0]
    xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
    label_point = np.zeros([n_points, 1]).astype('int32')
    label_point_onehot = np.zeros([14]).astype(('int32'))
    """ index of faces shape=[N, 3] """
    index_face = np.concatenate((faces[:, 0]), axis=0).reshape(n_face, 3)
    norm = points[:, 3:6]

    points_cn = np.concatenate((xyz, norm), axis=1).astype('float32')

    """ RGB of faces shape=[N, 3] """
    RGB_face = points[:, 6:9]
    """ get label of each face """
    for i, label in enumerate(labels):
        label_point[(RGB_face == label).all(axis=1)] = i
        # label_point_onehot[(RGB_face == label).all(axis=1), i] = 1
        if (i in label_point) and i != 0 and i <= 14:
            label_point_onehot[i - 1] = 1

    # 在每个样本中，对于存在的类别，选择7个类别
    selected_classes = []
    for i in range(1, 15):  # 类别编号从1到14
        if label_point_onehot[i - 1] == 1:
            selected_classes.append(i)

    # 从选择的类别中随机选择7个类别
    selected_classes = np.random.choice(selected_classes, 7, replace=False)

    # 保留只属于选中类别的点，删除其他点
    selected_points_indices = np.where(np.isin(label_point, selected_classes) | (label_point == 0))[0]
    selected_points = points[selected_points_indices]
    selected_xyz = selected_points[:, :3]
    selected_norm = selected_points[:, 3:6]
    selected_points_cn = np.concatenate((selected_xyz, selected_norm), axis=1).astype('float32')

    # 更新 label_point_onehot
    label_point_onehot = np.zeros([14]).astype(('int32'))
    for i in selected_classes:
        label_point_onehot[i - 1] = 1

    # 更新 label_point
    selected_label_point = np.zeros([selected_points.shape[0], 1]).astype('int32')
    for i, label in enumerate(labels):
        selected_label_point[(selected_points[:, 6:9] == label).all(axis=1)] = i

    return index_face, selected_points_cn, selected_label_point, label_point_onehot, selected_points


def get_data_seg(path=""):
    row_data = PlyData.read(path)  # read ply file
    points = np.array(pd.DataFrame(row_data.elements[0].data))
    n_points = points.shape[0]
    xyz = points[:, :3]  # coordinate of vertex shape=[N, 3]
    label_point = np.zeros([n_points, 2]).astype('int32')
    """ index of faces shape=[N, 3] """
    norm = points[:, 3:6]

    points_cn = np.concatenate((xyz, norm), axis=1).astype('float32')

    """ get label of each face """
    for i in range(n_points):
        if (points[i, 6] == 1):
            label_point[i, 0] = 1
            label_point[i, 1] = 0
        else:
            label_point[i, 0] = 0
            label_point[i, 1] = 1
    return points_cn, label_point, points, xyz


class plydataset_seg(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """

    def __init__(self, path="data/train"):
        self.root_path = path
        self.file_list = os.listdir(path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        points_cn, label_point, points, xyz = get_data_seg(path=read_path)

        return points_cn, label_point, self.file_list[item], xyz


class plydataset(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """

    def __init__(self, path="data/train", patch_size=8000):
        self.root_path = path
        self.file_list = os.listdir(path)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_cn, label_point, label_point_onehot, points = get_data_v2(path=read_path)
        raw_points_cn = points_cn.copy()

        # centre
        centre = points_cn[:, :3].mean(axis=0)
        points[:, :3] -= centre
        max = points.max()
        points_cn[:, :3] = points_cn[:, :3] / max

        # normalized data
        # maxs = points[:, :3].max(axis=0)
        # mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points_cn[:, 3:].mean(axis=0)
        nstds = points_cn[:, 3:].std(axis=0)

        for i in range(3):
            # normalize coordinate
            points_cn[:, i] = (points_cn[:, i] - means[i]) / stds[i]  # point 1
            # normalize normal vector
            points_cn[:, i + 3] = (points_cn[:, i + 3] - nmeans[i]) / nstds[i]  # normal1

        positive_idx = np.argwhere(label_point > 0)[:, 0]  # tooth idx
        negative_idx = np.argwhere(label_point == 0)[:, 0]  # gingiva idx
        num_positive = len(positive_idx)

        if num_positive > self.patch_size:  # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:  # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive  # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        points_cn_selected = points_cn[selected_idx, :]
        label_point_selected = label_point[selected_idx, :]

        return points_cn_selected, label_point_selected, label_point_onehot, self.file_list[item]


class plydataset_pred(Dataset):
    """
    Input:
        path: root path
        downsample: type of downsample
        ratio: down sample scale
    Return:
        sampled_index_face: [N, 3]
        sampled_point_face: [N, 12]
        sampled_label_face: [N, 1]
    """

    def __init__(self, path="data/train", patch_size=12000):
        self.root_path = path
        self.file_list = os.listdir(path)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        read_path = os.path.join(self.root_path, self.file_list[item])
        index_face, points_cn, label_point, label_point_onehot, points = get_data_v2(path=read_path)
        raw_points_cn = points_cn.copy()

        # centre
        centre = points_cn[:, :3].mean(axis=0)
        points[:, :3] -= centre
        max = points.max()
        points_cn[:, :3] = points_cn[:, :3] / max

        # normalized data
        # maxs = points[:, :3].max(axis=0)
        # mins = points[:, :3].min(axis=0)
        means = points[:, :3].mean(axis=0)
        stds = points[:, :3].std(axis=0)
        nmeans = points_cn[:, 3:].mean(axis=0)
        nstds = points_cn[:, 3:].std(axis=0)

        for i in range(3):
            # normalize coordinate
            points_cn[:, i] = (points_cn[:, i] - means[i]) / stds[i]  # point 1
            # normalize normal vector
            points_cn[:, i + 3] = (points_cn[:, i + 3] - nmeans[i]) / nstds[i]  # normal1

        return points_cn, label_point, label_point_onehot, self.file_list[item]


if __name__ == "__main__":
    points_cn, label_point, label_point_onehot = get_data_vtk("data/vtk2/train/TP_013NUWYR_lower.vtk")
    train_dataset = plydataset("data/vtk2/train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print("ok")
