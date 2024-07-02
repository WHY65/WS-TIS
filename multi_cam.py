import numpy as np
import torch
from torch.autograd import Variable
from plyfile import PlyData
import pandas as pd
import os
from TSGCNet import TSGCNet
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

k = 16
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
filepath = r'./experiment/TSGCNet/checkpoints/coordinate_130_0.776841.pth'
model = TSGCNet(in_channels=3, output_channels=14, k=k)
checkpoints = torch.load(filepath)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoints.items()})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def get_data_v2(path=""):
    # labels = ((255, 255, 255), (255, 134, 55), (255, 125, 72), (34, 0, 34), (255, 255, 0), (0, 255, 0),
    #           (255, 0, 0), (0, 0, 0), (125, 125, 125), (0, 125, 125), (125, 0, 125), (0, 0, 125), (255, 255, 1),
    #           (88, 125, 88),
    #           (125, 0, 0))
    labels = ((160, 160, 160), (96, 25, 134), (180, 212, 101), (129, 81, 28), (235, 104, 163), (0, 158, 150),
              (125, 0, 34), (244, 152, 0), (255, 0, 255), (164, 0, 91), (0, 0, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (212, 22, 26), (255, 255, 255), (255, 134, 55))
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
    return index_face, points_cn, label_point, label_point_onehot, points, xyz


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
        index_face, points_cn, label_point, label_point_onehot, points, xyz = get_data_v2(path=read_path)

        # # centre
        # centre = points_cn[:, :3].mean(axis=0)
        # points[:, :3] -= centre
        # max = points.max()
        # points_cn[:, :3] = points_cn[:, :3] / max
        #
        # # normalized data
        # means = points[:, :3].mean(axis=0)
        # stds = points[:, :3].std(axis=0)
        # nmeans = points_cn[:, 3:].mean(axis=0)
        # nstds = points_cn[:, 3:].std(axis=0)
        #
        # for i in range(3):
        #     # normalize coordinate
        #     points_cn[:, i] = (points_cn[:, i] - means[i]) / stds[i]  # point 1
        #     # normalize normal vector
        #     points_cn[:, i + 3] = (points_cn[:, i + 3] - nmeans[i]) / nstds[i]  # normal1

        return index_face, points_cn, label_point, self.file_list[item], xyz


def generate_cam(featuremap, linearlayer, index=0):
    featuremap_select = featuremap
    weight_select = linearlayer.data
    weight_select_see = weight_select.cpu().data.numpy()
    num_points = featuremap_select.size(2)
    weight_select = weight_select.repeat(num_points, 1)
    weight_select = weight_select.permute(1, 0)
    featuremap_select_see = featuremap_select.cpu().data.numpy()
    CAM = torch.mul(featuremap_select, weight_select)
    CAM = torch.sum(CAM, dim=1, keepdim=False)
    # CAM = CAM[:, 13, :]
    # CAM = CAM.flatten()
    CAM = CAM.cpu().data.numpy()  # (1, 17, 16031)
    maxs = CAM.max(axis=1)  # (1, 17)
    mins = CAM.min(axis=1)  # (1, 17)
    CAM = (CAM - mins) / (maxs - mins)
    return CAM.flatten()


def generate_vtk_cam(index_face, points, cam, path=''):
    unique_index = np.unique(index_face.flatten())
    with open(path, "a") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS " + str(unique_index.shape[0]) + " float\n")
        for data in points:
            x, y, z = data[0], data[1], data[2]
            f.write('{} {} {}\n'.format(x, y, z))

        f.write("POLYGONS " + str(index_face.shape[0]) + " " + str((index_face.shape[0]) * 4) + "\n")

        for idx in index_face:
            a, b, c = idx[0], idx[1], idx[2]
            f.write('{} {} {} {}\n'.format(int(3), a, b, c))
        f.write('POINT_DATA ' + str(points.shape[0]) + '\n')
        f.write("SCALARS cam float\n")
        f.write("LOOKUP_TABLE camTable\n")
        for i_cam in cam:
            f.write('{}\n'.format(i_cam))


def generate_plyfile_cam(index_face, points, label_point, cam, path=" "):
    """
    Input:
        index_face: index of points in a face [N, 3]
        points_face: 3 points coordinate in a face + 1 center point coordinate [N, 12]
        label_face: label of face [N, 1]
        path: path to save new generated ply file
    Return:
    """
    unique_index = np.unique(index_face.flatten())  # get unique points index
    with open(path, "a") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write("element vertex " + str(unique_index.shape[0]) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property int label_point\n")
        f.write("property float cam\n")
        f.write("element face " + str(index_face.shape[0]) + "\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        max_cam = 0
        max_label = 0
        flag = 0
        for i in range(unique_index.shape[0]):
            xyz = points[i, 0:3]  # Get coordinate
            xyz_nor = points[i, 3:6]
            label = label_point[i]
            f.write(str(xyz[0]) + " " + str(xyz[1]) + " " + str(xyz[2]) + " " + str(xyz_nor[0]) + " "
                    + str(xyz_nor[1]) + " " + str(xyz_nor[2]) + " " + str(int(label)) + " " + str(cam[i]) + "\n")

        for i in range(index_face.shape[0]):  # write new point index for every face
            face = index_face[i]  # Get RGB value according to face label
            f.write(str(3) + " " + str(face[0]) + " " + str(face[1]) + " "
                    + str(face[2]) + "\n")
        f.close()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = r'./data/train'
    vtk_path = r'./data/vtk'
    train_dataset = plydataset_pred(path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    for batch_id, (index_face, points, label_point, name, raw_point) in tqdm(enumerate(train_loader),
                                                                             total=len(train_loader),
                                                                             smoothing=0.9):
        coordinate = points.transpose(2, 1)  # (6, 16000)
        coordinate = Variable(coordinate.float())
        coordinate = coordinate.cuda()
        raw_point = raw_point.view(-1, 3)

        index_face = index_face.view(-1, 3)
        index_face = index_face.numpy()
        points = points.view(-1, 6)
        points = points.numpy()
        label_point = label_point.view(-1, 1)
        label_point = label_point.numpy()

        with torch.no_grad():
            pred, A, feature, M_r, linearlayer = model(coordinate)
        pred = torch.sigmoid(pred)
        M = M_r[:, 13, :, :]
        linearlayer_p = linearlayer[13].weight
        # if pred[0][1] > 0.5:
        #     y_pred = 1
        #     if y_pred == label_points_onehot[0][1]:
        #         corr = 'TP'
        #     else:
        #         corr = 'FP'
        # else:
        #     y_pred = 0
        #     if y_pred == label_points_onehot[0][1]:
        #         corr = 'TN'
        #     else:
        #         corr = 'FN'
        cam = generate_cam(M, linearlayer_p, index=0)

        # 获取元组中的第一个元素作为文件名
        file_name = name[0]
        # 删除单引号和括号
        file_name = file_name.strip("('').")
        # 获取文件名前的名称
        file_name_without_extension = os.path.splitext(file_name)[0]

        new_name = '{}.ply'.format(file_name_without_extension)
        load_path = os.path.join(vtk_path, new_name)
        # generate_vtk_cam(index_face, points, cam, path=load_path)
        generate_plyfile_cam(index_face, points, label_point, cam, path=load_path)
