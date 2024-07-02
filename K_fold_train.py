# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:15:46 2022

@author: WORK
"""
import numpy
import torch
import os
import numpy as np
from dataloader import plydataset, plydataset_pred
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import time
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from pathlib import Path
import torch.nn.functional as F
import datetime
import logging
from utils import test_cls, test_cls_pred, abs_loss, FocalLoss
import torch.nn as nn
import random
from TSGCNet import TSGCNet
import random
from torch.autograd import gradcheck


def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    X_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part = X[idx]
        if (i == 0) & (j == k - 1):
            X_val = X_part
        elif j == i - 1:
            X_val = X_part
        elif j == i:  ###第i折作valid
            X_test = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)  # dim=0增加行数，竖着连接
    return X_train, X_test, X_val


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# torch.cuda.set_device(2)
# path = r'/home/likehan/tooth_segmentation/data_dental/ds_challenge_ply_mesh_160159/upper/all'
path = r'./data'
file_list = os.listdir(path)
file_array = np.array(file_list)
random.shuffle(file_array)
num_classes = 14

"""-------------------------- parameters --------------------------------------"""
k_fold = 5
batch_size = 2

k = 16

"""--------------------------- create Folder ----------------------------------"""
experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)
current_time = str(datetime.datetime.now().strftime('%m-%d_%H-%M'))
file_dir = Path(str(experiment_dir) + '/TSGCNet')
file_dir.mkdir(exist_ok=True)
log_dir, checkpoints = file_dir.joinpath('logs/'), file_dir.joinpath('checkpoints')
log_dir.mkdir(exist_ok=True)
checkpoints.mkdir(exist_ok=True)

formatter = logging.Formatter('%(name)s - %(message)s')
logger = logging.getLogger("all")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
writer = SummaryWriter(file_dir.joinpath('tensorboard'))

train_accuracy = []
val_accuracy = []
test_accuracy = []

# class_prob = [76, 336, 332, 332, 310, 339, 343, 344, 343, 308, 332, 326, 337, 82]
# class_fre = [1 / 76, 1 / 336, 1 / 332, 1 / 332, 1 / 310, 1 / 339, 1 / 343, 1 / 344, 1 / 343,
#              1 / 308, 1 / 332, 1 / 326, 1 / 337, 1 / 82]

# 采样负样本的频率(第八类因为没有负样本，使用正样本采样)
# class_prob = [76, 302, 302, 302, 310, 301, 301, 300, 301, 308, 301, 303, 310, 82]
class_prob = [268, 42, 42, 42, 34, 43, 43, 44, 43, 36, 43, 41, 34, 262]

"""--------------------------- Dataloader ----------------------"""
train_dataset = plydataset("data/train")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

train_val_dataset = plydataset("data/train")
train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset = plydataset("data/val")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

model = TSGCNet(in_channels=3, output_channels=14, k=k)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# torch.cuda.set_device(3)
model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-5
)
#optimizer = torch.optim.SGD(
#    model.parameters(),
#    lr=1e-3,
#    momentum=0.9,
#    weight_decay=1e-5
#)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

"""------------------------------------- train --------------------------------"""
logger.info("------------------train------------------")
best_acc = 0
LEARNING_RATE_CLIP = 1e-5
his_loss = []
his_smotth = []
best_mAP = 0

# neg_counts_per_class = torch.zeros(14)
# Folcal loss
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')

for epoch in range(0, 200):
    # scheduler.step()
    lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
    optimizer.param_groups[0]['lr'] = lr
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        points_face, label_face, label_face_onehot, name = data

        # neg_counts_per_class += torch.sum((label_face_onehot == 0).to(torch.float32), dim=0)
        # print(label_face_onehot)
        # print(neg_counts_per_class)

        coordinate = points_face.transpose(2, 1)

        coordinate, label_face = Variable(coordinate.float()), Variable(label_face.long())
        label_face_onehot = Variable(label_face_onehot)
        coordinate, label_face, label_face_onehot = coordinate.cuda(), label_face.cuda(), label_face_onehot.cuda()
        optimizer.zero_grad()
        pred, A, feature, M, linearlayer = model(coordinate)

        label_face = label_face.view(-1, 1)[:, 0]
        target = label_face_onehot.to(torch.float32)
        # print(pred)
        # print(target)
        loss_focal = criterion(pred, target)
        # loss_DB1 = loss_func(pred, target)
        total_loss = loss_focal
        total_loss.backward()

        # check = gradcheck(loss_func, (pred, target), eps=1e-6)
        # print("梯度检查是否通过:", check)
        # for param in model.parameters():
        #     if torch.isnan(param.grad).any():
        #         print("NaN in gradients detected!")

        optimizer.step()
        his_loss.append(total_loss.cpu().data.numpy())

    # for class_idx, neg_count in enumerate(neg_counts_per_class):
    #     print(f"类别 {class_idx} 的负样本数量：{neg_count.item()}")
    # sampled_indices = sampler.sampled_indices
    # print(f"Epoch {epoch + 1}: Sampled {len(sampled_indices)} indices.")
    if epoch % 5 == 0:
        # print('total_fold =', k_fold)
        # print('k_fold = ', i_k)
        print('Learning rate: %f' % (lr))
        print("loss: %f" % (np.mean(his_loss)))
        writer.add_scalar("loss", np.mean(his_loss), epoch)
        macc, cat_acc, mfpr, mAP, cat_mAP = test_cls(model, val_loader, num_classes=14)
        macc_train, cat_acc_train, mfpr_train, mAP_train, cat_mAP_train = test_cls(model, train_val_loader,
                                                                                    num_classes=14)
        print("Epoch %d, val_accuracy= %f" % (epoch, macc))
        print("train_accuracy= %f" % (macc_train))
        print("Epoch %d, val_mAP= %f" % (epoch, mAP))
        print("train_mAP= %f" % (mAP_train))
        logger.info(
            "Epoch: %d, accuracy= %f, loss= %f" % (epoch, macc, np.mean(his_loss)))
        logger.info("train_accracy= %f" % (macc_train))
        logger.info("val_mAP= %f" % (mAP))
        writer.add_scalar("accuracy", macc, epoch)

        if (mAP > best_mAP):
            best_mAP = mAP
            best_epoch = epoch
            print("best mAP: %f" % (best_mAP))
            print("cat_mAP:")
            print(cat_mAP)
            print("cat_acc:")
            print(cat_acc)
            print("FP/(FP+TN): %f" % (mfpr))

            torch.save(model.state_dict(), '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_mAP))
            best_pth = '%s/coordinate_%d_%f.pth' % (checkpoints, epoch, best_mAP)
            logger.info("cat_acc")
            logger.info(cat_acc)
            logger.info("FP/(FP+TN): %f" % (mfpr))
            logger.info("mAP: %f" % (mAP))
            logger.info("cat_mAP")
            logger.info(cat_mAP)

        # if epoch == 0:
        #     print(cat_miss_train)
        #     print(cat_miss)
        #     logger.info(cat_miss_train)
        #     logger.info(cat_miss)

        his_loss.clear()
        writer.close()

# root_path = r'/home/omnisky/data1/likehan/ply_k_fold/experiment/TSGCNet/checkpoints'
root_path = r'./experiment/TSGCNet/checkpoints'
doc_name = 'coordinate_%d_%f.pth' % (best_epoch, best_mAP)
model_path = os.path.join(root_path, doc_name)
model = TSGCNet(in_channels=3, output_channels=14, k=16)
checkpoints_model = torch.load(model_path)
model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoints_model.items()})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_accuracy.append(macc_train)
val_accuracy.append(best_acc)
test_dataset = plydataset_pred("data/test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)
# metrics_test, mIoU_test, cat_iou_test, test_loss = test_semseg_pred(model, test_loader, num_classes=15,
#                                                                     generate_ply=False)
macc_test, cat_acc_test, mfpr_test, mAP_test, cat_mAP_test = test_cls_pred(model, test_loader, num_classes=14)
test_accuracy.append(macc_test)
# logger.info('{} fold done, total fold = {}'.format(i_k, k_fold))
logger.info('train_accuracy={}, val_accuracy={}, test_accuracy={}'.format(
    macc_train, best_acc, macc_test))
# print('{} fold done, total fold = {}'.format(i_k, k_fold))
print('train_accuracy=', macc_train)
print('val_accuracy=', best_acc)
print('test_accuracy=', macc_test)

logger.info('train_mAP={}, val_mAP={}, test_mAP={}'.format(
    mAP_train, best_mAP, mAP_test))
# print('{} fold done, total fold = {}'.format(i_k, k_fold))
print('train_mAP=', mAP_train)
print('val_mAP=', best_mAP)
print('test_mAP=', mAP_test)

experiment_dir_test = Path('./result/')
experiment_dir_test.mkdir(exist_ok=True)
formatter = logging.Formatter('%(name)s - %(message)s')
logger = logging.getLogger("all")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(str(log_dir) + '/result.txt')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

train_accuracy = np.mean(train_accuracy)
val_accuracy = np.mean(val_accuracy)
test_accuracy = np.mean(test_accuracy)
print('train_accuracy_average={}'.format(train_accuracy))
print('val_accuracy_average={}'.format(val_accuracy))
print('test_accuracy_average={}'.format(test_accuracy))

logger.info("------------------result------------------")
logger.info("train_accuracy= %f" % (train_accuracy))
logger.info("val_accuracy= %f" % (val_accuracy))
logger.info("test_accuracy= %f" % (test_accuracy))
# logger.info(cat_iou)
