import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
# from thop import profile
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import models

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)

parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200)# salinas=204;pc=102,ip=200
parser.add_argument("-w","--class_num",type = int, default =16)
parser.add_argument("-m","--test_class_num",type=int, default=16)

args = parser.parse_args(args=[])
nDataSet =5
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
punish_spa = 0.1
punish_dul=0.1
punish_recon = 0.1
dataset = 'IP'
mask_number_pre = 60
mask_number = 20

test_data = 'datasets/IP/indian_pines_corrected.mat'
test_label = 'datasets/IP/indian_pines_gt.mat'

seeds = [1337,1224,1226,1235,1233]

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()
# load source domain data set
with open(os.path.join('datasets',  'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data'] # (77592, 9, 9, 128)
labels_train = source_imdb['Labels'] # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) #(9, 9, 100, 77592)
print(source_imdb['data'].shape) # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)

# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = train_loader.__iter__().__next__()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain

def spatial_spectral_mask(input,num_masks):
    batch = input.shape[0]
    if TAR_INPUT_DIMENSION>150:
        x, y, z = 2, 2, 50
    else:
        x, y, z = 2, 2, 25
    output = input.cpu().detach().numpy()
    for j in range(batch):
        for i in range(num_masks):
            mask = np.zeros((z, 2, 2))
            rand_x = np.random.randint(0, 9-x)
            rand_y = np.random.randint(0, 9-y)
            rand_z = np.random.randint(0, TAR_INPUT_DIMENSION-z)
            output[j,rand_z:rand_z+z, rand_x:rand_x+x, rand_y:rand_y+y] = output[j,rand_z:rand_z+z, rand_x:rand_x+x, rand_y:rand_y+y] * mask
    output = torch.tensor(output)
    return output.cuda()
def spectral_mask(input,num_masks):
    batch = input.shape[0]
    output = input.cpu().detach().numpy()
    x, y, z = 9,9,2
    for j in range(batch):
        for i in range(num_masks):
            mask = np.zeros((2, 9, 9))
            rand_z = np.random.randint(0, TAR_INPUT_DIMENSION-z)
            output[j,rand_z:rand_z+z, :, :] = output[j,rand_z:rand_z+z, :, :] * mask
    output = torch.tensor(output)
    return output.cuda()
def spatial_mask(input,num_masks):
    batch = input.shape[0]
    x, y, z = 1, 1, TAR_INPUT_DIMENSION
    output = input.cpu().detach().numpy()
    for j in range(batch):
        for i in range(num_masks):
            mask = np.zeros((TAR_INPUT_DIMENSION, 1, 1))
            rand_x = np.random.randint(0, 9 - x)
            rand_y = np.random.randint(0, 9 - y)
            output[j, :, rand_x:rand_x + x, rand_y:rand_y + y] = output[j, :,rand_x:rand_x + x,rand_y:rand_y + y] * mask
    output = torch.tensor(output)
    return output.cuda()

class Stitch(nn.Module):
    def __init__(self, input_size):
        super(Stitch, self).__init__()
        self.weight = nn.Parameter(torch.eye(input_size, dtype=torch.float32))

    def forward(self, input):
        input1_reshaped = input.view(input.size(0), -1)
        output = torch.matmul(input1_reshaped, self.weight.cuda())
        input1_shape = list(input.size())
        output1 = output[:, :input1_reshaped.size(1)].view(input1_shape)

        return output1


# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class MaskClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(MaskClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(160, 320), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(320, 640),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(640, 160),
            nn.ReLU(),
            nn.Dropout(0.5)


        )
        self.mask = nn.Linear(160, 2) # 512

    def forward(self, x):
        x.register_hook(lambda grad: grad * (-1))
        x = self.layer(x)
        mask_y = nn.Softmax(dim=1)(self.mask(x))
        # print(mask_y)
        return mask_y


class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): #(1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True) #(1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2) #(1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1+x3, inplace=True) #(1,8,100,9,9)  (1,16,25,5,5)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160
        # self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM, bias=False)

    def forward(self, x): #x:(400,100,9,9)
        x = x.unsqueeze(1) # (400,1,100,9,9)
        x = self.block1(x) #(1,8,100,9,9)
        x = self.maxpool1(x) #(1,8,25,5,5)
        x = self.block2(x) #(1,16,25,5,5)
        x = self.maxpool2(x) #(1,16,7,3,3)
        x = self.conv(x) #(1,32,5,1,1)
        x = x.view(x.shape[0],-1) #(1,160)
        # y = self.classifier(x)
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.block1 = residual_block(1, 8)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(8, 16)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, bias=False)

        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, bias=False)
        # self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, bias=False)
        self.final_feat_dim = FEATURE_DIM  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)
    def forward(self, x,fea1,fea2, domain='source'):  # x
        # print(x.shape)
        if domain == 'target_query':
            x = self.target_mapping(x)
            x = x.unsqueeze(1)  # (400,1,100,9,9)
            x = self.block1(x)  # (1,8,100,9,9)
            x1 = self.maxpool1(x)  # (1,8,25,5,5)
            x = self.block2(x1+fea1)  # (1,16,25,5,5)
            x2 = self.maxpool2(x)  # (1,16,7,3,3)
            x = self.conv(x2+fea2)  # (1,32,5,1,1)
            feature = x.view(x.shape[0], -1)
            # print((feature.shape))
            output = self.classifier(feature)
        if domain == 'target_spatial':
            x = self.target_mapping(x)
            x = x.unsqueeze(1)  # (400,1,100,9,9)
            x = self.block1(x)  # (1,8,100,9,9)
            x1 = self.maxpool1(x)  # (1,8,25,5,5)
            x = self.block2(x1)  # (1,16,25,5,5)
            x2 = self.maxpool2(x)  # (1,16,7,3,3)
            x = self.conv2(x2)  # (1,32,5,1,1)
            feature = x.view(x.shape[0], -1)
            output = self.classifier(feature)

        if domain =='target' or domain == 'source':
            if domain == 'target':
                x = self.target_mapping(x)  # (45, 100,9,9)
            elif domain == 'source':
                x = self.source_mapping(x)  # (45, 100,9,9)
            x = x.unsqueeze(1)  # (400,1,100,9,9)
            x = self.block1(x)  # (1,8,100,9,9)
            x1 = self.maxpool1(x)  # (1,8,25,5,5)
            x = self.block2(x1)  # (1,16,25,5,5)
            x2 = self.maxpool2(x)  # (1,16,7,3,3)
            x = self.conv(x2)  # (1,32,5,1,1)
            feature = x.view(x.shape[0], -1)
            # print((feature.shape))
            output = self.classifier(feature)
        # else:
        #     print('error assignment')
        return feature, output

class AUTOcoder(nn.Module):
    def __init__(self):
        super(AUTOcoder, self).__init__()
        self.block1 = residual_block(1, 8)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(8, 16)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, bias=False)

        self.final_feat_dim = FEATURE_DIM  # 64+32
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.tconv = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=3, bias=False)

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm3d(8),
            nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ConvTranspose3d(in_channels=8, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm3d(8),
        )

        self.upsample1 = nn.Upsample(size=(25, 5, 5), mode='trilinear', align_corners=True)

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm3d(1),
            nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm3d(1),
        )
        self.upsample2 = nn.Upsample(size=(100, 9, 9), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(size=(TAR_INPUT_DIMENSION, 9, 9), mode='trilinear', align_corners=True)
    def forward(self, x):  # x
        x = spatial_spectral_mask(x,mask_number)
        x = self.target_mapping(x)  # (45, 100,9,9)
        x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)
        x1 = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x1)  # (1,16,25,5,5)
        x2 = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.conv(x2)  # (1,32,5,1,1)
        feature = x.view(x.shape[0], -1)  # (1,160)
          # (45, 64)

        x = feature.view(feature.shape[0], 32, 5, 1, 1)  # (1,32,5,1,1)
        x = F.relu(self.tconv(x), inplace=True)  # (1,16,7,3,3)
        # x = x + res4
        x = self.upsample1(x)
        # x = x + res3
        x = F.relu(self.tconv1(x), inplace=True)
        # x = x + res2
        x = self.upsample2(x)
        # x = x + res1
        x = F.relu(self.tconv2(x), inplace=True)
        x = self.upsample3(x)
        data_cuba = x.squeeze(1)

        return x1,x2,feature, data_cuba

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()
recon_criterion = nn.MSELoss().cuda()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


# run 10 times

acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    # model
    feature_encoder = Network()
    autocoder = AUTOcoder()
    domain_classifier = models.DomainClassifier()
    mask_classifier = MaskClassifier()
    random_layer = models.RandomLayer([args.feature_dim, args.class_num], 1024)

    feature_encoder.apply(weights_init)
    autocoder.apply(weights_init)
    domain_classifier.apply(weights_init)
    mask_classifier.apply(weights_init)

    feature_encoder.cuda()
    autocoder.cuda()
    domain_classifier.cuda()
    mask_classifier.cuda()
    random_layer.cuda()  # Random layer

    feature_encoder.train()
    autocoder.train()
    domain_classifier.train()
    mask_classifier.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    autocoder_optim = torch.optim.Adam(autocoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    mask_classifier_optim = torch.optim.Adam(mask_classifier.parameters(), lr=args.learning_rate)
    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(15000):
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.__next__()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.__next__()

        try:
            target_data, target_label = target_iter.__next__()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.__next__()

        # source domain few-shot + domain adaptation
        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''
            # get few-shot classification samples
            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().__next__() # (75,100,9,9)

            # calculate features
            fea1 = torch.empty(1,8,25,5,5)
            fea2 = torch.empty(1, 16, 7,3, 3)
            support_features, support_outputs = feature_encoder(supports.cuda(),fea1,fea2)  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.cuda(),fea1,fea2)  # torch.Size([409, 32, 7, 3, 3])
            target_features, target_outputs = feature_encoder(target_data.cuda(),fea1,fea2 ,domain='target')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss
            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''domain adaptation'''
            # calculate domain adaptation loss
            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            # set label: source 1; target 0
            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
            domain_label[:supports.shape[0] + querys.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            # total_loss = fsl_loss + domain_loss
            loss = f_loss + domain_loss  # 0.01

            # Update parameters
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]
        # target domain few-shot + domain adaptation
        else:
            '''Few-shot classification for target domain data set'''
            # get few-shot classification samples
            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

            # sample datas
            supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
            querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)

            fea1,fea2,feat, data_cub = autocoder(querys.cuda())
            recon_loss = recon_criterion(querys.cuda(), data_cub)

            stitch = Stitch(FEATURE_DIM)
            stitch1 = Stitch(5000)
            stitch2 = Stitch(1008)

            support_features, support_outputs = feature_encoder(supports.cuda(),fea1,fea2,  domain='target')  # torch.Size([409, 32, 7, 3, 3])
            query_features, query_outputs = feature_encoder(querys.cuda(),stitch1(fea1),stitch2(fea2),  domain='target_query')  # torch.Size([409, 32, 7, 3, 3])
            source_features, source_outputs = feature_encoder(source_data.cuda(),fea1,fea2, domain = 'source')  # torch.Size([409, 32, 7, 3, 3])

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto = support_features

            # fsl_loss

            logits = euclidean_metric(query_features, support_proto)
            f_loss = crossEntropy(logits, query_labels.cuda())

            '''domain adaptation'''
            features = torch.cat([support_features, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            query_mask_spatial = spatial_mask(querys, mask_number_pre)
            # query_mask_spectral = spectral_mask(querys, mask_number_pre)

            query_mask_feaspatial, _ = feature_encoder(query_mask_spatial.cuda(),fea1,fea2, domain='target_spatial')
            logits_spatial = euclidean_metric(query_mask_feaspatial, support_proto)
            f_loss2 = crossEntropy(logits_spatial, query_labels.cuda())

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
            domain_label[supports.shape[0] + querys.shape[0]:] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])

            mask_label = torch.zeros([2*feat.shape[0]]).cuda()
            mask_label[feat.shape[0]:2*feat.shape[0]] =  1
            # mask_label[2 * feat.shape[0]:] = 2

            #,query_mask_feaspectral
            mask_feature = torch.cat([query_features,query_mask_feaspatial],dim=0)

            randomlayer_out = random_layer.forward([features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])


            domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
            domain_loss = domain_criterion(domain_logits, domain_label)
            # print(domain_logits.shape,features.shape)
            mask_logits = mask_classifier(mask_feature)
            mask_loss =  crossEntropy(mask_logits.float(), mask_label.long())

            loss = f_loss + domain_loss  + punish_dul * mask_loss+ punish_spa*f_loss2 +punish_recon*recon_loss# 0.01 0.5=78;0.25=80;0.01=80

            # Update parameters
            autocoder.zero_grad()
            feature_encoder.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            domain_classifier_optim.step()
            autocoder_optim.step()
            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:  # display
            train_loss.append(loss.item())#f_spa.item(),f_spe.item(),,fsl_spa loss: {:6.4f},fsl_spe loss: {:6.4f},
            print('episode {:>3d}:  domain loss: {:6.4f},mask loss: {:6.4f}, recon loss: {:6.4f},fsl loss: {:6.4f},fsl_spa loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                domain_loss.item(),mask_loss.item(),recon_loss.item(),
                                                                                                                f_loss.item(),f_loss2.item(),
                                                                                                                total_hit / total_num,
                                                                                                                loss.item()))
        # or episode == 0
        if (episode + 1) % 1000 == 0 :
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)


            train_datas, train_labels = train_loader.__iter__().__next__()
            train_features, _ = feature_encoder(Variable(train_datas).cuda(),fea1,fea2, domain='target')  # (45, 160)

            max_value = train_features.max()  # 89.67885
            min_value = train_features.min()  # -57.92479
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(),fea1,fea2, domain='target')  # (100, 160)
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)

                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))


for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):

    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [219, 94, 86]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [219, 141, 86]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [219, 188, 86]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [203, 219, 86]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [156, 219, 86]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [109, 219, 86]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [86, 219, 110]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [86, 219, 157]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [86, 219, 203]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [86, 187, 219]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [86, 141, 219]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [86, 94, 219]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [125, 86, 219]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [172, 86, 219]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [219, 86, 219]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [219, 86, 172]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/ {}+{:.2f}+{:.2f}.png".format(dataset,TAR_INPUT_DIMENSION,OAMean))
output_path = "./records/{}+{:.2f}+{:.2f}.txt".format(dataset,TAR_INPUT_DIMENSION,OAMean)

models.record_experiment_results(train_start, train_end, test_end, OAMean, OAStd, AAMean, AAStd, kMean, kStd, AMean, AStd, output_path)