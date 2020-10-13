import linecache

import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import torchvision.datasets as dset
from PIL import Image


class OmniglotTrain(Dataset):

    def __init__(self, dataPath, transform=None):
        super(OmniglotTrain, self).__init__()
        np.random.seed(0)
        # self.dataset = dataset
        self.transform = transform
        self.datas, self.num_a, self.num_p, self.num_n = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        for i in range(1, 82577):
            datas[i] = []
        # agrees = [0]
        # idx = 0
        for alphaPath in os.listdir(dataPath):
            for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                filePath = os.path.join(dataPath, alphaPath, samplePath)
                s = Image.open(filePath).convert('RGB')
                s = s.resize((224, 224), Image.ANTIALIAS)
                datas[int(samplePath.split('.')[0])].append(s)
                # print(datas[int(samplePath.split('.')[0])])
        print("finish loading training dataset to memory")

        num_a = {}
        num_p = {}
        num_n = {}

        for i in range(1, 720001):
            num_a[i] = linecache.getline('/seu_share/home/zhangjinxia/jxseu/yk/AP_selection.txt', 2 * i - 1).split('/')[7].split('.')[0]
            num_p[i] = linecache.getline('/seu_share/home/zhangjinxia/jxseu/yk/AP_selection.txt', 2 * i).split('/')[7].split('.')[0]
            #print(i)
            # num_n[i] = linecache.getline('/home/yk/siamese-pytorch/omniglot/python/n_out.txt', i).split('/')[8].split('.')[0]
            # print(num_a[i])

        return datas, num_a, num_p, num_n

    def __len__(self):
        return 720000

    def __getitem__(self, index):

        # print(index)
        # print(self.num_a)
        # print(self.num_p)
        # print(self.num_n)
        index = index % 720000
        img_a = self.datas[int(self.num_a[index + 1])][0]
        img_p = self.datas[int(self.num_p[index + 1])][0]
        # print(type(img_p))
        k = random.randint(1, len(self.datas))
        while abs(k - index) < 50:
            k = random.randint(1, len(self.datas))
        img_n = self.datas[k][0]


        # print(type(img_a))

        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)
        return img_a, img_p, img_n      # , torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):

    def __init__(self, dataPath, transform=None, times=100, way=160):
        np.random.seed(1)
        super(OmniglotTest, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        for alphaPath in os.listdir(dataPath):
            # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
            datas[idx] = []
            for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                filePath = os.path.join(dataPath, alphaPath, samplePath)
                s = Image.open(filePath).convert('RGB')
                s = s.resize((224, 224), Image.ANTIALIAS)
                datas[idx].append(s)
            idx += 1
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None

        # generate image pair from same class
        self.c1 = random.randint(0, self.num_classes - 1)
        self.img1 = random.choice(self.datas[self.c1])

        if idx % 2 == 0:
            img2 = random.choice(self.datas[self.c1])
        # generate image pair from different class
        else:
            c2 = random.randint(0, self.num_classes - 1)
            while self.c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            img2 = random.choice(self.datas[c2])

        if self.transform:
            # print(type(self.img1))
            # print(type(img2))
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        # print('datas.shape = ', len(self.datas))
        # print('img1 = ', img1.shape)
        # print('img2 = ', img2.shape)
        return img1, img2


class OmniglotTest_CMC(Dataset):

    def __init__(self, dataPath, transform=None, times=100, way=160):
        np.random.seed(1)
        super(OmniglotTest_CMC, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_a, self.num_p, self.num_n = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading training dataset to memory")
        datas = {}
        for i in range(1, 196057):
            datas[i] = []
        for alphaPath in os.listdir(dataPath):
            for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                filePath = os.path.join(dataPath, alphaPath, samplePath)
                s = Image.open(filePath).convert('RGB')
                s = s.resize((105, 105), Image.ANTIALIAS)
                datas[int(samplePath.split('.')[0])].append(s)
        print("finish loading training dataset to memory")

        num_a = {}
        num_p = {}
        num_n = {}

        for i in range(1, 1001):
            num_a[i] = linecache.getline('/home/yk/桌面/workshop/code/backup_V1/mAP_query.txt', i).split('/')[7].split('.')[0]
        for i in range(1, 1000001):
            num_p[i] = linecache.getline('/home/yk/桌面/workshop/code/backup_V1/mAP_samples.txt', i).split('/')[7].split('.')[0]

        return datas, num_a, num_p, num_n

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index // self.way
        label = None
        # print(self.num_a[idx + 1])

        self.img1 = self.datas[int(self.num_a[idx + 1])][0]
        img2 = self.datas[int(self.num_p[index + 1])][0]
        label = 0

        if index % 1000 == 0:
            label = 1

        if self.transform:
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)

        return img1, img2, label


class Omniglotreal_test(Dataset):

    def __init__(self, dataPath, transform=None, times=100, way=166):
        np.random.seed(1)
        super(Omniglotreal_test, self).__init__()
        self.transform = transform
        self.times = times
        self.way = way
        self.img1 = None
        self.c1 = None
        self.datas, self.num_classes = self.loadToMem(dataPath)

    def loadToMem(self, dataPath):
        print("begin loading test dataset to memory")
        datas = {}
        idx = 0
        datas = []
        for alphaPath in os.listdir(dataPath):
            # for charPath in os.listdir(os.path.join(dataPath, alphaPath)):
                # for samplePath in os.listdir(os.path.join(dataPath, alphaPath)):
                filePath = os.path.join(dataPath, alphaPath)
                print(filePath)
                s = Image.open(filePath).convert('RGB')
                s = s.resize((105, 105), Image.ANTIALIAS)
                datas.append(s)
                # print(len(datas))
        print("finish loading test dataset to memory")
        return datas, idx

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        label = None

        # # generate image pair from same class
        # self.c1 = random.randint(0, self.num_classes - 1)
        # self.img1 = random.choice(self.datas[self.c1])
        #
        # if idx % 2 == 0:
        #     img2 = random.choice(self.datas[self.c1])
        # # generate image pair from different class
        # else:
        #     c2 = random.randint(0, self.num_classes - 1)
        #     while self.c1 == c2:
        #         c2 = random.randint(0, self.num_classes - 1)
        #     img2 = random.choice(self.datas[c2])
        # print(idx)
        # print(len(self.datas))
        self.img1 = self.datas[2 * idx]
        img2 = self.datas[2 * idx + 1]
        # print(img2.shape)

        if self.transform:
            # print(type(self.img1))
            # print(type(img2))
            img1 = self.transform(self.img1)
            img2 = self.transform(img2)
        # print('datas.shape = ', len(self.datas))
        # print('img1 = ', img1.shape)
        # print('img2 = ', img2.shape)
        return img1, img2



# test
if __name__=='__main__':
    omniglotTrain = OmniglotTrain('./images_background', 30000*8)
    print(omniglotTrain)
