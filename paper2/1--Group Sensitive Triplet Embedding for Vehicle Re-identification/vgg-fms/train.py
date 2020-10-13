import torch
import pickle
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from model import Siamese, TripletLoss
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR
import sys
from collections import deque
import os
from arg import opt
from functions import *
import torch.nn.functional as F


if __name__ == '__main__':

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    vgg16 = models.vgg16(pretrained=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print('weight_decay = ', opt.set_weight_decay)
    print('learning rate = ', opt.lr)

    loss_softmax = torch.nn.Softmax()

    device_ids = 0

    if opt.cuda:
        vgg16.cuda()

    vgg16.train()

    optimizer = torch.optim.Adam(vgg16.parameters(), lr=opt.lr, weight_decay=opt.set_weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    trainSet = OmniglotTrain(opt.train_path, transform=data_transforms)
    validationSet = OmniglotTest(opt.test_path, transform=transforms.ToTensor(), times=opt.times, way=opt.way)
    validationLoader = DataLoader(validationSet, batch_size=opt.way, shuffle=False, num_workers=opt.workers)
    trainLoader = DataLoader(trainSet, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    grp_center, labels = None, None
    l_inter = 0
    l_intra = 0
    clusters = 4

    for epoch in range(100):
        # Group label initial assignment by K-means
        # online cluster and update group labels
        if epoch % 2 == 0:
            for id, (img) in enumerate(trainLoader, 1):
                out = torch.zeros(img.size())
                for i in range(5):
                    out[i] = vgg16.forward(img[i])
                    # out[i] 3维 5 x 10 x vector
                    if grp_center is None:
                        grp_center, labels = kmeans(out[i])
                    else:
                        a, b = kmeans(out[i])
                        grp_center = torch.cat((grp_center, a), 0)
                        labels = torch.cat((labels, b), 0)

        for id, (img) in enumerate(trainLoader, 1):

            optimizer.zero_grad()

            # img 5-dimension eg: 5(set) x 10(samples) x 3 x 244 x 244
            batch_id = len(trainLoader) * epoch + id
            if opt.cuda:
                img = Variable(img.cuda())

            out = torch.zeros(img.size())
            for i in range(5):
                out[i] = vgg16.forward(img[i])
                # out[i] 3维 5 x 10 x vector

            # i-> class; j-> group
            for i in range(5):

                cls_center, cls_pos, cls_neg = cls_para(out, i)
                l_inter += loss_inter(cls_center, cls_pos, cls_neg)

                for j in range(clusters):
                    grp_pos, grp_neg = grp_para(grp_center[(id - 1) * 5 + i][j], labels[(id - 1) * 5 + i], out, i, j)
                    l_intra += loss_intra(grp_center[(id - 1) * 5 + i][j].expand_as(grp_pos), grp_pos, grp_neg.expand_as(grp_pos))

            loss_icv = 0.5 * (l_inter + l_intra)
            loss = opt.omega * loss_softmax() + (1 - opt.omega) * loss_icv
            loss_val += loss.item()
            loss.backward()
            optimizer.step()
            if batch_id % opt.show_every == 0:
                print('[%d]\tloss:\t%f\ttime lapsed:\t%.2f s' % (
                batch_id, loss_val / opt.show_every, time.time() - time_start))
                loss_val = 0
                time_start = time.time()





            # if batch_id % opt.save_every == 0:
            #     torch.save(vgg16.state_dict(), opt.model_path + '/model-inter-' + str(batch_id) + ".pt")
            # if batch_id % opt.test_every == 0:
            #     right = np.zeros((20))
            #     error = np.zeros((20))
            #
            #     for idd, (validation_p, validation_n) in enumerate(validationLoader, 1):
            #
            #         if opt.cuda:
            #             validation_p, validation_n = validation_p.cuda(), validation_n.cuda()
            #         validation_p, validation_n = Variable(validation_p), Variable(validation_n)
            #
            #         out_p = vgg16.forward(validation_p)
            #         out_n = vgg16.forward(validation_n)
            #
            #         output = F.pairwise_distance(out_p, out_n)
            #
            #         for j in range(10):
            #             for i in range(len(output)):
            #                 if ((output[i] - float(j) / 10 < 0) & (i % 2 == 0)) | (
            #                         (output[i] - float(j) / 10 >= 0) & (i % 2 != 0)):
            #                     right[j] = right[j] + 1
            #                 else:
            #                     error[j] = error[j] + 1
            #
            #     print('*' * 70)
            #     for i in range(10):
            #         print('[%f]\tValidation set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
            #         float(i) / 10, right[i], error[i], right[i] * 1.0 / (right[i] + error[i])))
            #     print('*' * 70)
            #     print('Learning Rate: ', opt.lr)
            #     print('*' * 70)
        scheduler.step()
        f = open('/seu_share/home/zhangjinxia/jxseu/yk/backup_V1/lr.txt', 'a+')
        f.write(str(epoch) + ' ' + str(optimizer.param_groups[0]['lr']))
        f.write('\n')
        f.close()
        # if batch_id % 9000 == 0:
        #     queue.append(right[5] * 1.0 / (right[5] + error[5]))
        #     l1 = len(queue)
        #     count = 0
        #     if l1 >= 20:
        #         for j1 in range(19):
        #             if queue[l1 - 20 + j1] >= queue[l1 - 20 + j1 + 1]:
        #                 count = count + 1
        #     if count == 20:
        #         # torch.save(vgg16.state_dict(), opt.model_path + '/model-inter-' + str(batch_id + 1) + ".pt")
        #         break

    #     train_loss.append(loss_val)
    # #  learning_rate = learning_rate * 0.95

    # with open('train_loss', 'wb') as f:
    #     pickle.dump(train_loss, f)
    #
    # acc = 0.0
    # for d in queue:
    #     acc += d
    # print("#"*70)
    # print("final accuracy: ", acc/20)
