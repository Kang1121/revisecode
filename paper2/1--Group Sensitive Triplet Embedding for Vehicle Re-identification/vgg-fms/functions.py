import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
from arg import opt


class TripletLoss(torch.nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
                self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            # print(an_dist - ap_dist)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        # print(loss)
        return loss


def cls_para(out, i):

    # Compute l_inter via class center
    # anchor dimension = 10 x vector(eg 1024)
    cls_center = (torch.sum(out[i], dim=0) / out.shape[1]).expand_as(out[i])
    # calculate closest negative for anchor
    pos = [10000, 0, 0]  # min, j, pos
    for j in range(5):
        if j != i:
            dis = F.pairwise_distance(cls_center, out[j])
            if dis[torch.argmin(dis)] < pos[0]:
                pos[0] = dis[torch.argmin(dis)]
                pos[1] = j
                pos[2] = torch.argmin(dis)
    cls_neg = out[pos[1]][pos[2]].expand_as(out[i])
    cls_pos = out[i]

    return cls_center, cls_pos, cls_neg


def grp_para(grp_center, labels, out, i, j):

    # Compute l_intra via group center
    # grp_center dimension 10 x vector
    # label dimension 10
    grp_pos = None
    pos = [10000, 0]
    for k in range(10):
        if labels[k] == j:
            if grp_pos is None:
                grp_pos = out[i][k]
            else:
                grp_pos = torch.cat((grp_pos, out[i][k]), 0)
        else:
            dis = F.pairwise_distance(grp_center, out[i][k])
            if dis[torch.argmin(dis)] < pos[0]:
                pos[0] = dis[torch.argmin(dis)]
                pos[1] = torch.argmin(dis)
    grp_neg = out[i][pos[1]]

    return grp_pos, grp_neg


def loss_inter(anchor, positive, negative):

    loss = TripletLoss(margin=opt.alpha1)

    return loss.forward(anchor, positive, negative)


def loss_intra(anchor, positive, negative):

    loss = TripletLoss(margin=opt.alpha2)

    return loss.forward(anchor, positive, negative)


def kmeans(samplesets):

    k = KMeans(n_clusters=4).fit(samplesets)

    return k.cluster_centers_, k.labels_


target = torch.empty(3, dtype=torch.long).random_(5)
input = torch.randn(3, 5, requires_grad=True)
print(target)
print(input)