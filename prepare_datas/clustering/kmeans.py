# kmeans clustering and assigning sample weight based on cluster information

import numpy as np
from sklearn.cluster import KMeans
import logging
import os
# import ipdb
import random
import torch
import time
from tqdm import tqdm


def prepare_intermediate_folders(pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)


class KMEANS:
    def __init__(self, n_clusters, max_iter, device=torch.device("cuda:0")):

        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = torch.tensor(x[init_row.cpu().numpy().astype(int)]).cuda()
        self.centers = init_points
        while True:
            print(self.count)
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)

            if self.count == self.max_iter:
                break

            self.count += 1
        return self.labels

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        x = torch.tensor(x).cuda()
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        self.dists = dists

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        x = torch.tensor(x).cuda()
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]

            #             print('cluster_samples', cluster_samples.shape)
            #             print('centers', centers.shape)

            if len(cluster_samples.shape) == 1:
                if cluster_samples.shape[0] == 0:
                    centers = torch.cat([centers, self.centers[i].unsqueeze(0)], (0))
                else:
                    cluster_samples.reshape((-1, cluster_samples.shape[0]))
            else:
                centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    result_path = '../clustering_results/kmeans/cifar10/'
    prepare_intermediate_folders([result_path])
    cluster_mode = 1

    if cluster_mode:
        N = 50
        M = 10
        data = np.loadtxt('../feats/cifar10/feat.txt')
        label = np.loadtxt('../feats/cifar10/label.txt')

        kmeans = KMEANS(n_clusters=N, max_iter=M)
        predict_labels = kmeans.fit(data)
        predict_labels=predict_labels.cpu().detach().numpy()
        np.save(result_path + 'results_' + str(N) + '.npy', predict_labels)












































