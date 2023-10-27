#import scipy.io as sio
import numpy as np
#from sklearn.manifold import TSNE
import io
import os
import os.path
import time
import numpy as np
import logging
import torch

def prepare_intermediate_folders(pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)


def art(M, label, rho, beta):
    '''
    % M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    %rho: 警戒参数
    %save_path_root: path to save clustering results for further analysis
    '''

    NAME = 'art'
    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function; to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    alpha = 0.01

    # has no significant impact on performance with a moderate value of [0.4,0.7]
    # beta = 0.6

    sigma = 0  # the percentage to enlarge or shrink vigilance region

    # rho needs carefully tune; used to shape the inter-cluster similarity;
    # rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    # rho = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape
    M=torch.tensor(M).cuda()
    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    print(NAME + "algorithm starts")

    #create initial cluster with the first data sample
        #initialize cluster parameters
    Wv = np.zeros((row, col))
    J = 0  # number of clusters
    L = np.zeros((1,row))  # size of clusters; note we set to the maximun number of cluster, i.e. number of rows
    Assign = np.zeros((1,row), dtype=np.int)  # the cluster assignment of objects
        #first cluster
    print('Processing data sample 0')
    Wv[0, :] = M[0, :]
    J = 1
    L[0,J-1] = 1
    Assign[0,0] = J-1 #note that python array index trickily starts from 0

    #processing other objects
    for n in range(1,row):

        print('Processing data sample %d' % n)

        T_max = -1 #the maximun choice value
        winner = -1 #index of the winner cluster

        #compute the similarity with all clusters; find the best-matching cluster
        for j in range(0,J):

            #compute the match function
            Mj_numerator_V = np.sum(np.minimum(M[n,:],Wv[j,:]))
            Mj_V = Mj_numerator_V / np.sum(M[n,:])

            if Mj_V >= rho:
                #compute choice function
                Tj = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))
                if Tj >= T_max:
                    T_max = Tj
                    winner = j

        #Cluster assignment process
        if winner == -1: #indicates no cluster passes the vigilance parameter - the rho
            #create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            L[0, J - 1] = 1
            Assign[0,n] = J - 1
        else: #if winner is found, do cluster assignment and update cluster weights
            #update cluster weights
            Wv[winner, :] = beta * np.minimum(Wv[winner, :], M[n, :]) + (1 - beta) * Wv[winner, :]
            #cluster assignment
            L[0, winner] += 1
            Assign[0,n] = winner



    print("algorithm ends")
    # Clean indexing data
    Wv = Wv[0: J, :]
    L = L[:, 0: J]


path_data_train = '../feats/cifar10/feat.txt'
path_label_train = '../feats/cifar10/label.txt'

path_data = [path_data_train]
path_labels = [path_label_train]


hidden_vector_scaled_train = np.loadtxt(path_data[0])
hidden_vector_classIDs_train = np.loadtxt(path_labels[0])

data = hidden_vector_scaled_train
label = hidden_vector_classIDs_train


rho = 0.95
beta = 5e-1

cluster_path = '../cluster_results/art/cifar10/' + str(rho) +'/'

prepare_intermediate_folders([cluster_path])

Wv, L, J, rho, Assign = art(data, label, rho, beta)
print(Assign)
for j in range(J):
    cls_j = np.where(Assign[0] == j)[0]
    print(cls_j)
    label_cls_j = label[cls_j]
    unq, unq_cnt = np.unique(label_cls_j, return_counts=True)
    tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    print(tmp)

np.savez(cluster_path + 'networks' + '.npz', W=Wv, J=J, L=L, rhos=rho)
np.save(cluster_path + 'data_Assign' + '.npy', Assign)


