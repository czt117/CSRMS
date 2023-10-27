import os
import os.path
import numpy as np
import logging
import torch.utils.data

def prepare_intermediate_folders(pathes):
    for path in pathes:
        if not os.path.exists(path):
            os.makedirs(path)

# global setting
tensor_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_feat = 2048
num_class = 10
sample_number = 5
clustering_algorithm = 'kmeans'


# algorithm parameters
rho_0 = 0.95
rho_1=0.95
rho_2=0.65


if clustering_algorithm=='art':
    cluster_path = './clustering_results/art/cifar10/'+str(rho_0)+'/'
    logger = logging.getLogger()
    net_name='networks.npz'
    net = np.load(cluster_path+net_name)
    data_name='data_Assign.npy'
    W, J, L, rhos = net['W'], net['J'], net['L'], net['rhos']
    data_assign = torch.tensor(np.load(cluster_path+data_name))
    data_assign=torch.squeeze(data_assign,dim=0)
    L = torch.tensor(L)


elif clustering_algorithm=='kmeans':
    N=50
    cluster_path = './clustering_results/kmeans/cifar10/'+'/results_' + str(N) + '.npy'
    data_assign = np.load(cluster_path)
    J = N
    data_path = './feats/cifar10/feat.txt'
    label_path = './feats/cifar10/label.txt'


data = np.loadtxt(data_path)
label = np.loadtxt(label_path)
num_data = data.shape[0]


# clustering results
clusters=[]
clusters_statistics=[]
cluster_numbers=[]
for j in range(J):
    cls_j = np.where(data_assign == j)[0]
    clusters.append(cls_j)
    label_cls_j = label[cls_j]
    unq, unq_cnt = np.unique(label_cls_j, return_counts=True)
    class_and_number = {unq[i]: unq_cnt[i] for i in range(len(unq))}
    clusters_statistics.append(class_and_number)
    cluster_number = 0
    for k in range(len(unq)):
        cluster_number += unq_cnt[k]
    cluster_numbers.append(cluster_number)


# positive sampling
posi_indexs=[]
for i in range(num_data):
    label_i=label[i]
    class_i = np.where(label == label[i])[0]
    sample_candidates = []
    max_cluster_index=-1
    max_cluster_number=-1
    for j in range(J):
        class_and_number=clusters_statistics[j]
        for key,value in class_and_number.items():
            if key==label_i:
                if value>=max_cluster_number:
                    max_cluster_index=j
                    max_cluster_number=value
    for k in clusters[max_cluster_index]:
        if label[k]==label_i and k!=i:
            sample_candidates.append(k)
    posi_indexs.append(np.array(tuple(sample_candidates)))
np.save('./samples/cifar10/posi_samples.npy', np.array(posi_indexs))
print("positive sampling")


# negative sampling
nega_indexs = []
for i in range(num_data):
    sample_candidates=[]
    cls_index=data_assign[i]
    cls_i=np.where(data_assign == cls_index)[0]
    for k in cls_i:
        if label[k]!=label[i]:
            sample_candidates.append(k)
    nega_indexs.append(np.array(tuple(sample_candidates)))
np.save('./samples/cifar10/nega_samples.npy', np.array(nega_indexs))
print("negative sampling")


# curriculum
e_m_h = np.zeros(num_data)
for j in range(J):
    unbalanceflag = False
    class_and_number_j = clusters_statistics[j]
    cluster_number_j = cluster_numbers[j]
    cluster_j=clusters[j]

    for key, value in class_and_number_j.items():
        ratio=value/cluster_number_j
        if ratio<=(1-rho_1):          # hard
            unbalanceflag=True
            for x in cluster_j:
                if label[x] == key:
                    e_m_h[x]=2
        elif ratio>=rho_1:            # easy
            for x in cluster_j:
                if label[x]==key:
                    e_m_h[x]=0
        else:                         # mid
            for x in cluster_j:
                if label[x]==key:
                    e_m_h[x]=1
np.save('./samples/cifar10/easy_mid_hard.npy', np.array(e_m_h))
print("curriculum construction")


# cluster weights
max_cluster_indexs=[]
for c in range(num_class):
    max_cluster_index=-1
    max_cluster_number=-1
    for j in range(J):
        class_and_number=clusters_statistics[j]
        for key,value in class_and_number.items():
            if key==c:
                if value>=max_cluster_number:
                    max_cluster_index=j
                    max_cluster_number=value
    max_cluster_indexs.append(max_cluster_index)
centers=[]
for c in range(num_class):
    tmp=np.zeros(num_feat)
    class_c=np.where(label==c)[0]
    for i in class_c:
        if data_assign[i]==max_cluster_indexs[c]:
            d=data[i]
            tmp+=data[i]
    center=tmp/len(class_c)
    centers.append(center)
cluster_weights=[]
for i in range(num_data):
    cluster_weights.append(centers[int(label[i])])
np.save('./samples/cifar10/cluster_weights.npy', np.array(cluster_weights))
print("cluster weights")


# class prototypes
class_prototypes=[[] for i in range(num_class)]
for j in range(J):
    unbalanceflag = False
    class_and_number_j = clusters_statistics[j]
    cluster_number_j = cluster_numbers[j]
    cluster_j=clusters[j]

    for key, value in class_and_number_j.items():
        ratio=value/cluster_number_j
        if ratio>=rho_2:
            for x in cluster_j:
                if label[x] == key:
                    class_prototypes[key].append(x)
centers = []
for c in range(num_class):
    tmp=np.zeros(num_feat)
    for i in class_prototypes[c]:
        d=data[i]
        tmp += data[i]
    center=tmp/len(class_prototypes[c])
    centers.append(center)
np.save('./samples/cifar10/class_prototypes.npy', np.array(centers))
print("class prototypes")

print('Ends')







