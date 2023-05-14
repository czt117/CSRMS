import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, ImageFolder, DatasetFolder, utils
import torch
import os
import os.path
import logging
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb
import io
import scipy.io as matio


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


class CIFAR10_dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sample_k=1):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.sample_k = sample_k
        self.data, self.target = self.__build_truncated_dataset__()
        self.posi_indexs = np.load('./samples/CIFAR10/posi.npy', allow_pickle=True)
        self.nega_indexs = np.load('./samples/CIFAR10/nega.npy', allow_pickle=True)
        self.cluster_weights=np.load('./samples/CIFAR10/cluster_weights.npy', allow_pickle=True)
        self.class_prototypes = np.load('./samples/CIFAR10/class_prototypes.npy', allow_pickle=True)

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        return data, target

    def _sample_posi(self, index):
        posi_index = self.posi_indexs[index]
        #         print()
        posi_data_index = np.random.choice(posi_index, self.sample_k)

        return posi_data_index

    def _sample_nega(self, index):
        nega_index = self.nega_indexs[index]
        #         print()
        nega_data_index = np.random.choice(nega_index, self.sample_k)

        return nega_data_index

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if self.train==False:
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        else:
            posi_data_index = self._sample_posi(index)
            nega_data_index = self._sample_nega(index)
            posi_data = self.data[posi_data_index]
            nega_data = self.data[nega_data_index]
            cluster_weight=self.cluster_weights[index]
            class_prototype=self.class_prototypes[index]

            if self.transform is not None:
                posi_data = torch.cat([self.transform(posi_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                nega_data = torch.cat([self.transform(nega_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                img = self.transform(img)

            return img, target, posi_data, nega_data, cluster_weight,class_prototype

    def __len__(self):
        return len(self.data)


class CIFAR100_dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sample_k=1):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.sample_k = sample_k
        self.data, self.target = self.__build_truncated_dataset__()
        self.posi_indexs = np.load('./samples/CIFAR100/posi.npy', allow_pickle=True)
        self.nega_indexs = np.load('./samples/CIFAR100/nega.npy', allow_pickle=True)
        self.cluster_weights=np.load('./samples/CIFAR100/cluster_weights.npy', allow_pickle=True)
        self.class_prototypes = np.load('./samples/CIFAR100/class_prototypes.npy', allow_pickle=True)

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        return data, target

    def _sample_posi(self, index):
        posi_index = self.posi_indexs[index]
        #         print()
        posi_data_index = np.random.choice(posi_index, self.sample_k)
        return posi_data_index

    def _sample_nega(self, index):
        nega_index = self.nega_indexs[index]
        #         print()
        nega_data_index = np.random.choice(nega_index, self.sample_k)
        return nega_data_index

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if self.train==False:
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        else:
            posi_data_index = self._sample_posi(index)
            nega_data_index = self._sample_nega(index)
            posi_data = self.data[posi_data_index]
            nega_data = self.data[nega_data_index]
            cluster_weight=self.cluster_weights[index]
            class_prototype=self.class_prototypes[index]

            if self.transform is not None:
                posi_data = torch.cat([self.transform(posi_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                nega_data = torch.cat([self.transform(nega_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                img = self.transform(img)

            return img, target, posi_data, nega_data,  cluster_weight,class_prototype

    def __len__(self):
        return len(self.data)


class Vireo172_dataset(torch.utils.data.Dataset):
    def __init__(self, image_path=None, data_path=None, transform=None, loader=default_loader, mode=None,sample_k=1):
        if mode == 'train':
            with io.open(data_path + 'TR.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'train_label.mat')['train_label'][0]
        elif mode == 'test':
            with io.open(data_path + 'TE.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'test_label.mat')['test_label'][0]
        elif mode == 'val':
            with io.open(data_path + 'VAL.txt', encoding='utf-8') as file:
                path_to_images = file.read().split('\n')[:-1]
            labels = matio.loadmat(data_path + 'val_label.mat')['validation_label'][0]
        else:
            assert 1 < 0, 'Please fill mode with any of train/val/test to facilitate dataset creation'

        # import ipdb; ipdb.set_trace()

        self.image_path = image_path
        self.path_to_images = path_to_images
        self.labels = np.array(labels, dtype=int)
        self.transform = transform
        self.loader = loader
        self.mode=mode
        self.sample_k=sample_k
        self.posi_indexs = np.load('./samples/CIFAR100/posi.npy', allow_pickle=True)
        self.nega_indexs = np.load('./samples/CIFAR100/nega.npy', allow_pickle=True)
        self.cluster_weights = np.load('./samples/CIFAR100/cluster_weights.npy', allow_pickle=True)
        self.class_prototypes = np.load('./samples/CIFAR100/class_prototypes.npy', allow_pickle=True)

    def _get_data(self):
        datas=[]
        for i in range(len(self.path_to_images)):
            path = self.path_to_images[i]
            img = self.loader(self.image_path + path)
            datas.append(img)
            return datas

    def _sample_posi(self, index):
        posi_index = self.posi_indexs[index]
        #         print()
        posi_data_index = np.random.choice(posi_index, self.sample_k)
        return posi_data_index

    def _sample_nega(self, index):
        nega_index = self.nega_indexs[index]
        #         print()
        nega_data_index = np.random.choice(nega_index, self.sample_k)
        return nega_data_index


    def __getitem__(self, index):
        # get image matrix and transform to tensor

        datas=self._get_data()
        img=datas[index]
        target = self.labels[index]
        target -= 1  # change vireo labels from 1-indexed to 0-indexed values
        if self.mode =='test':
            if self.transform is not None:
                img = self.transform(img)
            return [img, target]
        else:
            posi_data_index = self._sample_posi(index)
            nega_data_index = self._sample_nega(index)
            posi_data = datas[posi_data_index]
            nega_data = datas[nega_data_index]
            cluster_weight = self.cluster_weights[index]
            class_prototype = self.class_prototypes[index]

            if self.transform is not None:
                posi_data = torch.cat([self.transform(posi_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                nega_data = torch.cat([self.transform(nega_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                img = self.transform(img)
            return img, target, posi_data, nega_data, cluster_weight, class_prototype

    def __len__(self):
        return len(self.path_to_images)



class nuswide_dataset(data.Dataset):
    def __init__(self, transform=None, target_transform=None, mode=None,sample_k=1):
        self.mode = mode
        if  self.mode == 'train':
            list_image = 'TR'
        elif self.mode == 'test':
            list_image = 'TE'
        path_root = '/data_NUS_WIDE/'  # path to root folder
        path_img = '/data_NUS_WIDE/NUS_images/'  # path to image folder
        path_data = path_root  # path to data folder
        path_class_name = path_data + '/NUS_labels/Concepts81.txt'  # path to the list of names for classes
        img_path_file = path_data + list_image
        if self.mode == 'train':
            img_path_label = path_data + 'NUS_train_labels'  # +'.npy' # nus-wide-128
        elif self.mode == 'test':
            img_path_label = path_data + 'NUS_test_labels'  # +'.npy' # nus-wide-128
        # load image paths
        with io.open(img_path_label, encoding='utf-8') as file:
            path_to_label = file.read().split('\n')[:-1]
        self.img_label = []
        for path in path_to_label:
            label_str = path.split(' ')
            label_str.pop()
            self.img_label.append([float(i) for i in label_str])
        self.img_label = torch.tensor(self.img_label)

        with io.open(img_path_file, encoding='utf-8') as file:
            path_to_images = file.read().split('\n')[:-1]

        self.loader = default_loader
        self.path_to_images=path_to_images
        self.image_path = path_img
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.sample_k = sample_k
        self.posi_indexs = np.load('./samples/CIFAR100/posi.npy', allow_pickle=True)
        self.nega_indexs = np.load('./samples/CIFAR100/nega.npy', allow_pickle=True)
        self.cluster_weights = np.load('./samples/CIFAR100/cluster_weights.npy', allow_pickle=True)
        self.class_prototypes = np.load('./samples/CIFAR100/class_prototypes.npy', allow_pickle=True)

    def _get_data(self):
        datas = []
        for i in range(len(self.path_to_images)):
            path = self.path_to_images[i]
            img = self.loader(self.image_path + path)
            datas.append(img)
            return datas

    def _sample_posi(self, index):
        posi_index = self.posi_indexs[index]
        #         print()
        posi_data_index = np.random.choice(posi_index, self.sample_k)
        return posi_data_index

    def _sample_nega(self, index):
        nega_index = self.nega_indexs[index]
        nega_data_index = np.random.choice(nega_index, self.sample_k)
        return nega_data_index

    def __getitem__(self, index):
        # get image matrix and transform to tensor

        datas = self._get_data()
        img = datas[index]
        target = self.img_label[index]
        target -= 1  # change vireo labels from 1-indexed to 0-indexed values
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return [img, target]
        else:
            posi_data_index = self._sample_posi(index)
            nega_data_index = self._sample_nega(index)
            posi_data = datas[posi_data_index]
            nega_data = datas[nega_data_index]
            cluster_weight = self.cluster_weights[index]
            class_prototype = self.class_prototypes[index]

            if self.transform is not None:
                posi_data = torch.cat([self.transform(posi_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                nega_data = torch.cat([self.transform(nega_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                img = self.transform(img)
            return img, target, posi_data, nega_data, cluster_weight, class_prototype

    def __len__(self):
        return len(self.path_to_images)


def get_dataloader(dataset, datadir, train_bs, test_bs, noise_level=0,sample_k=1):
    print(dataset)
    if dataset == 'cifar10':
        dl_obj = CIFAR10_dataset

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = dl_obj(datadir, train=True, transform=transform_train, download=True,sample_k=sample_k)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'cifar100':
        dl_obj = CIFAR10_dataset

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = dl_obj(datadir, train=True, transform=transform_train, download=True,sample_k=sample_k)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)


    elif dataset == 'vireo172':
        dl_obj = Vireo172_dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        image_path = '/data_vireo172/ready_chinese_food/'
        data_path = '/data_vireo172/SplitAndIngreLabel/'

        train_ds = dl_obj(image_path, data_path, transform_train, mode='train',sample_k=sample_k)
        test_ds = dl_obj(image_path, data_path, transform_test, mode='test')

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    elif dataset == 'nuswide':
        dl_obj = nuswide_dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(224, 160),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])

        train_ds = dl_obj(transform=transform_train,target_transform=None,mode='train',sample_k=sample_k)
        test_ds = dl_obj(transform=transform_test,target_transform=None,mode='train')
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=6)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=6)

    return train_dl, test_dl
