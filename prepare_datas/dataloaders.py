import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import CIFAR10
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable


def default_loader(image_path):
    return Image.open(image_path).convert('RGB')


class cifar10_dataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, sample_k=1):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.sample_k = sample_k
        self.data, self.target = self.__build_truncated_dataset__()
        self.posi_indexs = np.load('./samples/cifar10/posi_samples.npy', allow_pickle=True)
        self.nega_indexs = np.load('./samples/cifar10/nega_samples.npy', allow_pickle=True)
        self.curriculum_estimation = np.load('./samples/cifar10/easy_mid_hard.npy', allow_pickle=True)
        self.cluster_weights=np.load('./samples/cifar10/cluster_weights.npy', allow_pickle=True)
        self.class_prototypes = np.load('./samples/cifar10/class_prototypes.npy', allow_pickle=True)

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
        posi_data_index = np.random.choice(posi_index, self.sample_k)

        return posi_data_index

    def _sample_nega(self, index):
        nega_index = self.nega_indexs[index]
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
            curriculum_estimation=np.array([self.curriculum_estimation[index] for i in range(self.sample_k)])
            cluster_weight=self.cluster_weights[index]
            class_prototype=self.class_prototypes[index]

            if self.transform is not None:
                posi_data = torch.cat([self.transform(posi_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                nega_data = torch.cat([self.transform(nega_data[i]).unsqueeze(0) for i in range(self.sample_k)], dim=0)
                img = self.transform(img)

            return img, target, posi_data, nega_data, cluster_weight,class_prototype,curriculum_estimation

    def __len__(self):
        return len(self.data)





def get_dataloader(dataset, datadir, train_bs, test_bs, noise_level=0,sample_k=1):
    print(dataset)
    if dataset == 'cifar10':
        dl_obj = cifar10_dataset

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


    return train_dl, test_dl
