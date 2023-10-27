import numpy as np
import torch
import torch.nn as nn
import random
from config import get_train_config
from utils import *
from graph.calcu_graph import *
from prepare_datas.dataloaders import *
from models.resnet import *
from models.lenet import *
from models.gcn import *

def set():
    config = get_train_config()
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('cuda:', torch.cuda.is_available())
    dataset = 'cifar10'
    alpha = 0.5
    feature_model = ResNet18(num_classes=config.num_classes)
    dim = 512
    model = GCN(nfeat=dim, nhid=256, nclass=config.num_classes, dropout=0.5)
    modelname = 'ResNet18'
    print('dataset:', dataset, ',model_name:', modelname, ',epoch:', config.epoch,
          ',lr:', config.lr, ',lr-feature:', config.lr_feature, ',decay:', config.lr_decay,
          ',alpha:', alpha, ',sample_k:', config.sample_number, ',decay_rate:', config.decay_rate)
    # create optimizers
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)

    optimizer2 = torch.optim.SGD(
        params=feature_model.parameters(),
        lr=config.lr_feature,
        weight_decay=config.wd,
        momentum=0.9)

    train_dataloader, test_dataloader = get_dataloader(dataset=dataset, datadir=os.path.join(config.data_dir, config.dataset),
                                                       train_bs=config.batch_size, test_bs=config.batch_size,
                                                       sample_k=config.sample_number)
    set_dict={
        'config':config,
        'optimizer':optimizer,
        'optimizer2':optimizer2,
        'model':model,
        'feature_model':feature_model,
        'alpha':alpha,
        'train_dataloader':train_dataloader,
        'test_dataloader':test_dataloader,
        'dataset':dataset,
        'modelname':modelname
              }
    return set_dict