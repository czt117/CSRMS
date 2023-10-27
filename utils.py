import os
import torch
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res



def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay, decay_rate):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (decay_rate ** (epoch // lr_decay))

    if epoch % lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def normalize_data(data,device,config):
    data=data.float().to(device)
    data=data.reshape(-1, 3, config.image_size, config.image_size)
    return data

def normalize_target(target,device):
    target=target.to(device)
    target=target.reshape(-1)
    return target

def L2(p, q):
    d = p - q
    sum_d = torch.norm(d, p=2, dim=1)
    #     ipdb.set_trace()
    loss = torch.sum(sum_d/ p.shape[0])
    return loss



def nega_loss_calcu(p,q,sample_k,weight):
    p = torch.cat([p.unsqueeze(0)] * sample_k, dim=0)
    nega_loss=0
    for i in range(p.shape[0]):
        loss=-torch.log(torch.div(torch.tensor(1.0),L2(p[i].unsqueeze(0),q[i].unsqueeze(0))+torch.tensor(1.0)))
        nega_loss+=torch.div(torch.tensor(1.0),torch.tensor(1.0)+loss)*weight[i]
    return nega_loss



def inter_class_loss(feature,target,config,device):
    numpyfeature = feature
    numpyfeature = numpyfeature.cpu().detach().numpy()
    numpytarget = target
    numpytarget = numpytarget.cpu().detach().numpy()[:, 0]
    #     ipdb.set_trace()

    class_features = []
    for i in range(config.num_classes):
        classindexs = np.where(numpytarget == i)[0]
        if len(classindexs) != 0:
            avgfeature = np.mean(numpyfeature[classindexs * config.sample_number], axis=0)
            class_features.append(torch.tensor(avgfeature).to(device))
    elosses = torch.tensor([0.0]).to(device)
    for i in range(len(class_features)):
        for j in range(len(class_features)):
            if i != j:
                eloss = -torch.log(torch.div(torch.tensor(1.0), L2(class_features[i].unsqueeze(0), class_features[j].unsqueeze(0)) + torch.tensor(1.0)))
                elosses += eloss
    elosses /= len(class_features)
    return elosses


def save_model(acc,model,set_dict):
    best_acc=0.0
    base_path = set_dict['config'].save_model_path
    name = set_dict['dataset'] + set_dict['modelname'] + '-' + str(set_dict['config'].lr) + '-' + str(set_dict['config'].lr_feature)+ \
           '-' + str(set_dict['config'].epochs) + '-' + str(set_dict['config'].lr_decay) + '.pth'
    path = os.path.join(base_path, name)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), path)


def curriculum_set(a, alpha_i, alpha_f):
    b = np.zeros_like(a, dtype=float)
    b[a == 0] = alpha_i
    b[a == 1] = alpha_f
    b[a == 2] = 1 - alpha_f
    return b


