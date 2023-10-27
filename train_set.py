from utils import *
from graph.calcu_graph import *
from models.gcn import *

config = get_train_config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss().to(device)

def train_epoch(feature_model, epoch, model, data_loader, optimizer, optimizer2, alpha):
    acc1s=[]
    acc5s=[]
    losses=[]
    feature_model=feature_model.to(device)
    model=model.to(device)
    model.train()
    feature_model.train()
    optimizer = exp_lr_scheduler(optimizer, epoch, config.lr, config.lr_decay, config.decay_rate)
    optimizer2 = exp_lr_scheduler(optimizer2, epoch, config.lr_feature, config.lr_decay, config.decay_rate)

    for idx,(batch_data,batch_target,posi_data,nega_data,cluster_weight,class_prototype, curriculum_esitimation) in enumerate(data_loader):
        # data
        batch_data = normalize_data(batch_data, device, config)
        posi_data = normalize_data(posi_data, device, config)
        nega_data = normalize_data(nega_data, device, config)
        batch_target = normalize_target(batch_target, device)

        # prototype
        alpha_i=config.epoch-epoch/config.epoch
        alpha_f=config.epoch-epoch/config.epoch
        cluster_weight = cluster_weight.reshape(-1, cluster_weight.shape[2])
        class_prototype = class_prototype.reshape(-1, class_prototype.shape[2])
        coefficient=torch.tensor(curriculum_set(curriculum_esitimation,alpha_i,alpha_f))

        # training
        optimizer.zero_grad()
        optimizer2.zero_grad()
        batch_features,feature_pred = feature_model(batch_data)
        batch_features=torch.div(batch_features+cluster_weight+class_prototype,3)
        posi_features, _ = feature_model(posi_data)
        nega_features, _ = feature_model(nega_data)
        cat_feature = torch.cat([batch_features, posi_features], dim=0)
        graph = load_graph(cat_feature).to(device)
        _, batch_pred = model(cat_feature, graph)
        batch_pred = batch_pred[:config.batch_size, :]

        # loss & acc
        ce_loss = criterion(batch_pred, batch_target)
        ce_loss2=criterion(feature_pred, batch_target)
        nega_loss=nega_loss_calcu(batch_features, nega_features, config.sample_number,coefficient)
        ic_loss=inter_class_loss(batch_features,batch_target,config,device)
        total_loss=alpha*ce_loss+((1-alpha)/3)*ce_loss2+((1-alpha)/3)*nega_loss+((1-alpha)/3)*ic_loss
        total_loss.backward()
        optimizer.step()
        optimizer2.step()

        acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
        acc1s.append(acc1.item())
        acc5s.append(acc5.item())
        losses.append(total_loss.item())

    epoch_loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    print(
        'Train Epoch: {:03d},Total_Loss: {:.4f},Acc@1: {:.2f},Acc@5: {:.2f}'.
            format(epoch, epoch_loss, acc1, acc5))

