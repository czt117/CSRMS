import torch
from utils import *
from graph.calcu_graph import *

config = get_train_config()
device = torch.device(config.device if torch.cuda.is_available() else "cpu")

def test_epoch(feature_model, epoch, model, data_loader):
    acc1s = []
    acc5s = []
    feature_model=feature_model.to(device)
    model=model.to(device)
    feature_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(data_loader):
            img = img.float().to(device)
            target = target.to(device)

            features, feature_pred = feature_model(img)
            cat_feature = torch.cat([features, torch.cat([features] * config.sample_number, dim=0)], dim=0)
            graph = load_graph(cat_feature).to(device)
            _, batch_pred = model(cat_feature, graph)
            batch_pred = batch_pred[:config.batch_size, :]

            acc1, acc5 = accuracy(batch_pred, target, topk=(1, 5))
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    print('Test Epoch: {:03d},   Acc@1: {:.2f}, Acc@5: {:.2f}'.
          format(epoch, acc1, acc5))
    return acc1,model
