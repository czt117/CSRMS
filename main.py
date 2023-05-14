from global_set import set
from train_set import train_epoch
from test_set import test_epoch
from utils import *

set_dict=set()
for epoch in range(1,set_dict['config'].epoch+1):
    train_epoch(set_dict['feature_model'],epoch,set_dict['model'],set_dict['train_dataloader'],set_dict['optimizer'],set_dict['optimizer2'],set_dict['alpha'])
    acc,model=test_epoch(set_dict['feature_model'], epoch, set_dict['model'], set_dict['test_dataloader'])
    save_model(acc,model,set_dict)
