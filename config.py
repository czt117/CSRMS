import argparse


def get_train_config():
    parser = argparse.ArgumentParser("mm23")

    # basic config
    parser.add_argument("--device", type=str, default='cuda:0', help='device')
    parser.add_argument("--data-dir", type=str, default='../data/', help='data folder')
    parser.add_argument("--pretrain_path", type=str, default='./pretrain/resnet18.pth', help='data folder')
    parser.add_argument("--dataset", type=str, default='cifar10',help="dataset")
    parser.add_argument("--num-classes", type=int, default=10, help="number of classes in dataset")
    parser.add_argument("--image-size", type=int, default=224,help="size of image")
    parser.add_argument("--result-dir", type=str, default='./checkpoint/', help='result path')
    parser.add_argument("--sample-k", type=int, default=5, help="number of samples")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--lr-feature", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--epoch", type=int, default=300, help='epoch')
    parser.add_argument("--lr-decay", type=int, default=30, help='decay the lr')
    parser.add_argument("--decay-rate", type=int, default=0.5, help='decay rate')


    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")  # 8->5
    parser.add_argument("--train-steps", type=int, default=10000, help="number of training/fine-tunning steps")
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')
    parser.add_argument("--wd", type=float, default=0.0, help='weight decay')  # 1e-4
    parser.add_argument("--seed", type=int, default=1, help="learning rate")
    #     parser.add_argument("--model-name", type=str, default="ViT", help='model_name')
    config = parser.parse_args()

    # model config
    #     config = eval("get_{}_config".format(config.model_arch))(config)
    # print_config(config)
    return config
