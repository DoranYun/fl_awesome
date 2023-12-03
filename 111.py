#%%
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from args import args_parser
from utils.sampling import mnist_iid, mnist_noniid
from model.update import LocalUpdate
from model.Net import CNNMnist
from model.Fed import FedAvg
from model.test import test_img
import copy

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

dict_users = mnist_noniid(dataset_train, args.num_users)

img_size = dataset_train[0][0].shape
print('img_size',img_size)
net_glob = CNNMnist(args=args).to(args.device)
print(net_glob)
net_glob.train()

w_glob = net_glob.state_dict()

loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
test_accuacy = []
brier_scores = []

if args.all_clients:
    print("Aggregation over all clients")
    w_locals = [w_glob for i in range(args.num_users)]
for iter in range(args.epochs):
    loss_locals = []
    if not args.all_clients:
        w_locals = []
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for idx in idxs_users:
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
        if args.all_clients:
            w_locals[idx] = copy.deepcopy(w)
        else:
            w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
    # update global weights
    w_glob = FedAvg(w_locals)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    # print loss
    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

    # testing
    net_glob.eval()
    acc_train, loss_train1, brier_score_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test, brier_score_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    brier_scores.append(brier_score_test)
    test_accuacy.append(acc_test.item())
    # print(test_accuacy)

#%%
# plot loss curve
# import matplotlib.pyplot as plt
# brier_scores_np = np.array(brier_scores)

# plt.figure()
# plt.plot(range(len(brier_scores)), brier_scores_np)
# plt.ylabel('brier_scores')
# plt.show()
# plt.savefig('./save/fed_Mnist_CNNMnist_{}_C{}_iid{}.png'.format(args.epochs, args.frac, args.iid))


df = pd.DataFrame(brier_scores)
df.to_csv('brier_scores_FedAvg_noniid.csv'.format(brier_scores), index = False)
# 将DataFrame保存为CSV文件
# df.to_csv('test_accuacy_B_{}_E_{}_noniid.csv'.format(args.local_bs, args.local_ep), index=False)
