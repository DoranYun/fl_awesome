#%%
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader

from args import args_parser
from utils.sampling import mnist_iid
from model.update import LocalUpdate
from model.Net import CNNMnist
from model.test import test_for_deep_ensemble

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

dict_users = mnist_iid(dataset_train, args.num_users)

img_size = dataset_train[0][0].shape
print('img_size',img_size)

loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []
test_accuacy = []
brier_scores = []

m = max(int(args.frac * args.num_users), 1)

models = []
idxs_users = np.random.choice(range(args.num_users), m, replace=False)
for idx in idxs_users:
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    _, __, model = local.train(net=CNNMnist(args=args).to(args.device))
    models.append(model)

#testing
def ensemble_predict(models, x):
    # 确保模型处于评估模式
    acc_train, loss_train1, brier_score_train = test_for_deep_ensemble(models, dataset_train, args)
    acc_test, loss_test, brier_score_test = test_for_deep_ensemble(models, dataset_test, args)

# acc_train, loss_train1, brier_score_train = test_img(net_glob, dataset_train, args)
# acc_test, loss_test, brier_score_test = test_img(net_glob, dataset_test, args)
# print("Training accuracy: {:.2f}".format(acc_train))
# print("Testing accuracy: {:.2f}".format(acc_test))
# brier_scores.append(brier_score_test)
# test_accuacy.append(acc_test.item())
    # print(test_accuacy)