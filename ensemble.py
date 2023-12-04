#%%
import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data.dataloader as dataloader
from torch.utils.data import DataLoader, Dataset

from args import args_parser
from utils.sampling import mnist_iid
from model.update import LocalUpdate, DatasetSplit
from model.Net import CNNMnist
from model.test_copy import test_for_deep_ensemble

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model.test import test_img

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
net_glob = CNNMnist(args=args).to(args.device)
net_glob.train()

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

# idxs_users = np.random.choice(range(args.num_users), m, replace=False)

models = [CNNMnist(args=args).to(args.device) for i in range(m)]
optimizers = [torch.optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum) for i in range(m)]
loss_funcs = [torch.nn.CrossEntropyLoss() for i in range(m)]
for iter in range(args.local_ep):
    
    for idx in range(m):
        net = models[idx]
        net.train()
        loss_func = loss_funcs[idx]
        optimizer = optimizers[idx]
        ldr_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=args.local_bs, shuffle=True)
        for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()   
        # print('done_{}'.format(idx))       
    
    if iter % 10 == 0:
        acc_test, loss_test, brier_score_test = test_for_deep_ensemble(models, dataset_test, args)
        brier_scores.append(brier_score_test)

#testing

    # 确保模型处于评估模式

# accuracy, test_loss, brier_score = ensemble_predict(models,dataset_train)
# acc_train, loss_train1, brier_score_train = test_for_deep_ensemble(net_glob, dataset_train, args)
# acc_test, loss_test, brier_score_test = test_for_deep_ensemble(net_glob, dataset_test, args)
# print("Training accuracy: {:.2f}".format(acc_train))
# print("Testing accuracy: {:.2f}".format(acc_test))
# brier_scores.append(brier_score_test)
# test_accuacy.append(acc_test.item())
# print(test_accuacy)

# print("brier_score_test",brier_score_test)

df = pd.DataFrame([brier_scores])
df.to_csv('brier_scores_de_6000.csv', index = False)