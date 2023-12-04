import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss
import numpy as np

# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy, test_loss


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    brier_scores = []
    total_brier_score = 0
    total_samples = 0

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        # for prob in avg_log_probs:
        avg_log_probs = F.softmax(log_probs, dim=1)
        one_hot_targets = F.one_hot(target, num_classes=10)
        squared_errors = (avg_log_probs - one_hot_targets.float()) ** 2
        total_brier_score += squared_errors.sum().item()
        total_samples += target.size(0)

    brier_score = total_brier_score / total_samples
    # brier_score = sum(brier_scores) / len(brier_scores)
    #print('brier_score',brier_score)
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, brier_score

def test_for_deep_ensemble(models, datatest, args):
    for model in models:
        model.eval()

    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    brier_scores = []
    total_brier_score = 0
    total_samples = 0


    for data, target in data_loader:
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()

        ensemble_log_probs = []
        for model in models:
            log_probs = model(data)
            ensemble_log_probs.append(F.softmax(log_probs, dim=1))

        # 平均模型输出
        avg_log_probs = torch.mean(torch.stack(ensemble_log_probs), dim=0)

        # 计算损失和准确率
        test_loss += F.cross_entropy(avg_log_probs, target, reduction='sum').item()
        y_pred = avg_log_probs.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.view_as(y_pred)).sum().item()
        
        # for prob in avg_log_probs:
        one_hot_targets = F.one_hot(target, num_classes=10)
        squared_errors = (avg_log_probs - one_hot_targets.float()) ** 2
        total_brier_score += squared_errors.sum().item()
        total_samples += target.size(0)

        # 计算Brier分数
        # 注意：这里我们使用平均概率，而不是独立模型的输出
        # one_hot_vector = torch.zeros_like(avg_log_probs)
        # max_index = torch.argmax(avg_log_probs, dim=1)
        # one_hot_vector[max_index] = 1
        # brier_scores.append(brier_score_loss(one_hot_vector.cpu().detach().numpy(), avg_log_probs.cpu().detach().numpy()))

    brier_score = total_brier_score / total_samples
    print("brier_score",brier_score)
    # brier_score = sum(brier_scores) / len(brier_scores)
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Brier Score: {:.4f}\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy, brier_score))

    return accuracy, test_loss, brier_score
