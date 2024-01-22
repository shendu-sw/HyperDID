import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, discriminator, contrastor, train_loader, criterion, pseudo_criterion,adver_pseudo_criterion,env_criterion, optimizer, optimizer1, alpha = 1, beta = 1, gama=1):

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target, batch_pseudo_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_pseudo_target = batch_pseudo_target.cuda()

        optimizer.zero_grad()
        features_1, features_2, batch_pred = model(batch_data)

        env_labels = torch.cat([torch.ones(features_1.shape[0]), torch.zeros(features_2.shape[0])], 0).cuda()

        #print(env_labels.long().shape)
        #aa = torch.cat([features_1, features_2], 0)
        #print(aa.shape)
        loss_4 = env_criterion(torch.cat([features_1, features_2], 0), env_labels.long())

        pseudo_pred = discriminator(features_1)
        real_pred = contrastor(features_2)

        loss_1 = criterion(batch_pred, batch_target)

        #pseudo_pred_1 = torch.roll(pseudo_pred, 1, 0)
        #real_pred_1 = torch.roll(real_pred, 1, 0)

        pseudo_indicate = torch.eq(batch_pseudo_target, torch.roll(batch_pseudo_target, 1, 0)) * 1  + \
            torch.ne(batch_pseudo_target, torch.roll(batch_pseudo_target, 1, 0)) * (-1)
        
        real_indicate = torch.eq(batch_target, torch.roll(batch_target, 1, 0)) * 1  + \
            torch.ne(batch_target, torch.roll(batch_target, 1, 0)) * (-1)
        
        loss_2 = pseudo_criterion(pseudo_pred, torch.roll(pseudo_pred, 1, 0), pseudo_indicate)
        loss_3 = adver_pseudo_criterion(real_pred, torch.roll(real_pred, 1, 0), real_indicate)
        
        
        loss = loss_1 + alpha * loss_2 + beta * loss_3 + gama *  loss_4

        loss.backward()
        optimizer.step()

        #pseudo_pred = discriminator(features_1.detach())
        #pseu_loss = adver_pseudo_criterion(pseudo_pred, batch_pseudo_target)
        #optimizer1.zero_grad()
        #pseu_loss.backward()
        #optimizer1.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        _, _, batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre


#-------------------------------------------------------------------------------
# test model
def test_epoch(model, test_loader, criterion, optimizer):

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        _, _, batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())

    return pre
#-------------------------------------------------------------------------------


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()


#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))