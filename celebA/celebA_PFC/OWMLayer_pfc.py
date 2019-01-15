# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as scio
import copy
dtype = torch.cuda.FloatTensor# run on GPU


class OWMLayer:

    def __init__(self,  shape_list, alpha, l2_reg_lambda):

        input_size = int(shape_list[0][0])
        hidden_size = int(shape_list[1][0])
        self.class_num = int(shape_list[1][1])
        self.P1 = Variable((1.0 / alpha[0]) * torch.eye(input_size).type(dtype), volatile=True)
        self.P2 = Variable((1.0 / alpha[1]) * torch.eye(hidden_size).type(dtype), volatile=True)
        self.w1 = get_weight(shape_list[0], seed=0)
        self.w2 = get_weight(shape_list[1], zeros=True)
        wordvet = scio.loadmat('wordvet.mat')
        self.word_vec = wordvet['wordvet']
        np.random.seed(1)
        self.w_context = np.random.normal(0, 1.0, [self.word_vec.shape[1], hidden_size]) / np.sqrt(self.word_vec.shape[1])
        self.w_c = np.matmul(self.word_vec, self.w_context)

        self.myAFun = nn.ReLU().cuda()

    def owm_learn(self, batch_x, batch_y, alpha_list, task_index=None): # input_(batch,input_size)
        batch_x = torch.from_numpy(batch_x)
        batch_x = Variable(batch_x.type(dtype), requires_grad=False).cuda()

        batch_y = torch.from_numpy(batch_y)
        labels = Variable(batch_y.type(dtype), requires_grad=False).cuda()

        y1 = self.myAFun(batch_x.mm(self.w1))

        # context
        norm_old = torch.norm(batch_x, 2, 1)
        w_class = torch.from_numpy(self.w_c[task_index, :])
        w_class = Variable(w_class.expand_as(y1).type(dtype), volatile=True)
        w_class = self.myAFun(w_class)
        y1 = y1*w_class
        norm_new = torch.norm(y1, 2, 1)
        g = (norm_old/norm_new).repeat(y1.size(1), 1)
        y1 = y1*g.t()

        y2 = y1.mm(self.w2)

        r = torch.mean(y1, 0, True)
        k = torch.mm(self.P2, torch.t(r))
        # c2 = 1.0/(alpha_list[0] + torch.mm(r, k))
        self.P2.sub_(torch.mm(k, torch.t(k)) / (alpha_list[2] + torch.mm(r, k)))

        if len(labels.size()) == 1:
            labels = torch.unsqueeze(labels, 1)
        e = torch.mean(y2 - labels, 0, True)
        dw2 = torch.mm(k.data, e.data)

        # Backward + Optimize
        # if learn_w_in:
        #     r = torch.mean(batch_x, 0, True)
        #     k = torch.mm(self.P1, torch.t(r))
        #     self.P1.sub_(torch.mm(k, torch.t(k)) / (alpha_list[1] + torch.mm(r, k)))
        #     delta = (torch.mean(y1, 0, True).data > 0).type(dtype)  # delta ReLU
        #     delta = Variable(delta, requires_grad=False)
        #     e = torch.mm(e, torch.t(self.w2)) * delta
        #     dw1 = (torch.mm(k.data, e.data))
        #     self.w1.data -= alpha_list[0] * dw1
        self.w2.data -= alpha_list[0] * dw2

    def predict_labels(self, batch_x, batch_y, task_index=None):
        batch_x = torch.from_numpy(batch_x)
        batch_x = Variable(batch_x.type(dtype))

        batch_y = torch.from_numpy(batch_y)
        labels = batch_y

        # Forward
        y1 = self.myAFun(batch_x.mm(self.w1))
        norm_old = torch.norm(batch_x, 2, 1)
        w_class = torch.from_numpy(self.w_c[task_index, :])
        w_class = Variable(w_class.expand_as(y1).type(dtype), volatile=True)
        w_class = self.myAFun(w_class)
        y1 = y1 * w_class
        norm_new = torch.norm(y1, 2, 1)
        g = (norm_old / norm_new).repeat(y1.size(1), 1)
        y1 = y1 * g.t()

        y_pred = y1.mm(self.w2)

        # _, predicted = torch.max(y_pred.data, 1)
        predicted = torch.round(y_pred.data)
        predicted = torch.squeeze(predicted, 1)
        correct = torch.eq(predicted.cpu(), labels).sum()
        return correct, labels.size(0)


def get_weight(shape, zeros=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if zeros is None:
        w = np.random.normal(0, 1.0, shape)
        w = torch.from_numpy(w/(np.sqrt(shape[0])))
        w = w/torch.norm(w)
    else:
        w = 0*np.random.normal(0, 1.0, shape)
        w = torch.from_numpy(w)
    return Variable(w.type(dtype), requires_grad=True)


def trans_onehot(index, batch_size_=200, class_num_=10):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y = torch.LongTensor(batch_size_, 1).random_()
    y[:] = index
    y_onehot = torch.FloatTensor(batch_size_, class_num_)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot