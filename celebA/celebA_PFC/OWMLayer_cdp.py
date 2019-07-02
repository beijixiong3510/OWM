# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as scio
import copy
dtype = torch.cuda.FloatTensor# run on GPU


class OWMLayer_cdp:

    def __init__(self,  shape_list, alpha, l2_reg_lambda, train_context):

        input_size = int(shape_list[0][0])
        contex_size = int(shape_list[1][0])
        self.class_num = int(shape_list[1][1])
        self.w_in = get_weight(shape_list[0], requires_grad=False)

        self.w1 = get_weight(shape_list[1], zeros=True)
        self.P1 = Variable((1.0 / alpha[0]) * torch.eye(int(shape_list[1][0])).type(dtype), volatile=True)

        # Context
        wordvet = scio.loadmat('wordvet.mat')
        self.word_vec = wordvet['wordvet']
        # self.word_vec = np.random.randn([200, 5000])
        self.train_context = train_context
        self.w_c = get_weight([self.word_vec.shape[1], contex_size], requires_grad=self.train_context)
        self.P_c = Variable((1.0 / alpha[0]) * torch.eye(int(self.w_c.size(0))).type(dtype), volatile=True)

        self.myAFun = nn.ReLU().cuda()
        self.criterion = nn.MSELoss().cuda()
        self.l2_reg_lambda = l2_reg_lambda

    def owm_learn(self, batch_x, batch_y, train=False, alpha_list=None, task_index=None): # input_(batch,input_size)

        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)

        context_input = torch.from_numpy(self.word_vec[task_index, :])
        context_input = torch.unsqueeze(context_input, 0)

        labels = Variable(batch_y.type(dtype), requires_grad=False).cuda()
        batch_x = Variable(batch_x.type(dtype), requires_grad=False).cuda()
        context_input = Variable(context_input.type(dtype), requires_grad=False).cuda()

        norm_old = torch.norm(batch_x, 2, 1)
        y0 = self.myAFun(batch_x.mm(self.w_in))
        # context
        context = self.myAFun(context_input.mm(self.w_c))
        batch_x = y0 * context
        g = (norm_old / torch.norm(batch_x, 2, 1)).repeat(batch_x.size(1), 1).type(dtype)
        batch_x.data *= g.data.t()
        y_pred = batch_x.mm(self.w1)

        if train:
            loss = self.criterion(y_pred, labels) + self.l2_reg_lambda*(torch.norm(context, p=1))
            loss.backward()
            # context
            if self.train_context:
                self.learning(context_input, self.P_c, self.w_c, alpha_list[0]*10, alpha_list[1])
            self.learning(batch_x, self.P1, self.w1, alpha_list[0], alpha_list[2])
        else:
            predicted = torch.round(y_pred.data)
            predicted = torch.squeeze(predicted, 1)
            correct = torch.eq(predicted.cpu(), batch_y).sum()
            return correct, batch_y.size(0)

    def learning(self, input_, Pro, weight, lr=0, alpha=1.0):
        r = torch.mean(input_, 0, True)
        k = torch.mm(Pro, torch.t(r))
        Pro.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
        weight.data -= lr * torch.mm(Pro.data, weight.grad.data)
        weight.grad.data.zero_()


def get_weight(shape, zeros=None, seed=0, requires_grad=True):
    if seed is not None:
        np.random.seed(seed)
    if zeros is None:
        w = np.random.normal(0, np.sqrt(2.0 / sum(shape)), shape)
    else:
        w = np.zeros(shape)
    w = torch.from_numpy(w)
    return Variable(w.type(dtype), requires_grad=requires_grad)


def trans_onehot(index, batch_size_=None, class_num_=None):
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y = torch.LongTensor(batch_size_, 1).random_()
    y[:] = index
    y_onehot = torch.FloatTensor(batch_size_, class_num_)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot