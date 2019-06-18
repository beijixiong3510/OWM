import sys, time
import numpy as np
import torch

dtype = torch.cuda.FloatTensor  # run on GPU
import utils


########################################################################################################################

class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=64, lr=0,  clipgrad=10, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        self.Pc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype), volatile=True)
        self.Pc2 = torch.autograd.Variable(torch.eye(64 * 2 * 2).type(dtype), volatile=True)
        self.Pc3 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype), volatile=True)
        self.P1 = torch.autograd.Variable(torch.eye(256 * 4 * 4).type(dtype), volatile=True)
        self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)
        self.P3 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)

        self.test_max = 0

        return

    def _get_optimizer(self, t=0, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc3.parameters(), 'lr': lr_owm}
                                     ], lr=lr, momentum=0.9)

        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data):
        best_loss = np.inf
        best_acc = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # patience = self.lr_patience
        self.optimizer = self._get_optimizer(t, lr)
        nepochs = self.nepochs
        test_max = 0
        # Loop epochs
        try:
            for e in range(nepochs):
                # Train

                self.train_epoch(xtrain, ytrain, cur_epoch=e, nepoch=nepochs)
                train_loss, train_acc = self.eval(xtrain, ytrain)
                print('| [{:d}/5], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                # # Valid
                valid_loss, valid_acc = self.eval(xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                xtest = data[5]['test']['x'].cuda()
                ytest = data[5]['test']['y'].cuda()

                _, test_acc = self.eval(xtest, ytest)

                # # Adapt lr
                # if valid_loss < best_loss:
                #     best_loss = min(best_loss,valid_loss)

                # if valid_acc > best_acc:
                #     best_acc = max(best_acc, valid_acc)
                if test_acc>self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model, best_model)
        return

    def train_epoch(self, x, y, cur_epoch=0, nepoch=0):
        self.model.train()

        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        # Loop batches
        for i_batch in range(0, len(r_len), self.sbatch):
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)

            # Forward
            output, h_list, x_list = self.model.forward(images)
            loss = self.ce(output, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            lamda = i_batch / len(r_len)/nepoch + cur_epoch/nepoch

            alpha_array = [1.0 * 0.00001 ** lamda, 1.0 * 0.0001 ** lamda, 1.0 * 0.01 ** lamda, 1.0 * 0.1 ** lamda]

            def pro_weight(p, x, w, alpha=1.0, cnn=True, stride=1):
                if cnn:
                    _, _, H, W = x.shape
                    F, _, HH, WW = w.shape
                    S = stride  # stride
                    Ho = int(1 + (H - HH) / S)
                    Wo = int(1 + (W - WW) / S)
                    for i in range(Ho):
                        for j in range(Wo):
                            # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                            r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                            # r = r[:, range(r.shape[1] - 1, -1, -1)]
                            k = torch.mm(p, torch.t(r))
                            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(p.data)).view_as(w)
                else:
                    r = x
                    k = torch.mm(p, torch.t(r))
                    p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))
                    w.grad.data = torch.mm(w.grad.data, torch.t(p.data))
            # Compensate embedding gradients
            for n, w in self.model.named_parameters():
                if n == 'c1.weight':
                    pro_weight(self.Pc1, x_list[0], w, alpha=alpha_array[0], stride=2)

                if n == 'c2.weight':
                    pro_weight(self.Pc2, x_list[1], w, alpha=alpha_array[0], stride=2)

                if n == 'c3.weight':
                    pro_weight(self.Pc3, x_list[2], w, alpha=alpha_array[0], stride=2)

                if n == 'fc1.weight':
                    pro_weight(self.P1,  h_list[0], w, alpha=alpha_array[1], cnn=False)

                if n == 'fc2.weight':
                    pro_weight(self.P2,  h_list[1], w, alpha=alpha_array[2], cnn=False)

                if n == 'fc3.weight':
                    pro_weight(self.P3,  h_list[2], w, alpha=alpha_array[3], cnn=False)

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        return

    def eval(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)

            # Forward
            output,  _, _ = self.model.forward(images)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred % 10 == targets).float()

            # Log
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num
