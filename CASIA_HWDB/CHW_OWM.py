import torch.nn as nn
from MLP_OWM.OWMLayer_two import *
import numpy as np
import scipy.io as scio
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu

class_num = 3755 # mnist
num_epochs = 50
batch_size = 50
dtype = torch.cuda.FloatTensor  # run on GPU
Path_All = '/data/CHW_mat3755/'
Data_Path_train = os.path.join(Path_All, 'train_each')
Data_Path_val = os.path.join(Path_All, 'test_each')


def my_test(class_begin=0, class_end=3755):
    correct = 0
    total = 0
    batch_size = 100
    for num_index in range(class_begin, class_end):
        mat_data = os.path.join(Data_Path_val, 'chwdata'+str(num_index))
        test = scio.loadmat(mat_data)
        testimages = test['test_data_each']
        testlabels = test['test_lable_each']
        test_length = len(testimages) # 60
        for i in range(round(test_length / batch_size)):
            start = batch_size * i
            index_end = min(start + batch_size, test_length)
            batch_x = testimages[start:index_end, :]
            batch_y = testlabels[start:index_end, :]

            accu_all = OWM.predict_labels(batch_x, batch_y)
            total += np.shape(batch_x)[0]
            correct += round(np.shape(batch_x)[0]*accu_all)
    test_accu = 100*correct / total
    return test_accu


lambda_loss = 0
middle = 4000
OWM = OWMLayer([[1024, middle], [middle, class_num]], alpha=[100.0, 100.0], l2_reg_lambda=lambda_loss)
accu_max = 0

for num_index in range(0, 3755):
    mat_data = os.path.join(Data_Path_train, 'chwdata'+str(num_index))
    train = scio.loadmat(mat_data)
    trainimages = train['train_data_each']
    trainlabels = train['train_lable_each']
    train_length = len(trainimages)
    accu_old = 0
    accu_all = 0
    flag_break = 0
    for epoch in range(num_epochs):
        ss = np.arange(train_length)
        np.random.shuffle(ss)
        trainimages = trainimages[ss, :]
        trainlabels = trainlabels[ss]
        for i in range(round(train_length/batch_size)):
            lamda = i/round(train_length/batch_size)
            start = batch_size*i
            index_end = min(start+batch_size, train_length)
            batch_x = trainimages[start:index_end, :]
            batch_y = trainlabels[start:index_end, :]
            lr_list = [2.0, 1.0*0.02**lamda, 0.5]
            loss = OWM.owm_learn(batch_x, batch_y, lr_list)
        accu_all = my_test(class_begin=num_index, class_end=num_index + 1)
        auc_delta = ((accu_all-accu_old)/(accu_old+1e-8)*100)
        if (1.0 >= auc_delta >= 0 or round(accu_all) == 100) and accu_all > 50:
            flag_break = 1
            print('Mat_number:[{:d}/{:d}], Epoch_number:[{:d}/{:d}],curr_acc:{:.2f} %'
                  .format(num_index, 3755, epoch+1, num_epochs, accu_all))
            break
        else:
            accu_old = copy.deepcopy(accu_all)
    if flag_break == 0:
        print('Mat_number:[{:d}/{:d}], Epoch_number:[{:d}/{:d}],curr_acc:{:.2f} %'
              .format(num_index, 3755, epoch, num_epochs, accu_all))

accu_test = my_test()
print('All_acc:{:.2f} %'.format(accu_test))





