from celebA_PFC.OWMLayer_pfc import *
import numpy as np
import scipy.io as scio
import os

import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1

class_num = 1  # mnist
num_epochs = 1
# batch_size = 100
dtype = torch.cuda.FloatTensor  # run on GPU
# data path
Path_All = './data/celebA_mat50/'
Data_Path_train = os.path.join(Path_All, 'train')
Data_Path_val = os.path.join(Path_All, 'test')


def my_test(class_begin=0, class_end=40):
    batch_size = 100
    acc_array = []
    for task_index in range(class_begin, class_end):
        correct = 0
        total = 0
        for num_index in range(10):
            mat_data = os.path.join(Data_Path_val, 'celebAdata'+str(num_index))
            test = scio.loadmat(mat_data)
            testimages = test['data']
            testlabels = test['lables'][:, task_index]
            test_length = len(testimages) # 60
            for i in range(math.ceil(test_length / batch_size)):
                start = batch_size * i
                index_end = min(start + batch_size, test_length)
                batch_x = testimages[start:index_end, :]
                batch_y = testlabels[start:index_end]

                correct_each, total_each = OWM.predict_labels(batch_x, batch_y,task_index)
                total += total_each
                correct += correct_each
        test_acc = 100 * correct / total
        acc_array.append(test_acc)

    return test_acc, acc_array

lambda_loss = 0
middle = 5000
for time in range(5):
    OWM = OWMLayer([[2048, middle], [middle, class_num]], alpha=[1.0, 1.0], l2_reg_lambda=lambda_loss)

    is_training = True
    accu_max = 0
    Task_array = [2, 19, 21, 31, 36]  # 1 simple 21 ;5 simple [3 20 22 32 37]
    Task_num = len(Task_array)
    if is_training:
        for task_index in Task_array:
            print("Training celebA_PFC:->>> [ {:d} / {:d} ]".format(task_index + 1, Task_num))
            for num_index in range(1):
                mat_data = os.path.join(Data_Path_train, 'celebAdata'+str(num_index))
                train = scio.loadmat(mat_data)
                Sample_Size = 200
                batch_size = 1
                # if Sample_Size < 2000:
                #     batch_size = min(max(math.ceil(Sample_Size/200), 1), 50)
                #     batch_size = 1
                # else:
                #     batch_size = 50
                np.random.seed(time+10)
                sample_num = np.random.permutation(len(train['data']))
                train_images = train['data'][sample_num[0:Sample_Size], :]
                train_labels = train['lables'][sample_num[0:Sample_Size], :]
                train_length = len(train_images)
                accu_old = 0
                accu_all = 0
                for epoch in range(num_epochs):
                    np.random.seed(time)
                    ss = np.arange(train_length)
                    np.random.shuffle(ss)
                    trainimages = train_images[ss, :]
                    trainlabels = train_labels[ss, task_index]
                    for i in range(math.ceil(train_length/batch_size)):
                        lamda = i/math.ceil(train_length/batch_size)
                        start = batch_size*i
                        index_end = min(start+batch_size, train_length)
                        batch_x = trainimages[start:index_end, :]
                        batch_y = trainlabels[start:index_end]
                        lr_list = [0.1, 1.0]
                        OWM.owm_learn(batch_x, batch_y, lr_list, task_index)
                # if num_index % 90 == 0:
            accu_all, _ = my_test(class_begin=task_index, class_end=task_index+1)
            print('Mat_number:[{:d}/{:d}], Epoch_number:[{:d}/{:d}],curr_acc:{:.2f} %, batch_size: {:d}'
                  .format(num_index+1, 92, epoch + 1, num_epochs, accu_all, batch_size))

    acc_array_5 = []
    for task_index in Task_array:
        test_acc, _ = my_test(class_begin=task_index, class_end=task_index+1)
        acc_array_5.append(test_acc)
    print(acc_array_5)
    print('All_acc:{:.2f} %'.format(np.mean(acc_array_5)))





