import math
import os
import torch
import numpy as np
import scipy.io as scio
from OWMLayer_cdp import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1

class_num = 1 # mnist
num_epochs = 1
batch_size = 20
dtype = torch.cuda.FloatTensor  # run on GPU
Path_All = './data/celebA_mat50/'
Data_Path_train = os.path.join(Path_All, 'train')
Data_Path_val = os.path.join(Path_All, 'test')


def my_test(class_begin=None, class_end=None):
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
                correct_each, total_each = OWM.owm_learn(batch_x, batch_y, train=False, task_index=task_index)
                total += total_each
                correct += correct_each
        test_acc = 100 * correct / total
        acc_array.append(test_acc)

    return test_acc, acc_array


lambda_loss = 1e-3
hidden_size = 4000
OWM = OWMLayer_cdp([[2048, hidden_size], [hidden_size, class_num]], alpha=[1.0, ],
                   l2_reg_lambda=lambda_loss, train_context=False)
print(hidden_size, 'train_context',OWM.train_context)

accu_max = 0
Task_array = range(40)  # 1 simple 21 ;5 simple [3 20 22 32 37]
Task_num = len(Task_array)

# is_training = True
# if is_training:
for seed_i in [4]:
    Task_array = np.arange(40)
    # if seed_i:
    seed = seed_i
    print('seed ', seed)
    np.random.seed(seed)
    np.random.shuffle(Task_array)
    task_j = 0
    for task_index in Task_array:
        task_j += 1
        print("Training owm_cdp:->>> [ {:d} / {:d} ] ... [ {:d} / {:d} ] ...".format(task_j, Task_num, task_index + 1, Task_num))
        for epoch in range(num_epochs):
            for num_index in range(0, 92):
                mat_data = os.path.join(Data_Path_train, 'celebAdata'+str(num_index))
                train = scio.loadmat(mat_data)
                trainimages = train['data']
                trainlabels = train['lables']
                train_length = len(trainimages)
                ss = np.arange(train_length)
                np.random.shuffle(ss)
                images = trainimages[ss, :]
                labels = trainlabels[ss, task_index]
                for i in range(math.ceil(train_length/batch_size)):
                    lamda = i/math.ceil(train_length/batch_size)
                    start = batch_size*i
                    index_end = min(start+batch_size, train_length)
                    batch_x = images[start:index_end, :]
                    batch_y = labels[start:index_end]
                    alpha_list = [0.15, 1.0, 0.1] # [2.0, 1.0, 0.1] for no training context
                    OWM.owm_learn(batch_x, batch_y, train=True, alpha_list=alpha_list, task_index=task_index)
            accu_all, _ = my_test(class_begin=task_index, class_end=task_index+1)
            print('Epoch_number:[{:d}/{:d}],curr_acc:{:.2f} %'.format(epoch + 1, num_epochs, accu_all))
    # else:
    #     pkl_file = open('./savedir/res_' + str(class_num) + '_' + str(middle) + '.pkl', 'rb')
    #     data = pickle.load(pkl_file)
    #     OWM = data['OWM']

    _, acc_array = my_test(class_begin=0, class_end=Task_num)
    print()
    print('All_acc:{:.2f} %'.format(np.mean(acc_array)))

# for i in acc_array:
#     print('{:.2f}'.format(i))
#
# print()
# print('All_acc:{:.2f} %'.format(np.mean(acc_array)))





