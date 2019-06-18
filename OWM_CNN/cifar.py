import os,sys
import numpy as np
import torch
import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle


def get(seed=0,pc_valid=0.10):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    # CIFAR10
    if not os.path.isdir('./data/binary_cifar/'):
        os.makedirs('./data/binary_cifar')
        t_num = 2
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dat={}
        dat['train']=datasets.CIFAR10('./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test']=datasets.CIFAR10('./data/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        for t in range(10//t_num):
            data[t] = {}
            data[t]['name'] = 'cifar10-' + str(t_num*t) + '-' + str(t_num*(t+1)-1)
            data[t]['ncla'] = t_num
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(t_num*t, t_num*(t+1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)
        t = 10 // t_num
        data[t] = {}
        data[t]['name'] = 'cifar10-all'
        data[t]['ncla'] = 10
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            data[t][s] = {'x': [], 'y': []}
            for image, target in loader:
                label = target.numpy()[0]
                data[t][s]['x'].append(image)
                data[t][s]['y'].append(label)

        # "Unify" and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser('./data/binary_cifar'), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser('./data/binary_cifar'), 'data' + str(t) + s + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(6))
    print('Task order =', ids)
    for i in range(6):
        data[i] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[i][s]={'x':[],'y':[]}
            data[i][s]['x']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar'),'data'+str(ids[i])+s+'x.bin'))
            data[i][s]['y']=torch.load(os.path.join(os.path.expanduser('./data/binary_cifar'),'data'+str(ids[i])+s+'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        data[i]['name'] = 'cifar10->>>' + str(i * data[i]['ncla']) + '-' + str(data[i]['ncla'] * (i + 1) - 1)

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla'] = n

    return data, taskcla[:10//data[0]['ncla']], size
