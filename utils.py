import argparse
import torch
import numpy as np
import random
import os

def makedir(path):
    is_exist = os.path.exists(path)
    if is_exist:
        return '%s already exists!'%path
    else:
        os.makedirs(path)

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class OrderNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self.__dict__['order'] = []
        super(OrderNamespace, self).__init__(**kwargs)
    def __setattr__(self,attr,value):
        #  如果没有这个if，args中的str类型会在log中打印两次。
        #  猜测可能是由于父类会对str重复调用__setattr__方法的缘故
        if attr not in self.__dict__['order']:
            self.__dict__['order'].append(attr)
        super(OrderNamespace, self).__setattr__(attr, value)


def fix_seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(i)