# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 16:51
# @Author  : LIU YI

class Configs(object):
    def __init__(self):
        self.data = 'cifar100'
        self.user_num = 5
        self.gpu = 1
        self.rounds = 150
        self.local_ep = 5
        self.iid = 1
        self.unequal = 1
        self.frac = 1
        self.lr = 0.005
        self.model = 'cnn'
