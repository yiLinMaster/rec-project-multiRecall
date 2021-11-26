# -*- coding:utf-8 -*-

import numpy as np

# 单例模式
class Singleton(type):
    _instance = {}

    def __call__(cls,*args,**kwargs):
        if cls not in Singleton._instance:
            Singleton._instance[cls] = type.__call__(cls,*args,**kwargs)
        return Singleton._instance[cls]

# para-server
# self.params_server格式：{key:weight_embedding}
# self.dim
class PS(metaclass = Singleton):
    def __init__(self, weight_dim):
        np.random.seed(2020)
        self.params_server = dict()
        self.dim = weight_dim
        print("PS inited...")

    def push(self, keys, values):
        for i in range(len(keys)):
            for j in range(len(keys[i])):
                self.params_server[keys[i][j]] = values[i][j]

    def pull(self, keys):
        values = []
        for k in keys:
            tmp = []
            for arr in k:
                value = self.params_server.get(arr,None)
                if value is None:
                    value = np.random.rand(self.dim)
                    self.params_server[arr] = value
                tmp.append(value)
            values.append(tmp)
        return np.asarray(values, dtype = 'float32')

    def delete(self, keys):
        for k in keys:
            self.params_server.pop(k)

    def save(self, path):
        print("总共包含 %d 个隐向量." % len(self.params_server))
        writer = open(path,"w")
        for k,v in self.params_server.items():
            writer.write(str(k) + '\t' + ','.join(['%.8f' % _ for _ in v]) + '\n')
        writer.close()