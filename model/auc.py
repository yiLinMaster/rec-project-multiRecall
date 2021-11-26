# -*- coding:utf-8 -*-

import numpy as np

# 定义模型评估类
from sklearn.metrics import roc_auc_score
class AUCUtils(object):
    def __init__(self):
        self.reset()

    def add(self, loss, g = np.array([]),p = np.array([])):
        self.loss.append(loss)
        self.ground_truth += g.flatten().tolist()
        self.prediction += p.flatten().tolist()

    def calc(self):
        return {
            "loss_num":  len(self.loss),
            "loss": np.array(self.loss).mean(),
            "auc_num": len(self.ground_truth),
            "auc": roc_auc_score(self.ground_truth,self.prediction) if \
                len(self.ground_truth) > 0 else 0,
            "pcoc": sum(self.prediction) / sum(self.ground_truth)
        }

    def calc_str(self):
        res = self.calc()
        return "loss: %f(%d), auc: %f(%d), pcoc: %f" % (res["loss"], \
            res["loss_num"], res["auc"], res["auc_num"], res["pcoc"])

    def reset(self):
        self.loss = []
        self.prediction = []
        self.ground_truth = []