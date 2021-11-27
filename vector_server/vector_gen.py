# -*- coding: utf-8 -*-

from numpy.core.arrayprint import _leading_trailing
import json
import numpy as np
from sklearn import neighbors

class VectorServer:
    def __init__(self, pool):

        self.pool = pool
        self.keys_index = []
        self.vector_matrix = []

        keys = self.pool.keys()
        pipe = self.pool.pipeline()

        key_list = []
        s = 0
        for key in keys:
            key_list.append(key)
            pipe.get(key)
            if s < 10000:
                s += 1
            else:
                # 使用pool.pipeline方法，每10000个请求处理一次
                for k,v in zip(key_list, pipe.execute()):
                    vec = json.loads(v)
                    self.keys_index.append(int(k))
                    self.vector_matrix.append(vec)
                s = 0
                key_list = []
            
            for k,v in zip(key_list, pipe.execute()):
                vec = json.loads(v)
                self.keys_index.append(int(k))
                self.vector_matrix.append(vec)
        
        item_emb_np = np.array(self.vector_matrix)
        item_emb_np = item_emb_np / np.linalg.norm(item_emb_np,axis = 1,keepdims = True)
        # print(item_emb_np.shape)
        print("Build balltree...")
        self.item_tree = neighbors.BallTree(item_emb_np, leaf_size=40)

    # input vector_list output_每个向量[[vector,value]...]
    def get_sim_item(self, items, cut_off):
        sim, idx = self.item_tree.query(items, cut_off)
        items_result= []
        for i in range(len(sim)):
            item_sim_score = dict(zip(idx[i], sim[i]))
            # print(len(item_sim_score.items()))
            item_sim_score = sorted(item_sim_score.items(), key = lambda _:_[1],
                reverse=True)[:cut_off]
            items_result.append(item_sim_score)
        # print(np.asarray(items_result).shape)
        return items_result



            