# -*- coding: utf-8 -*-

import redis
import traceback
import json
import numpy as np
from sklearn import neighbors
from tqdm import tqdm

# save_redis:key-value
def save_redis(items, db = 3):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0],item[1])
    except:
        traceback.print_exc()

# 读取matrix_cf部分计算的article_embedding文件
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]
            dic[int(tmp[0])] = embedding
    return dic

# embedding_server:使用balltree树索引算法快速返回最相似embedding
def embedding_sim(item_emb_file, cut_off = 20):
    item_embedding = read_embedding_file(item_emb_file)

    # 在进行balltree算法前保存index_id映射
    item_idx_2_rawid_dict = {}
    item_emb_np = []
    for i, (k,v) in enumerate(item_embedding.items()):
        item_idx_2_rawid_dict[i] = k    
        item_emb_np.append(v)

    # 向量归一化
    item_emb_np = np.asarray(item_emb_np)
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis = 1, keepdims = True)

    # 构建balltree树索引获得相似vector检索结果
    # 距离度量，metric默认值=“ minkowski”，其中p = 2（即欧氏度量）
    print("Build balltree...")
    item_tree = neighbors.BallTree(item_emb_np, leaf_size = 40)
    sim, idx = item_tree.query(item_emb_np, cut_off) # 对各个vector返回的都是list

    item_emb_sim_dict = {}
    for target_idx, sim_value_list, rele_idx_list in \
        tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        sim_tmp = {}

        # 注意balltree构建时候包括了本身，记得去掉
        for rele_idx, sim_value in zip(rele_idx_list [1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            sim_tmp[rele_raw_id] = sim_value

        item_emb_sim_dict[target_raw_id] = sorted(sim_tmp.items(), key=lambda _:_[1],\
            reverse = True)[:cut_off]
        
    print("Saving i2i_sim_matrix...")
    item_i2i_sim = [(_, json.dumps(v)) for _, v in item_emb_sim_dict.items()]
    save_redis(item_i2i_sim, db = 3)
    print("i2i_sim matrix saved.")

if __name__ == '__main__':
    item_emb_file = './data/matrixcf_articles_emb.csv'
    embedding_sim(item_emb_file, 20)







 
