# -*- coding: utf-8 -*-

import redis
import traceback
import json
import numpy as np
import pandas as pd
from sklearn import neighbors
from tqdm import tqdm

def bkdr2hash64(str):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str:
        hash = hash* seed + ord(s)
    return hash & mask60

# save_redis:key-value
def save_redis(items, db = 6):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()

# 读取fm部分计算的feature_embedding文件：hashkey-vector
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")[:-1]]  # 最后一位是线性w不要
            dic[int(tmp[0])] = embedding
    return dic

# 将各特征hash值和id进行一个映射，map hashkey -> raw value
# TODO:article_created_at_ts和article_words_count没分桶直接用好像有点不对
def get_hash2id(item_src_file,user_src_file):
    item_ds = pd.read_csv(item_src_file)
    user_ds = pd.read_csv(user_src_file)
    cats = list(item_ds['category_id'].unique())
    ts = list(item_ds['created_at_ts'].unique())
    words = list(item_ds['words_count'].unique())
    envs = list(user_ds['environment'].unique())
    regions = list(user_ds['region'].unique())  
    cats_dict = {bkdr2hash64("i_cat_id=" + str(cat)): int(cat) for cat in cats}
    ts_dict = {bkdr2hash64("i_created_at=" + str(t)): int(t) for t in ts}
    words_dict = {bkdr2hash64("i_words_cnt=" + str(word)): int(word) for word in words}
    envs_dict = {bkdr2hash64("u_environment=" + str(env)): int(env) for env in envs}
    regions_dict = {bkdr2hash64("u_region=" + str(region)): int(region) for region in regions}
    return cats_dict, ts_dict, words_dict, envs_dict, regions_dict

# 将user和item的key和vector对应划分到item_embedding user_embedding集合中
# "feature_name" + str(feature_value) -> vector
def split_user_item(embedding_file,item_src_file,user_src_file):
    embedding_dict = read_embedding_file(embedding_file)
    cats_dict, ts_dict, words_dict, envs_dict, regions_dict = get_hash2id(item_src_file,user_src_file)
    
    item_embedding = dict()
    user_embedding = dict()

    for k,v in embedding_dict.items():
        cat_id = cats_dict.get(k, None)
        ts_id = ts_dict.get(k, None)
        words_id = words_dict.get(k, None)
        envs_id = envs_dict.get(k,None)
        regions_id = regions_dict.get(k,None)

        if cat_id is not None:
            item_embedding["cat_id="+str(cat_id)] = v
        if ts_id is not None:
            item_embedding["ts_id="+str(ts_id)] = v
        if words_id is not None:
            item_embedding["words_id="+str(words_id)] = v
        if envs_id is not None:
            user_embedding["envs_id="+str(envs_id)] = v
        if regions_id is not None:
            user_embedding["regions_id="+str(regions_id)] = v

    print("item_embedding size: ",len(item_embedding))
    print("user_embedding size: ",len(user_embedding))
    return item_embedding, user_embedding # 5517, 30

# 改动4: 由于增加了item的feature数目，所以不能直接用item_feature_embedding,要构建item_embedding
# TODO：疑问feature_embedding累加得到item_embedding的深入解释
# embedding_server:使用balltree树索引算法快速返回最相似embedding
# TODO:一会读源文件一会读feature_server有点杂乱，待统一优化

# aritcle_id - article_embedding
def get_item_embedding(data_path, item_feature_embedding):
    ds = pd.read_csv(data_path)
    item_embedding = dict()
    for ind,row in ds.iterrows():
        # data[id] = [article_info['category_id'],article_info['created_at_ts'],article_info['words_count']]
        cat_id = "cat_id="+str(row["category_id"])
        ts_id = "ts_id="+str(row["created_at_ts"])
        words_id = "words_id="+str(row["words_count"])
        emb = [item_feature_embedding[cat_id],item_feature_embedding[ts_id],item_feature_embedding[words_id]]
        emb = np.sum(np.asarray(emb), axis=0)
        item_embedding[row["article_id"]] = emb
    return item_embedding

def embedding_sim(item_embedding, cut_off = 20):
    # 在进行balltree算法前保存index_id映射
    item_idx_2_rawid_dict = {}
    item_emb_np = []
    for i,(k,v) in enumerate(item_embedding.items()):
        item_idx_2_rawid_dict[i] = k    
        item_emb_np.append(v)

    # 向量归一化
    print(np.array(item_emb_np).shape)
    item_emb_np = np.asarray(item_emb_np)
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis = 1, keepdims = True)
    # print(item_emb_np.shape) # (5050, 1, 16)
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
            sim_tmp[int(rele_raw_id)] = float(sim_value)
        #  注意这里要显示指定一下int 不是int64，否则json出问题
        item_emb_sim_dict[int(target_raw_id)] = sorted(sim_tmp.items(), key=lambda _:_[1],\
            reverse = True)[:cut_off] #注意这里还要排序再截取，40并不是数目界限，而是balltree叶子节点数
        
    print("Saving i2i_sim_matrix...")
    item_i2i_sim = [(_, json.dumps(v)) for _, v in item_emb_sim_dict.items()]
    save_redis(item_i2i_sim, db = 6)
    print("i2i_sim matrix saved.")


def write_embedding(emb, file):
    wfile = open(file, 'w')
    for k,v in emb.items():
        wfile.write(str(k) + '\t' + ','.join(str(_) for _ in v) + '\n')

if __name__ == '__main__':
    # TODO：统一从feature_server里读
    user_src_file = './data/click_log.csv'
    item_src_file = './data/articles.csv'
    embedding_file = './data/saved_fm_weights_new'
    item_feature_embedding, user_feature_embedding = split_user_item(embedding_file,item_src_file,user_src_file)
    item_embedding = get_item_embedding(item_src_file,item_feature_embedding)
    embedding_sim(item_embedding, cut_off = 20)
    write_embedding(item_feature_embedding, "./data/fm_article_feature_emb")
    write_embedding(user_feature_embedding, "./data/fm_user_feature_emb")
    write_embedding(item_embedding, "./data/fm_article_emb")








 
