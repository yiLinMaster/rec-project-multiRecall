# -*- coding: utf-8 -*-

import redis
import traceback
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import math
import json

# save_redis:key-value
def save_redis(items, db = 4):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], json.dumps(item[1]))
    except:
        traceback.print_exc()

# 从用户行为记录中获取user_item_time矩阵
def get_user_item_time(click_df):
    click_df = click_df.sort_values('timestamp')

    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['timestamp']))

    user_item_time_df = click_df.groupby('user_id')['article_id','timestamp']\
        .apply(lambda x: make_item_time_pair(x)).reset_index().rename(\
            columns={0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list'])
    )

    return user_item_time_dict

# 计算i2i相似性矩阵，stored in redis db 4
# 相似性计算：user_item_time, item_feature
# 取相似性强的前200位
def item_cf_sim(user_item_time_dict, pool ,cut_off=20):
    
    # 时间优化trick：定义一个和redis之间的缓存，如果已经读取过item不用再去redis找
    item_info = {}

    i2i_sim = {}
    item_cnt= defaultdict(int)  # key不存在的时候返回0

    for user, item_time_list in tqdm(user_item_time_dict.items()): #tqdm进度条库
        for loc1, (i, i_clicktime) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})

            item_i_info = item_info.get(i,None)
            if item_i_info is None:
                item_i_info = json.loads(pool.get(str(i)))
                item_info[i] = item_i_info  

            for loc2, (j, j_clicktime) in enumerate(item_time_list):
                if i == j:
                    continue
                # 相似度计算：在基础的item_cf上增加两个关联规则权重
                # 规则1：同一个用户点击时间差越小相似度越高
                click_time_weight = np.exp(0.7 ** np.abs(j_clicktime-i_clicktime))

                item_j_info = item_info.get(j, None)
                if item_j_info is None:
                    item_j_info = json.loads(pool.get(str(j)))
                    item_info[j] = item_j_info 
                # 规则2：两篇文章类别相同权重大
                type_weight = 1.0 if item_i_info['category_id'] == \
                    item_j_info['category_id'] else 0.7

                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += \
                    click_time_weight * type_weight \
                    / math.log(len(item_time_list) + 1)

    print("item_info get nums: ",len(item_info))
    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        tmp = {}
        for j, wij in related_items.items():
            tmp[j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        i2i_sim_[i] = sorted(
            tmp.items(), key=lambda _:_[1], reverse=True)[:cut_off]
        
    save_redis(i2i_sim_, db = 4)

def main():
    click_file = open('./data/click_log.csv')
    click_df = pd.read_csv(click_file)

    print('Fetching user history...')
    user_item_time_dict = get_user_item_time(click_df)
    print('User history fetched.')

    redis_url = "redis://:@127.0.0.1:6379/2" 
    pool = redis.from_url(redis_url)
    print("Gererating i2i sim matrix...")
    item_cf_sim(user_item_time_dict, pool, cut_off=200)
    print("i2i sim matrix stored.")

if __name__ == '__main__':
    main()
