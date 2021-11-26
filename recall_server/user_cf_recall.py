# -*- coding: utf-8 -*-

import redis
import traceback
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import math
import json
from sklearn.preprocessing import MinMaxScaler

# save_redis:key-value
def save_redis(items, db = 5):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], json.dumps(item[1]))
    except:
        traceback.print_exc()

# 倒排表，文章-用户序列
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['timestamp']))

    # click_df = click_df.sort_values('timestamp')
    item_user_time_df = click_df.groupby('article_id')['user_id','timestamp']\
        .apply(lambda x: make_user_time_pair(x)).reset_index().rename(\
            columns={0: 'user_time_list'})
    item_user_time_dict = dict(
        zip(item_user_time_df['article_id'], item_user_time_df['user_time_list'])
    )

    return item_user_time_dict

# 计算相似性用到关联规则：用户活跃度(经过归一化处理)
def get_user_activate_degree_dict(click_df):
    all_click_df_ = click_df.groupby('user_id')['article_id'].count().\
                    reset_index()
    # print(all_click_df)
    mn = MinMaxScaler()  # 用户活跃度归一化
    all_click_df_['article_id'] = mn.fit_transform(all_click_df_[['article_id']])
    user_activate_degree_dict = dict(
        zip(all_click_df_['user_id'], all_click_df_['article_id'])
    )
    # print(user_activate_degree_dict)
    return user_activate_degree_dict

# 计算u2u相似性矩阵，stored in redis db 5
def user_cf_sim(item_user_time_dict, user_activate_degree_dict, pool ,cut_off=20):
    
    # 时间优化trick：定义一个和redis之间的缓存
    user_info = {}

    u2u_sim = {}
    user_cnt= defaultdict(int)  # key不存在的时候返回0

    for item, user_time_list in tqdm(item_user_time_dict.items()): #tqdm进度条库
        for u, clicktime in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})

            user_u_info = user_info.get(u,None)
            if user_u_info is None:
                user_u_info = json.loads(pool.get(str(u)))
                user_info[u] = user_u_info  

            for v, clicktime in user_time_list:
                if u == v:
                    continue
                
                u2u_sim[u].setdefault(v, 0)
                # 相似度计算
                user_v_info = user_info.get(v, None)
                if user_v_info is None:
                    user_v_info = json.loads(pool.get(str(v)))
                    user_info[v] = user_v_info
            
                # 改进1：增加地区规则environment region相同权重大一些。去掉平均活跃度。
                region_weight = 1.0 if user_u_info['region'] == \
                    user_v_info['region'] else 0.9

                environment_weight = 1.0 if user_u_info['environment'] == \
                    user_v_info['environment'] else 0.9
                # 规则：用户平均活跃度作为权重
                # 疑问1：比较activate_weight是否有用
                # activate_weight = 0.1 * 0.5 * (len(user_v_info['hists']) + 
                #                 len(user_u_info['hists']))
                # activate_weight = 0.5 * (user_activate_degree_dict[u] + 
                #                 user_activate_degree_dict[v])
                
                # u2u_sim[u][v] += region_weight * activate_weight / math.log(len(user_time_list) + 1)
                u2u_sim[u][v] += region_weight * environment_weight / math.log(len(user_time_list) + 1)

    print("user_info get nums: ",len(user_info))
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        tmp = {}
        for v, wuv in related_users.items():
            tmp[v] = wuv / math.sqrt(user_cnt[u] * user_cnt[v])
        u2u_sim_[u] = sorted(
            tmp.items(), key=lambda _:_[1], reverse=True)[:cut_off]
        
    save_redis(u2u_sim_, db = 5)

def main():
    click_file = open('./data/click_log.csv')
    click_df = pd.read_csv(click_file)

    print('Generating user_activate_degree_dict...')
    user_activate_degree_dict = get_user_activate_degree_dict(click_df)
    print('User_activate_degree_dict generated.')

    print('Generating item-user history...')
    item_user_time_dict = get_item_user_time_dict(click_df)
    print('Item-user history generated.')

    redis_url = "redis://:@127.0.0.1:6379/1" 
    pool = redis.from_url(redis_url)
    print("Gererating u2u sim matrix...")
    user_cf_sim(item_user_time_dict, user_activate_degree_dict, pool, cut_off=20)
    # user_cf_sim(item_user_time_dict, pool, cut_off=20)
    print("u2u sim matrix stored.")

if __name__ == '__main__':
    main()
