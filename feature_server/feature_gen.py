# -*- coding: utf-8 -*-

import redis
import pandas as pd
import traceback
import json

# 抽取特征存储于redis db1(user feature),db2(article feature)


# save_redis:key-value
def save_redis(items, db = 1):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items:
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()

# user features stored in redis db 1
# 10000 * userid:environment_region article_readtime_list
def get_user_feature():
    ds = pd.read_csv("./data/click_log.csv")
    click_df = ds.sort_values('timestamp')
    user_environment_region_dict = {}
    for info in zip(click_df['user_id'], click_df['environment'], click_df['region']):
        user_environment_region_dict[info[0]] = (info[1], info[2])
    print("Number of users:",len(user_environment_region_dict)) # 10000
    
    def make_item_time_pair(df):
        return list(zip(df['article_id'], df['timestamp']))

    user_item_time_df = click_df.groupby('user_id')['article_id','timestamp']\
        .apply(lambda x: make_item_time_pair(x)).reset_index().rename(\
            columns={0: 'item_time_list'})
    user_item_time_dict = dict(
        zip(user_item_time_df['user_id'], user_item_time_df['item_time_list'])
    )

    user_feature = []
    for user,item_time_dict in user_item_time_dict.items():
        info = user_environment_region_dict[user]
        tmp = (str(user),json.dumps({
            'user_id': user,
            "hists": item_time_dict,
            "environment": info[0],
            "region": info[1]
        }))
        user_feature.append(tmp)

    save_redis(user_feature, 1)

# item features stored in redis db 2
# 5050 * article_id:category_id,created_at_ts,words_count
def get_item_feature():
    ds = pd.read_csv("./data/articles.csv")
    ds = ds.to_dict(orient="records")  # 每条数据为一个dict
    item_feature = []
    for d in ds:
        d['article_id'] = int(d['article_id'])
        d['category_id'] = int(d['category_id'])
        d['created_at_ts'] = int(d['created_at_ts'])
        d['words_count'] = int(d['words_count'])
        # print(d)
        # print(d)
        item_feature.append((str(d['article_id']),json.dumps({
            'article_id': d['article_id'],
            "category_id": d['category_id'],
            "created_at_ts": d['created_at_ts'],
            "words_count": d['words_count']
        })))

    print("Number of articles: ",len(item_feature))

    save_redis(item_feature, 2)

if __name__ == "__main__":
    # print("Generating user feature...")
    # get_user_feature()
    print("Generating item feature...")
    get_item_feature()
