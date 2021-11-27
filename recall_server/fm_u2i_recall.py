# -*- coding: utf-8 -*-

import redis
import traceback
import json

# 只是将user_feature_embedding和item_embedding保存起来以备后续使用

def save_redis(items, db = 7):
    redis_url = "redis://:@127.0.0.1:6379/" + str(db)
    pool = redis.from_url(redis_url)
    try:
        for item in items.items():
            pool.set(item[0], item[1])
    except:
        traceback.print_exc()

# 读取fm部分计算的feature_embedding文件：hashkey-vector
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split("\t")
            embedding = [float(_) for _ in tmp[1].split(",")]  
            dic[tmp[0]] = json.dumps(embedding)
    return dic

if __name__ == "__main__":
    user_feature_embedding_file = "./data/fm_user_feature_emb"
    article_embedding_file = "./data/fm_article_emb"

    save_redis(read_embedding_file(user_feature_embedding_file), db = 8)
    save_redis(read_embedding_file(article_embedding_file), db = 7)