# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import json
import random
import tensorflow as tf
import redis
import os
from sklearn.preprocessing import LabelBinarizer, KBinsDiscretizer


#  1. 抽取样本特征以及构建正负样本,划分训练数据集、测试数据集
#  正负样本构建策略：对于click_log中每条数据看成一个正样本，随机抽取未点击的10个文章作为负样本
#  改进2-样本格式：article_category_id,article_created_at_ts,article_words_count,user_environment,user_region--label

def get_fm_recall_data(data_path, user_pool, item_pool):
    ds = pd.read_csv(data_path)
    ds = ds[['user_id', 'article_id']]

    items = list(ds['article_id'].unique())  # 去重,得到所有文章集合
    ds = zip(ds['user_id'], ds['article_id'])  # 转化为元组
    data = []

    # sample：user_id,user_info_environment,user_info_region,item_id -- label
    for u,i in ds:
        user_info = json.loads(user_pool.get(u))
        article_info = json.loads(item_pool.get(i))
        data.append([article_info['category_id'],article_info['created_at_ts'],article_info['words_count'],user_info['environment'], user_info['region'],1])
        for t in random.sample(items, 10):  # 随机取样
            data.append([article_info['category_id'],article_info['created_at_ts'],article_info['words_count'],user_info['environment'], user_info['region'],0])

    print("Input 5 features: article_category_id,article_created_at_ts,article_words_count,user_environment,user_region--label")
    print(data[0])
    random.shuffle(data)
    train_len = int(0.8 * len(data))
    train = data[:train_len]  # 571128
    test = data[train_len:]  # 142783
    return train,test

# 2.将样本各特征hash化并存入文件
def bkdr2hash64(str):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str:
        hash = hash* seed + ord(s)
    return hash & mask60

def tohash(data, save_path):
    print("start write in {}...".format(save_path))
    wf = open(save_path, "w")
    for line in data:
        i_cat_id = bkdr2hash64("i_cat_id=" + str(line[0]))
        i_created_at = bkdr2hash64("i_created_at=" + str(line[1]))
        i_words_cnt = bkdr2hash64("i_words_cnt=" + str(line[2]))
        u_environment = bkdr2hash64("u_environment=" + str(line[3]))
        u_region = bkdr2hash64("u_region=" + str(line[4]))
        wf.write(str(i_cat_id)+","+str(i_created_at)+','+str(i_words_cnt) +','\
            +str(u_environment)+","+str(u_region)+","+str(line[5])+"\n")
    wf.close()

# 3.将数据转化为tfrecords形式并且分多文件保存
def get_tfrecords_example(feature, label):
    tfrecords_features = {
        'feature':tf.train.Feature(
            int64_list = tf.train.Int64List(value=feature)),
        'label':tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }
    return tf.train.Example(
        features =  tf.train.Features(feature = tfrecords_features)
    )

def totfrecords(file, save_dir, feature_len):
    print("Processing to tfrecord file:%s..." % file)
    num = 0
    rec_num = 0
    writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) \
        + ".tfrecords")
    lines = open(file)
    for i, line in enumerate(lines):
        rec_num += 1
        tmp = line.strip().split(",")
        feature_list = []
        for j in range(feature_len):
            feature_list.append(int(tmp[j]))
        label = [float(tmp[feature_len])]
        example = get_tfrecords_example(feature_list, label)
        writer.write(example.SerializeToString())
        if (i+1) % 100000 == 0:
            writer.close()
            num += 1
            writer = tf.io.TFRecordWriter(save_dir + "/" + "part-0000" + str(num) \
        + ".tfrecords")
    print("Processed to tfrecord file:%s, %d records in total." % (file, rec_num))
    writer.close()

def main():
    # click_path = "./data/click_log.csv"
    # user_redis_url = "redis://:@127.0.0.1:6379/1"
    # item_redis_url = "redis://:@127.0.0.1:6379/2"
    # user_pool = redis.from_url(user_redis_url)
    # item_pool = redis.from_url(item_redis_url)
    # train, test = get_fm_recall_data(click_path, user_pool, item_pool)

    train_path = "./data/train_tohash_new"
    test_path = "./data/test_tohash_new"
    # tohash(train, train_path)
    # tohash(test, test_path)

    feature_len = 5
    train_totfrecords = "./data/train_new"
    test_totfrecords = "./data/test_new"
    os.mkdir(train_totfrecords)
    os.mkdir(test_totfrecords)
    totfrecords(train_path, train_totfrecords,feature_len)
    totfrecords(test_path, test_totfrecords,feature_len)

if __name__ == "__main__":
    main()



