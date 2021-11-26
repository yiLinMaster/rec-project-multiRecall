# -*- coding:utf-8 -*-

import tensorflow as tf

# mf model:同project1

def mf_fn(inputs, embedding_dim, is_test):
    embed_layer = inputs["feature_embedding"] # mat(batch_size, feature_num, embedding_dim)
    embed_layer = tf.reshape(embed_layer, shape = [-1,2,embedding_dim])
    # print("embed_layer size:",embed_layer.shape)
    label = inputs["label"] # mat(batch_size, 1)
    embed_layer = tf.split(embed_layer, num_or_size_splits = 2,axis = 1)
    user_id_embedding = tf.reshape(embed_layer[0], shape = [-1,embedding_dim]) # mat(batch_size, embedding_dim)
    movie_id_embedding = tf.reshape(embed_layer[1], shape = [-1,embedding_dim]) # mat(batch_size, embedding_dim)
    
    # 计算预测值
    # TODO: 只保留movie_id,user_id之前均存在的结果.tf熟练再说
    out_ = tf.reduce_mean(user_id_embedding * movie_id_embedding, axis = 1)
    # print("output size:",out_.shape)
    label_ = tf.reshape(label,[-1])

    # 验证集上结果
    out_tmp = tf.sigmoid(out_)
    if is_test:
        tf.compat.v1.add_to_collections("input_tensor",embed_layer)
        tf.compat.v1.add_to_collections("output_tensor",out_tmp)

    # 计算loss
    loss_ = tf.reduce_sum(tf.square(label_ - out_))

    out_dic = {
        "loss" : loss_,
        "ground_truth" : label_,
        "prediction" : out_
    }
    return out_dic

def setup_graph(inputs, embedding_dim, learning_rate, is_test = False):
    result = {}
    with tf.compat.v1.variable_scope("net_graph",reuse = is_test):
        net_out_dic = mf_fn(inputs, embedding_dim, is_test)
        loss = net_out_dic["loss"]
        result["out"] = net_out_dic

        if is_test:
            return result
        
        embedding_grad = tf.gradients(loss, [inputs["feature_embedding"]], \
            name = "feature_embedding")[0]
        result["feature"] = inputs["feature"] 
        result["feature_new_embedding"] = inputs["feature_embedding"] - \
            learning_rate * embedding_grad
        result["feature_embedding"] = inputs["feature_embedding"]
        return result
        