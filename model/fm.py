# -*- coding:utf-8 -*-

import tensorflow as tf

# fm model

# embed_size = 16
weight_dim = 17# 针对每一维特征训练出来的隐向量：含义前16维对应wij中的vi+线性部分的wi
learning_rate = 0.001
f_nums = 5

# fm_fn：predict 以及计算loss
def fm_fn(inputs, is_test):
    weight_ = tf.reshape(inputs["feature_embedding"],\
        shape=[-1, f_nums, weight_dim]) # batch_size * f_nums * weight_dim
    
    #  分开vi和wi
    weight_ = tf.split(weight_, num_or_size_splits=[weight_dim - 1,1], axis = 2)  # batch_size * f_nums * weight_dim - 1;batch_size * f_nums * 1

    # linear_part
    bias_part =  tf.compat.v1.get_variable("bias",[1,],initializer=tf.zeros_initializer())  # w0初始化为0
    linear_part = tf.nn.bias_add(tf.reduce_sum(weight_[1],axis = 1),bias_part)  # 相当于inputfeature为onehot编码，此部分lr==w1*x1 + w2*x2....wn*xn相加;得到batchsize * 1
    
    #  cross_part
    summed_square = tf.square(tf.reduce_sum(weight_[0], axis = 1)) # batchsize, weight_dim
    squared_sum = tf.reduce_sum(tf.square(weight_[0]), axis = 1)  # batchsize, weight_dim
    cross_part = 0.5 * tf.reduce_sum(tf.subtract(summed_square, squared_sum),axis = 1,keepdims = True)  # batchsize
    
    #  predict
    out_ = cross_part + linear_part# batchsize

    if is_test:
        out_tmp = tf.sigmoid(out_)
        tf.add_to_collections("input tensor", weight_)
        tf.add_to_collections("output tensor", out_tmp)

    #  loss
    loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits = out_, labels = inputs["label"]))

    out_dic = {
        "loss":loss_,
        "ground_truth":inputs["label"][:,0],
        "prediction":out_[:,0]
    }

    return out_dic

# setup_graph：定义优化方法
def setup_graph(inputs, is_test = False):
    result = {}
    with tf.variable_scope("net_graph", reuse = is_test):
        # init graph
        net_out_dic = fm_fn(inputs, is_test)

        loss = net_out_dic["loss"]
        result["out"] = net_out_dic

        if is_test:
            return result
        
        # SGD
        # 改进2:梯度改成对隐向量求
        emb_grad = tf.gradients(loss, [inputs["feature_embedding"]], name="feature_embedding")[0]
        
        result["feature_new_embedding"] = \
            inputs["feature_embedding"] - learning_rate * emb_grad
        result["feature_embedding"] = inputs["feature_embedding"]
        result["feature"] = inputs["feature"]

        return result





