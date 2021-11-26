# -*- coding:utf-8 -*-

from ps import PS
from inputs import InputFn
from auc import AUCUtils
from fm import f_nums, weight_dim, setup_graph
import tensorflow as tf

# para-server
local_ps = PS(weight_dim)

# metric
train_metric = AUCUtils()
test_metric = AUCUtils()

train_file = "./data/train_new"
test_file = "./data/test_new"
saved_weights = './data/saved_fm_weights_new'

# input
batch_size = 32
inputs = InputFn(local_ps, f_nums, batch_size)
train_iter , train_inputs = inputs.input_fn(train_file, is_test=False)
test_iter , test_inputs = inputs.input_fn(test_file, is_test=True)

train_dic = setup_graph(train_inputs, is_test=False)
test_dic = setup_graph(test_inputs, is_test=True)

# train paras
max_steps = 178500
train_log_step = 10000
test_show_step = 17850
last_test_auc = 0.5

def train():
    _step = 0
    print("#" * 80)

    def valid_step(sess, test_iter, test_dic):
        test_metric.reset()
        sess.run(test_iter.initializer)
        global last_test_auc
        while True:
            try:
                out = sess.run(test_dic["out"])
                test_metric.add(
                    out["loss"],
                    out["ground_truth"],
                    out["prediction"]
                )
            except tf.errors.OutOfRangeError:
                print("Test at step: %s" % test_metric.calc_str())
                test_auc = test_metric.calc()["auc"]
                if test_auc > last_test_auc:
                    last_test_auc = test_auc
                local_ps.save(saved_weights)
                break

    # start a session
    with tf.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        sess.run(train_iter.initializer)

        while _step < max_steps:
            feature_old_embedding, feature_new_embedding,  keys, out = sess.run(
                [train_dic["feature_embedding"],
                train_dic["feature_new_embedding"],
                train_dic["feature"],
                train_dic["out"]
                ]
            )

            train_metric.add(
                out["loss"],
                out["ground_truth"],
                out["prediction"]
            )

            local_ps.push(keys, feature_new_embedding)
            _step += 1

            if _step % train_log_step == 0:
                print("Train at step %d: %s"% (_step, train_metric.calc_str()))
                train_metric.reset()
            if _step % test_show_step == 0:
               valid_step(sess, test_iter, test_dic)
            
if __name__ == "__main__":
    train()




