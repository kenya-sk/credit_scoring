#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from datetime import datetime

def under_sampling(X, y):
    """
    ret: undersampled (X, y)
    """

    def select(length, k):
        """
        ret: array of boolean which length = length and #True = k
        """
        seed = np.arange(length)
        np.random.shuffle(seed)
        return seed < k

    assert X.shape[0] == len(y)

    positive = [0.0, 1.0]
    msk = (y == positive) # select all positive samples first
    msk[~msk] = select((~msk).sum(), msk.sum()) # select same number of negative samples with positive samples
    return X[msk[:,0]], y[msk[:, 0]]

    
def hard_negative_mining(X, y, loss):
    #get index that error is greater than the threshold
    def hard_negative_index(loss, thresh):
        index = np.where(loss > thresh)[0]
        return index

    # the threshold is five times the average
    thresh = np.mean(loss) * 1.5
    index = hard_negative_index(loss, thresh)
    hard_negative_image_arr = np.zeros((len(index), 286), dtype="float64")
    hard_negative_label_arr = np.zeros((len(index), 2), dtype="float64")
    for i, hard_index in enumerate(index):
        hard_negative_image_arr[i] = X[hard_index]
        hard_negative_label_arr[i] = y[hard_index]
    return hard_negative_image_arr, hard_negative_label_arr


def MLP(x,keep_prob):
    he_init = tf.variance_scaling_initializer()
    
    layer_1 = tf.layers.dense(x, 286, activation=tf.nn.relu, kernel_initializer=he_init)
    layer_1_drop = tf.nn.dropout(layer_1, keep_prob)
    
    layer_2 = tf.layers.dense(layer_1_drop, 286, activation=tf.nn.relu, kernel_initializer=he_init)
    layer_2_drop = tf.nn.dropout(layer_2, keep_prob)
    
    layer_3 = tf.layers.dense(layer_2_drop, 143, activation=tf.nn.relu, kernel_initializer=he_init)
    layer_3_drop = tf.nn.dropout(layer_3, keep_prob)
    
    layer_4 = tf.layers.dense(layer_3_drop, 72, activation=tf.nn.relu, kernel_initializer=he_init)
    layer_4_drop = tf.nn.dropout(layer_4, keep_prob)
    
    layer_5 = tf.layers.dense(layer_4_drop, 36, activation=tf.nn.relu, kernel_initializer=he_init)
    layer_5_drop = tf.nn.dropout(layer_5, keep_prob)
    
    out = tf.layers.dense(layer_5_drop, 2)
    return out

def mlp_main(train_X_path, train_y_path, test_path, reuse_model_path, learning=False):
    # preprocessing
    X = np.load(train_X_path)
    y = np.load(train_y_path)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # one-hot-vectorに変換
    y_train = np.eye(2)[y_train.astype("int")]
    y_val = np.eye(2)[y_val.astype("int")]

    print("X_train shape: {}".format(X_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_val shape: {}".format(X_val.shape))
    print("y_val shape: {}".format(y_val.shape))

    # define graph
    tf.reset_default_graph()
    # parameter
    n_epoch = 100
    batchsize = 250
    # input
    X = tf.placeholder(tf.float64, [None, 286])
    y_ = tf.placeholder(tf.float64, [None, 2])
    is_training = tf.placeholder(tf.bool)
    # dropout ratio
    keep_prob = tf.placeholder(tf.float64)
    # prediction
    y = MLP(X, keep_prob)
    # cost function
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    loss = tf.reduce_mean(xent)
    # optimizer
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)
    # Output the class probabilities to I can get the AUC
    predict = tf.argmax(y, axis=1)
    predict_prob = tf.nn.softmax(y)
    # diff score
    train_diff = tf.abs(predict_prob - y_)

    with tf.Session() as sess:
        # save weight
        saver = tf.train.Saver()
        # model exist: True or False
        ckpt = tf.train.get_checkpoint_state(reuse_model_path)
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            print("START: Relearning")
            print("LODE: {}".format(last_model))
            saver.restore(sess, last_model)
        else:
            print("START: learning")
            # initialize all variable
            tf.global_variables_initializer().run()

        if learning:
            # learning
            print("********************* LEARNING ************************")
            train_auc, val_auc = [], []
            n_rounds_not_improved = 0
            early_stopping_epochs = 3
            
            # hrad negative array
            hard_negative_X_arr = np.zeros((1, 286), dtype="float64")
            hard_negative_y_arr = np.zeros((1, 2), dtype="float64")
            for epoch in range(n_epoch):
                print ('epoch %d | ' % epoch, end="")

                # Training
                # under sampling
                under_X_train, under_y_train = under_sampling(X_train, y_train)
                under_N_train = under_X_train.shape[0]
                
                print("hard negative data: {}".format(hard_negative_y_arr.shape[0] - 1))
                if hard_negative_y_arr.shape[0] > 1:
                    under_X_train = np.append(under_X_train, hard_negative_X_arr[1:], axis=0)
                    under_y_train = np.append(under_y_train, hard_negative_y_arr[1:], axis=0)
                    
                # 次のepoch用にhard_begariveを初期化
                hard_negative_X_arr = np.zeros((1, 286), dtype="float64")
                hard_negative_y_arr = np.zeros((1, 2), dtype="float64")
                    
                for i in range(0, under_N_train, batchsize):
                    # ミニバッチ分のデータを取ってくる
                    X_batch = under_X_train[i:i+batchsize]
                    y_batch = under_y_train[i:i+batchsize]

                    _, diff  = sess.run([train_step, train_diff], feed_dict={X:X_batch.reshape(-1, 286),
                                                            y_:y_batch.reshape(-1, 2),
                                                            keep_prob:0.6})
                    
                    # hard negative mining
                    batch_hard_negative_X_arr, batch_hard_negative_y_arr = \
                            hard_negative_mining(X_batch, y_batch, diff)
                    if batch_hard_negative_y_arr.shape[0] > 0: # there are hard negative data
                        hard_negative_X_arr = np.append(hard_negative_X_arr, batch_hard_negative_X_arr, axis=0)
                        hard_negative_y_arr = np.append(hard_negative_y_arr, batch_hard_negative_y_arr, axis=0)
                    else:
                        pass
                    
                # AUCの確認
                y_pred_train, y_prob_train = sess.run([predict, predict_prob],
                                                    feed_dict={X:X_train, y_:y_train, keep_prob:0.6})
                train_auc.append(roc_auc_score(y_train[:, 1], y_prob_train[:, 1]))

                y_pred_val, y_prob_val = sess.run([predict, predict_prob],
                                        feed_dict={X:X_val, y_:y_val, keep_prob:1.0})
                val_auc.append(roc_auc_score(y_val[:, 1], y_prob_val[:, 1]))

                # Early stopping
                if epoch > 1:
                    best_epoch_so_far = np.argmax(val_auc[:-1])
                    if val_auc[epoch] <= val_auc[best_epoch_so_far]:
                        n_rounds_not_improved += 1
                    else:
                        n_rounds_not_improved = 0       
                    if n_rounds_not_improved > early_stopping_epochs:
                        print('Early stopping due to no improvement after {} epochs.'.format(early_stopping_epochs))
                        break
                print('Epoch = {}, Train AUC = {:.8f}, Test AUC = {:.8f}'.format(epoch, train_auc[epoch], val_auc[epoch]))

            date = datetime.now()
            learning_date = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
            saver.save(sess, "../../all/tensor_model/" + learning_date + "/model.ckpt")
            print("DONE! LEARNING")
            print("************************************************************")

        # prediction
        print("********************* PREDICTION ************************")
        X_test = np.load(test_path)
        test_pred, test_pred_prob = sess.run([predict, predict_prob], feed_dict={X:X_test, keep_prob:1.0})
        print("*********************************************************")
        
    return test_pred_prob[:, 1]

if __name__ == "__main__":
    train_X_path = "../../all/train_X.npy"
    train_y_path = "../../all/train_target.npy"
    test_path = "../../all/test.npy"
    reuse_model_path = "../../all/tensor_model/"
    mlp_main(train_X_path, train_y_path, test_path, reuse_model_path, learning=True)