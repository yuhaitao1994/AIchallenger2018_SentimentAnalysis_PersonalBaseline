
# 细粒度用户情感分析_个人baseline

首先是数据预处理的工作，简单的清洗，然后分词，训练词向量


```python
"""
数据预处理，清洗，分词，训练词向量

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import os
import word2vec
from tqdm import tqdm

def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    """
    加载数据成dataframe
    """
    df = pd.read_csv(file_name, header=header, encoding=encoding)
    return df

def data_clean(dataframe, out_path, isTest=False):
    """
    数据清洗，并保存清洗后的数据
    """
    stop_words = ['.', '，', '…', '、', ' ', '\n', '\t', '~'] # 停用词表还需要扩充
    new_dataframe = pd.DataFrame(columns=dataframe.columns)
    for index, rows in tqdm(dataframe.iterrows()):
        sentence = rows["content"]
        new_sentence = ""
        # 清洗评论
        for word in sentence:
            if word not in stop_words:
                new_sentence += word
        rows["content"] = new_sentence
        # 清洗标签
        if not isTest:
            for column, value in rows.iteritems():
                if column != "content":
                    if np.isnan(float(rows[column])) == True:
                        rows[column] = -2
                        print("遇到空值:{}".format(index))
        new_dataframe.loc[index] = rows
    print("清洗完成！")
    # 存储新的dataframe
    new_dataframe.to_csv(out_path, index=None)
    return new_dataframe
    

def seg_words(contents):
    """
    分词
    """
    contents_segs = []
    for content in contents:
        segs = jieba.cut(content)
        contents_segs.append(" ".join(segs))
    return contents_segs

train_file = "./sentiment_analysis_trainingset.csv"
validation_file = "./sentiment_analysis_validationset.csv"
test_file = "./sentiment_analysis_testa.csv"
train_after_clean = "./train_after_clean.csv"
val_after_clean = "./val_after_clean.csv"
test_after_clean = "./test_after_clean.csv"
seg_txt = "./seg_list.txt"
embedding_bin = "./embedding.bin"
content_limit = 500

def preprocess():
    """
    数据预处理函数
    """
    train_df = load_data_from_csv(train_file)
    val_df = load_data_from_csv(validation_file)
    test_df = load_data_from_csv(test_file)
    
    # 数据清洗
    train_df = data_clean(train_df, train_after_clean)
    val_df = data_clean(val_df, val_after_clean)
    test_df = data_clean(test_df, test_after_clean, isTest=True)
    
    train_content = train_df.iloc[:,1]
    val_content = val_df.iloc[:,1]
    test_content = test_df.iloc[:,1]
    
    # 分词，构造语料库
    all_content = []
    all_content.extend(train_content)
    all_content.extend(val_content)
    all_content.extend(test_content)
    print(len(all_content))
    all_seg_words = seg_words(all_content)
    with open(seg_txt, "w+") as txt_write:
        for sentence in tqdm(all_seg_words):
            sentence = sentence.replace("\n","") + "\n"
            txt_write.write(sentence)
    txt_write.close()
    
    # 调用word2vec
    word2vec.word2vec(seg_txt, embedding_bin, min_count=5, size=100, verbose=True)
    

# 数据预处理
# preprocess()

```

经过数据预处理后，接下来需要生成dataset


```python
"""
生成dataset以备训练与测试使用

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import random

def get_passage_limit(): # 最长3000多，还需清洗
    train_df = load_data_from_csv(train_after_clean)
    val_df = load_data_from_csv(val_after_clean)
    test_df = load_data_from_csv(test_after_clean)
    train_content = train_df.iloc[:,1]
    val_content = val_df.iloc[:,1]
    test_content = test_df.iloc[:,1]
    all_content = []
    all_content.extend(train_content)
    all_content.extend(val_content)
    all_content.extend(test_content)
    print(len(all_content))
    all_seg_words = seg_words(all_content)
    max = 0
    max_sen = []
    for sentence in all_seg_words:
        sentence = sentence.replace("\n","") + "\n"
        if len(sentence) > max:
            max = len(sentence)
            max_sen = sentence
    return max, max_sen
    

def get_dataset(file): # 最长的一个1302,我们设置长度限制为500
    """
    生成train和validation的dataset，ids：数据的id号，x：content分词列表，y:标签列表
    """
    dataframe = load_data_from_csv(file)
    dataset = []
    for index, rows in dataframe.iterrows():
        ids = rows["id"]
        x = list(jieba.cut(rows["content"]))
        if len(x) > content_limit:
            x = x[:content_limit]
        y = list(rows[2:])
        dataset.append({"id":ids,"content":x,"labels":y})
    return dataset
        
    
def batch_generator(dataset, batch_size, word2vec_bin, shuffle=True):
    """
    batch生成器
    """
    word2vec_model = word2vec.load(word2vec_bin)
    if shuffle:
        random.shuffle(dataset)
    data_num = len(dataset)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_num:
            # 最后一个batch的操作
            one_batch = dataset[batch_count * batch_size:data_num]
            for i in range(batch_size + batch_count * batch_size - data_num):
                one_batch.append(dataset[i])
            batch_count = 0
            if shuffle:
                random.shuffle(dataset)
        else:
            one_batch = dataset[batch_count * batch_size:batch_count * batch_size + batch_size]
            batch_count += 1
        # 提取数据
        index = 0
        one_batch_ids = []
        one_batch_inputs = np.zeros(shape=[batch_size, content_limit, 100], dtype=np.float32)
        one_batch_labels = []
        for one in one_batch:
            one_batch_ids.append(one["id"])
            one_batch_labels.append(one["labels"])
            for i in range(len(one["content"])):
                if one["content"][i] in word2vec_model:
                    one_batch_inputs[index,i,:] = word2vec_model[one["content"][i]] # 这里会出现keyerror，以后要注意一下
            index += 1
        one_batch_ids = np.array(one_batch_ids)
        one_batch_labels = np.array(one_batch_labels)
        yield one_batch_ids, one_batch_inputs, one_batch_labels 
        
            
                
# get_passage_limit()

"""
ge = batch_generator(get_dataset(val_after_clean)[:5],2,embedding_bin)
for i in range(6):
    aa, bb, cc = next(ge)
    print(aa)
    print(bb)
    print(cc)
"""
```




    '\nge = batch_generator(get_dataset(val_after_clean)[:5],2,embedding_bin)\nfor i in range(6):\n    aa, bb, cc = next(ge)\n    print(aa)\n    print(bb)\n    print(cc)\n'



训练模块


```python
"""
训练模块

@author: yuhaitao
"""
# -*- coding:utf-8 -*-
import tensorflow as tf
from sklearn.metrics import f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
save_dir = "./model/"

def normalize(data):
    """
    标签数据规范化成（0，1）
    """
    data = data.astype(np.float32)
    max = data.max()
    min = data.min()
    data = (data - min) / (max - min)
    return data

def predict(logits):
    """
    根据神经网络输出预处真实标签
    """
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            if logits[i][j] < 0.5:
                logits[i][j] = -2
            elif logits[i][j] < 0.65:
                logits[i][j] = -1
            elif logits[i][j] < 0.8:
                logits[i][j] = 0
            else:
                logits[i][j] = 1
    logits = logits.astype(np.int32)
    return logits
        

def f1_score_from_sklearn(predictions_dict, targets_dict):
    """
    计算F1得分
    """
    f1_list = []
    for keys, value in predictions_dict.items():
        tar = targets_dict[keys]
        f1_list.append(f1_score(tar, value, average="macro"))
    f1_list = np.array(f1_list)
    return np.mean(f1_list)
    

def f1_score_bymyself(predictions_dict, targets_dict):
    """
    自己计算F1得分
    """

def dense(inputs, hidden, use_bias=True, scope="dense"):
    """
    全连接层
    """
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def train(batch_size=64, hidden=60, grad_clip=5.0, init_lr=1.0):
    """
    训练与验证的函数
    """
    # 搭建模型
    print("Building model...")
    inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, content_limit, 100], name="inputs")
    labels = tf.placeholder(dtype=tf.float32, shape=[batch_size, 20], name="labels")
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    global_step = tf.get_variable(name='global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
    # 做mask处理
    mask = tf.cast(inputs[:,:,0], tf.bool)
    seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    max_len = tf.reduce_max(seq_len)
    mask = tf.slice(mask, [0,0], [batch_size, max_len])
    seq_f = tf.slice(inputs, [0,0,0], [batch_size, max_len, 100])
    seq_b = tf.reverse_sequence(seq_f, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
    # 双向 GRU layer, f:forward, b:backward
    gru_f = tf.contrib.rnn.GRUCell(hidden,name="gru_f")
    gru_b = tf.contrib.rnn.GRUCell(hidden,name="gru_b")
    init_f = tf.tile(tf.Variable(tf.zeros([1, hidden]), name="init_f"), [batch_size, 1])
    init_b = tf.tile(tf.Variable(tf.zeros([1, hidden]), name="init_b"), [batch_size, 1])
    with tf.variable_scope("BiGRU"):
        seq_f = tf.nn.dropout(seq_f, keep_prob)
        seq_b = tf.nn.dropout(seq_b, keep_prob)
        f, f_state = tf.nn.dynamic_rnn(gru_f, seq_f, seq_len, initial_state=init_f, dtype=tf.float32)
        b_, b_state = tf.nn.dynamic_rnn(gru_b, seq_b, seq_len, initial_state=init_b, dtype=tf.float32)
        b = tf.reverse_sequence(b_, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
        gru_out = tf.nn.dropout(tf.concat([f, b], axis=2), keep_prob=keep_prob)
    with tf.variable_scope("attention"):
        att_inputs = tf.nn.relu(dense(gru_out, hidden, use_bias=False, scope="att_inputs"))
        att_memory = tf.nn.relu(dense(gru_out, hidden, use_bias=False, scope="att_memory"))
        att_1 = tf.matmul(att_inputs, tf.transpose(att_memory, [0,2,1])) / (hidden ** 0.5)
        att_mask = tf.tile(tf.expand_dims(mask, axis=1), [1,tf.shape(att_inputs)[1],1])
        att_softmax_mask = att_1 - (1e30 * (1 - tf.cast(att_mask, tf.float32)))
        att_logits = tf.nn.softmax(att_softmax_mask)
        att_out = tf.nn.dropout(tf.matmul(att_logits, gru_out), keep_prob=keep_prob)
    with tf.variable_scope("pooling"):
        po_1 = tf.nn.tanh(dense(att_out, hidden, scope="po_1"))
        po_2 = dense(po_1, 1, use_bias=False, scope="po_2")
        po_softmax_mask = tf.squeeze(po_2,[2]) - (1e30 * (1 - tf.cast(mask, tf.float32)))
        po_weight = tf.expand_dims(tf.nn.softmax(po_softmax_mask),axis=2)
        po_out = tf.reduce_sum(po_weight * att_out, axis=1)
    with tf.variable_scope("fully_connect"):
        w = tf.get_variable(name="w",shape=[2*hidden, 20])
        b = tf.get_variable(name="b",shape=[20], initializer=tf.constant_initializer(0.))
        logits_ = tf.nn.bias_add(tf.matmul(po_out, w), b)
        logits = tf.nn.sigmoid(logits_)
    with tf.variable_scope("loss"):
        # loss = tf.reduce_mean(tf.square(normalize_labels - logits))
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    with tf.variable_scope("optimizer"):
        learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-6)
        grads = optimizer.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, grad_clip)
        train_op = optimizer.apply_gradients(zip(capped_grads, variables), global_step=global_step)
    
    # 执行sess
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True
    loss_save = 100.0
    patience = 0
    best_dev_f1 = 0.0
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        sess.run(tf.assign(learning_rate,tf.constant(init_lr, dtype=tf.float32)))
        # 准备batch
        train_set = batch_generator(get_dataset(train_after_clean), batch_size, embedding_bin)
        val_set = batch_generator(get_dataset(val_after_clean), batch_size, embedding_bin)
        # train
        print("Training...")
        for go in range(100000):
            steps = sess.run(global_step) + 1
            one_batch_ids, one_batch_inputs, one_batch_labels = next(train_set)
            one_batch_labels_ = normalize(one_batch_labels)
            feed = {inputs:one_batch_inputs, labels:one_batch_labels_, keep_prob:0.7}
            train_loss, train_optimizer, train_logits = sess.run([loss,train_op,logits], feed_dict=feed)
            if steps % 100 == 0:
                # train_prediction = predict(train_logits)
                # train_f1 = f1_score(train_prediction, one_batch_labels)
                print("steps:{},train_loss:{:.6f}".format(steps, float(train_loss)))
            if steps % 500 == 0:
                # 验证
                val_prediction_dict = {}
                val_truth_dict = {}
                val_losses = []
                for _ in range(15000 // batch_size + 1):
                    one_val_ids, one_val_inputs, one_val_labels = next(val_set)
                    one_val_labels_ = normalize(one_val_labels)
                    feed_val = {inputs:one_val_inputs, labels:one_val_labels_, keep_prob:1.0}
                    val_loss, val_logits = sess.run([loss,logits], feed_dict=feed)
                    val_prediction = predict(val_logits)
                    p_dict = {}
                    t_dict = {}
                    for ids, tr, pre in zip(one_val_ids, one_val_labels, val_prediction):
                        p_dict[str(ids)] = pre
                        t_dict[str(ids)] = tr
                    val_prediction_dict.update(p_dict)
                    val_truth_dict.update(t_dict)
                    val_losses.append(val_loss)
                val_loss = np.mean(val_losses)
                val_f1 = f1_score_from_sklearn(val_prediction_dict, val_truth_dict)
                print("steps:{}, val_loss:{:.6f}, f1_score:{:.5f}".format(steps, val_loss, val_f1))
                # 学习率
                if val_loss < loss_save:
                    loss_save = val_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= 3:
                    init_lr /= 2.0
                    loss_save = val_loss
                    patience = 0
                sess.run(tf.assign(learning_rate,tf.constant(init_lr, dtype=tf.float32)))
                # 保存模型
                if val_f1 > best_dev_f1:
                    best_dev_f1 = val_f1
                    filename = os.path.join(save_dir, "model_{}_f1_{:.5f}.ckpt".format(steps, best_dev_f1))
                    saver.save(sess, filename)
    print("finished!")

#train()
    
```

测试模块，读取模型，测试，生成提交文件


```python
"""
测试模块

@author: yuhaitao
"""
# -*- coding:utf-8 -*-

def test(batch_size=64):
    # 读取test数据
    word2vec_model = word2vec.load(embedding_bin)
    df = load_data_from_csv(test_file)
    test_df = load_data_from_csv(test_after_clean)
    # test
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.gpu_options.allow_growth = True
    print("testing ...")
    with tf.Session(config=sess_config) as sess:
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(save_dir) + ".meta")
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        inputs = tf.get_default_graph().get_tensor_by_name("inputs:0")
        keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
        logits = tf.get_default_graph().get_tensor_by_name("fully_connect/Sigmoid:0")
        prediction = []
        for i in tqdm(range(15000 // batch_size + 1)):
            one_batch_inputs = np.zeros(shape=[batch_size, content_limit, 100], dtype=np.float32)
            for j in range(batch_size):
                if i*batch_size + j < 15000:
                    one = test_df.iloc[i*batch_size + j]["content"]
                else:
                    one = "牛肉"
                one = list(jieba.cut(one))
                if len(one) > content_limit:
                    one = one[:content_limit]
                for index in range(len(one)):
                    if one[index] in word2vec_model:
                        one_batch_inputs[j,index,:] = word2vec_model[one[index]]
            feed = {inputs:one_batch_inputs, keep_prob:1.0}
            logits_ = sess.run(logits, feed_dict=feed)
            predict_logits = predict(logits_)
            # 与原content合成输出格式
            for j in range(batch_size):
                pre = []
                if i*batch_size + j < 15000:
                    predict_id = df.iloc[i*batch_size+j]["id"]
                    predict_content = df.iloc[i*batch_size+j]["content"]
                    pre.append(predict_id)
                    pre.append(predict_content)
                    pre += list(predict_logits[j,:])
                    prediction.append(pre)
        prediction_df = pd.DataFrame(data=prediction, columns=df.columns)
        prediction_df.to_csv("predictions.csv",index=None)
    print("finished!")

test()
```

    testing ...
    INFO:tensorflow:Restoring parameters from ./model/model_1500_f1_0.35543.ckpt


      0%|          | 0/235 [00:00<?, ?it/s]Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Dumping model to file cache /tmp/jieba.cache
    Dump cache file failed.
    Traceback (most recent call last):
      File "/home/yuhaitao/software/Python3/lib/python3.5/site-packages/jieba/__init__.py", line 152, in initialize
        _replace_file(fpath, cache_file)
    PermissionError: [Errno 1] Operation not permitted: '/tmp/tmpizwtt8jr' -> '/tmp/jieba.cache'
    Loading model cost 0.973 seconds.
    Prefix dict has been built succesfully.
    100%|██████████| 235/235 [02:03<00:00,  2.09it/s]


    finished!



```python

```
