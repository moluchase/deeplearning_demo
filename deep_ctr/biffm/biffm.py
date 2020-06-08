import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import random

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import itertools
from tensorflow.keras.initializers import (Zeros, glorot_normal, glorot_uniform)
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

"""
本程序涉及到：
如何自定义Layer
biffm是如何缩减参数的
打印模型结构


dataframe 的使用
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
user_profile.set_index("user_id", inplace=True)
user_profile.loc[[1,2,3,4]]['gender'].values
通过loc来索引值需要使用set_index来标识。


References
https://github.com/wziji/deep_ctr/tree/master/BilinearFFM
https://zhuanlan.zhihu.com/p/145928996
https://zhuanlan.zhihu.com/p/67795161
https://github.com/shenweichen/DeepCTR


pandas tutorial: https://pandas.pydata.org/pandas-docs/stable/index.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
https://www.tensorflow.org/api_docs/python/tf/keras/Input

https://www.tensorflow.org/tutorials/customization/custom_layers
https://www.tensorflow.org/guide/keras/custom_layers_and_models
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

https://www.tensorflow.org/api_docs/python/tf/keras/backend
https://www.machenxiao.com/blog/tensordot
"""


data_dir='data'
log_dir='log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# 负采样方式
"""
hist表示用户下一步要观看的电影
len(hist) 为reviewerID对应的group中元素的个数，应该是看过的movie个数

我理解这个构造集的含义是：当用户前面看了哪些movie后(hist[::-1])，接下来的movie是否会观看(pos_list[i] or neg_list[i*negsample+negi])
"""
def gen_data_set(data, negsample=0):

    data.sort_values("timestamp", inplace=True)
    item_ids = data['movie_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['movie_id'].tolist()
        rating_list = hist['rating'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set,size=len(pos_list)*negsample,replace=True)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1,len(hist[::-1]),rating_list[i]))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i*negsample+negi], 0,len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i],1,len(hist[::-1]),rating_list[i]))

    random.shuffle(train_set)
    random.shuffle(test_set)

    print(len(train_set[0]),len(test_set[0]))

    return train_set,test_set

#
"""

"""
def gen_model_input(train_set,user_profile,seq_max_len):

    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    # 还有个truncating参数，该参数默认为pre，表示超过部分应该按什么方式截取，默认是截取前面的部分
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', value=0)
    train_model_input = {"user_id": train_uid, "movie_id": train_iid, "hist_movie_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in ["gender", "age", "occupation", "zip"]:
        # 这里需要为user_profile指定索引才能如下获取
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input,train_label

#
def data_process():
    unames=['user_id','gender','age','occupation','zip']
    user = pd.read_csv(os.path.join(data_dir,'ml-1m/users.dat'),sep='::',header=None,names=unames)

    rnames = ['user_id','movie_id','rating','timestamp']
    ratings = pd.read_csv(os.path.join(data_dir,'ml-1m/ratings.dat'),sep='::',header=None,names=rnames)

    mnames = ['movie_id','title','genres']
    movies = pd.read_csv(os.path.join(data_dir,'ml-1m/movies.dat'),sep='::',header=None,names=mnames)

    data = pd.merge(pd.merge(ratings, movies), user) 

    """
    sklearn 函数的一般使用方法：
    fit 用来拟合数据，按传入的数据制定标准
    transform 用来转换数据
    """

    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    SEQ_LEN = 50
    negsample = 0
    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
    # drop_duplicates：Return DataFrame with duplicate rows removed
    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    # Set the DataFrame index using existing columns.
    user_profile.set_index("user_id", inplace=True)
    # Group DataFrame using a mapper or by a Series of columns.
    # 按user_id分组，并取movie_id字段，将该组中的元素组成一个list
    user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set(data, negsample)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 3. Create neg samples

    """
    该构造负采样的方法是建立在前面的基础上的：
        正样本：当用户看了hist movie集后，看了pos_list[i]；负样本：当用户看了hist movie集后，没有看neg_list[i*negsample+negi]
    此时构造数据集的时候：
        正样本是pos_list[i] / neg_list[i*negsample+negi]
        负样本是 != (hist movie集 & pos_list[i] / neg_list[i*negsample+negi] )
        没太理解这样取的方式???  我理解的目标是 是否该视频被推送给了该用户，1表示推送给了该用户，0表示没有推送给该用户
    """

    train_neg_sample_list = []
    test_neg_sample_list = []
    all_movie_list = set(data['movie_id'])
    neg_sample_num = 10

    for i in tqdm(range(len(train_label))):
        a = set(train_model_input['hist_movie_id'][i] + train_model_input['movie_id'][i])
        neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
        train_neg_sample_list.append(np.array(neg_list))
        
    for i in tqdm(range(len(test_label))):
        a = set(test_model_input['hist_movie_id'][i] + test_model_input['movie_id'][i])
        neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
        test_neg_sample_list.append(np.array(neg_list))

    # write to txt
    train = open("train.txt", "w")

    for i in range(len(train_label)):
        a = train_model_input["user_id"][i]
        b = train_model_input["gender"][i]
        c = train_model_input["age"][i]
        d = train_model_input["occupation"][i]
        e = train_model_input["zip"][i]
        
        h = train_model_input["movie_id"][i]
        m = train_neg_sample_list[i]
        
        train.write("%s\t%s\t%s\t%s\t%s\t%s\t1\n"\
                %(str(a), str(b), str(c), str(d), str(e), str(h)))
                
        for x_i in m:
            train.write("%s\t%s\t%s\t%s\t%s\t%s\t0\n"\
                %(str(a), str(b), str(c), str(d), str(e), str(x_i)))
        
    train.close()


    test = open("test.txt", "w")

    for i in range(len(test_label)):
        a = test_model_input["user_id"][i]
        b = test_model_input["gender"][i]
        c = test_model_input["age"][i]
        d = test_model_input["occupation"][i]
        e = test_model_input["zip"][i]
        
        h = test_model_input["movie_id"][i]
        m = test_neg_sample_list[i]
        
        test.write("%s\t%s\t%s\t%s\t%s\t%s\t1\n"\
                %(str(a), str(b), str(c), str(d), str(e), str(h)))
                
        for x_i in m:
            test.write("%s\t%s\t%s\t%s\t%s\t%s\t0\n"\
                %(str(a), str(b), str(c), str(d), str(e), str(x_i)))
    test.close()


def init_output():
    user_id = []
    gender = []
    age = []
    occupation = []
    zip = []
    movie_id = []
    label = []
    return user_id, gender, age, occupation, zip, movie_id, label

#
def file_generator(input_path, batch_size):

    user_id, gender, age, occupation, zip, movie_id, label = init_output()
    cnt = 0
    num_lines = sum([1 for line in open(input_path)])
    while True:
        with open(input_path, 'r') as f:
            for line in f.readlines():

                buf = line.strip().split('\t')

                user_id.append(int(buf[0]))
                gender.append(int(buf[1]))
                age.append(int(buf[2]))
                occupation.append(int(buf[3]))
                zip.append(int(buf[4]))
                movie_id.append(int(buf[5]))
                label.append(int(buf[6]))

                cnt += 1

                if cnt % batch_size == 0 or cnt == num_lines:
                    user_id = np.array(user_id, dtype='int32')
                    gender = np.array(gender, dtype='int32')
                    age = np.array(age, dtype='int32')
                    occupation = np.array(occupation, dtype='int32')
                    zip = np.array(zip, dtype='int32')
                    movie_id = np.array(movie_id, dtype='int32')
                    
                    label = np.array(label, dtype='int32')

                    yield [user_id, gender, age, occupation, zip, movie_id], label

                    user_id, gender, age, occupation, zip, movie_id, label = init_output()

#
class BilinearInteraction(Layer):
    """
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **str** : String, types of bilinear functions used in this layer.
        - **seed** : A Python integer to use as random seed.
    """

    def __init__(self, bilinear_type="each", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])

        if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        if self.bilinear_type == "all":
            p = [tf.multiply(tf.tensordot(v_i, self.W, axes=(-1, 0)), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [tf.multiply(tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError

        return Concatenate(axis=-1)(p)
        

    def compute_output_shape(self, input_shape):
        filed_size = len(input_shape)
        embedding_size = input_shape[0][-1]

        return (None, 1, filed_size * (filed_size - 1) // 2 * embedding_size)

    def get_config(self, ):
        config = {'bilinear_type': self.bilinear_type, 'seed': self.seed}
        base_config = super(BilinearInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#
def BilinearFFM(sparse_input_length=1,embedding_dim=64):
    # 1. Input layer
    user_id_input_layer=Input(shape=(sparse_input_length,),name='user_id_input_layer')
    gender_input_layer = Input(shape=(sparse_input_length, ), name="gender_input_layer")
    age_input_layer = Input(shape=(sparse_input_length, ), name="age_input_layer")
    occupation_input_layer = Input(shape=(sparse_input_length, ), name="occupation_input_layer")
    zip_input_layer = Input(shape=(sparse_input_length, ), name="zip_input_layer")
    item_input_layer = Input(shape=(sparse_input_length, ), name="item_input_layer")
    
    # 2. Embedding layer
    """
    我看了一下mask_zero的概念，对于该概念在推荐场景下的理解的不是非常透彻，
    我理解的是如果mask_zero=True，那么padding部分就不会被计算进去以及反向传播，对应的embedding也不会被更新，因为用tokenizer处理text的时候默认padding为0
    mask_zero文档中提到了说如果
    """
    user_id_embedding_layer = Embedding(6040+1, embedding_dim, mask_zero=True, name='user_id_embedding_layer')(user_id_input_layer)
    gender_embedding_layer = Embedding(2+1, embedding_dim, mask_zero=True, name='gender_embedding_layer')(gender_input_layer)
    age_embedding_layer = Embedding(7+1, embedding_dim, mask_zero=True, name='age_embedding_layer')(age_input_layer)
    occupation_embedding_layer = Embedding(21+1, embedding_dim, mask_zero=True, name='occupation_embedding_layer')(occupation_input_layer)
    zip_embedding_layer = Embedding(3439+1, embedding_dim, mask_zero=True, name='zip_embedding_layer')(zip_input_layer)
    item_id_embedding_layer = Embedding(3706+1, embedding_dim, mask_zero=True, name='item_id_embedding_layer')(item_input_layer)
    
    sparse_embedding_list = [user_id_embedding_layer, gender_embedding_layer, age_embedding_layer, \
                             occupation_embedding_layer, zip_embedding_layer, item_id_embedding_layer]
    
    # 3. Bilinear FFM
    bilinear_out = BilinearInteraction()(sparse_embedding_list)
    
    # Output
    dot_output = tf.nn.sigmoid(tf.reduce_sum(bilinear_out, axis=-1))

    sparse_input_list = [user_id_input_layer, gender_input_layer, age_input_layer, occupation_input_layer, zip_input_layer, item_input_layer]
    model = Model(inputs = sparse_input_list,
                  outputs = dot_output)
    
    return model


if __name__ == "__main__":
    if not os.path.exists('train.txt'):
        data_process()

    train_path = "train.txt"
    val_path = "test.txt"
    batch_size = 1000

    n_train = sum([1 for i in open(train_path)])
    n_val = sum([1 for i in open(val_path)])

    train_steps = n_train / batch_size
    train_steps_ = n_train // batch_size
    validation_steps = n_val / batch_size
    validation_steps_ = n_val // batch_size


    train_generator = file_generator(train_path, batch_size)
    val_generator = file_generator(val_path, batch_size)

    steps_per_epoch = train_steps_ if train_steps==train_steps_ else train_steps_ + 1
    validation_steps = validation_steps_ if validation_steps==validation_steps_ else validation_steps_ + 1

    print("n_train: ", n_train)
    print("n_val: ", n_val)

    print("steps_per_epoch: ", steps_per_epoch)
    print("validation_steps: ", validation_steps)
    

    """
    restore_best_weights:Whether to restore model weights from the epoch with the best value of the monitored quantity. 
    If False, the model weights obtained at the last step of training are used.
    """
    early_stopping_cb=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    min_delta=0.001,
                                                    patience=3,
                                                    restore_best_weights=True)
    callbacks = [early_stopping_cb]
    model=BilinearFFM()
    print(model.summary())
    #Converts a Keras model to dot format and save to a file.
    tf.keras.utils.plot_model(model, to_file='BilinearFFM_model.png', show_shapes=True)

    model.compile(loss='binary_crossentropy', \
        optimizer=Adam(lr=1e-3), \
        metrics=['accuracy'])


    history = model.fit(train_generator, \
                        epochs=2, \
                        steps_per_epoch = steps_per_epoch, \
                        callbacks = callbacks,
                        validation_data = val_generator, \
                        validation_steps = validation_steps, \
                        shuffle=True
                    )
    
    model.save_weights(os.path.join(log_dir,'BilinearFFM_model.h5'))    



