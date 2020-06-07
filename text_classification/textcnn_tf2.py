import tensorflow as tf
import os
import json

from tensorflow.keras.layers import Embedding,Conv1D,Dense,Dropout, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

"""
本程序中涉及到了如下几点：
1.如何处理文本数据
2.如何写一个自定义Model
3.如何使用单机多卡
4.如何执行回调


References:
https://github.com/ShaneTian/TextCNN/blob/master/train.py
https://www.tensorflow.org/api_docs/python/tf/keras/Input
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
https://www.tensorflow.org/tutorials/distribute/keras
https://www.pythonf.cn/read/100531
https://www.tensorflow.org/guide/keras/custom_layers_and_models
https://lambdalabs.com/blog/tensorflow-2-0-tutorial-04-early-stopping/
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#to_json
https://docs.python.org/3/library/json.html
https://stackoverflow.com/questions/13819496/what-is-different-between-makedirs-and-mkdir-of-os
https://www.tensorflow.org/tutorials/keras/save_and_load?hl=zh-cn
"""

train_path='data/train/part-00000.txt'
val_path='data/valid/part-00000.txt'
test_path='data/test/part-00000.txt'
vocab_file='vocab.pk'


num_classes=10
padding_size=20
batch_size=64
buffer_size=1000
embed_size=128
filter_sizes=[3,4,5]
filter_num=128
dropout_rate=0.5
learning_rate=0.01
epoch=15
regularizers_lambda=0.01

checkpoint_dir='log/checkpoint'
tensorboard_dir='log/tensorboard'
savemodel_dir='log/savemodel'
save_dir='model/textcnn_tf2'
save_path=os.path.join(save_dir,'textcnn_tf2.model')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if not os.path.exists(savemodel_dir):
    os.makedirs(savemodel_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



"""
tf.keras.Input
    shape:not including the batch_size, For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
tf.keras.layers.Reshape 变换维度  和tf.reshape的区别 :貌似是model级别和tensor级别的区别吧
tf.keras.layers.Flatten

下面我用了两种方法来定义TextCNN
"""
def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,filter_sizes, regularizers_lambda, dropout_rate):
    inputs = tf.keras.Input(shape=(feature_size,), name='input_data')
    embed_initer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
    embed = tf.keras.layers.Embedding(vocab_size, embed_size,
                                   embeddings_initializer=embed_initer,
                                   input_length=feature_size,
                                   name='embedding')(inputs)
    pool_outputs=[]
    for filter_size in filter_sizes:
        # [batch_size,input_length,emb_dim] -> [batch_size,input_length-filter_size+1,filter_num]
        conv=tf.keras.layers.Conv1D(filter_num,filter_size,activation='relu',name='conv_{:d}'.format(filter_size))(embed)
        pool_size=feature_size-filter_size+1
        # [batch_size,input_length-filter_size+1,filter_num] -> [batch_size,1,filter_num]
        pool=tf.keras.layers.MaxPool1D(pool_size,name='max_pooling_{:d}'.format(filter_size))(conv)
        pool_outputs.append(tf.squeeze(pool,1)) # [batch_size,filter_num]
#         print('test {pool}:',tf.squeeze(pool,1).shape)
    pool_outputs=tf.keras.layers.concatenate(pool_outputs,axis=1,name='concatenate')
    pool_outputs=tf.keras.layers.Dropout(dropout_rate,name='dropout')(pool_outputs)
    outputs=tf.keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=tf.keras.initializers.constant(0.1),
                                 kernel_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=tf.keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(pool_outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class TextCNN(Model):
    def __init__(self,
                 padding_size,
                 vocab_size,
                 embed_size,
                 filter_num,
                 num_classes,
                 kernel_sizes=[3,4,5],
                 kernel_regularizer=None,
                 last_activation='softmax'):
        super(TextCNN,self).__init__()
        self.padding_size=padding_size
        self.kernel_sizes=kernel_sizes
        self.num_classes=num_classes
        embed_initer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        self.embedding=Embedding(input_dim=vocab_size,
                                 output_dim=embed_size,
                                 input_length=padding_size,
                                 embeddings_initializer=embed_initer,
                                 name='embedding')
        self.conv1s=[]
        self.avgpools=[]
        for kernel_size in kernel_sizes:
            self.conv1s.append(Conv1D(filters=filter_num,kernel_size=kernel_size,activation='relu',kernel_regularizer=kernel_regularizer))
            self.avgpools.append(GlobalMaxPooling1D())
        self.classifier=Dense(num_classes,activation=last_activation,)
    
    def call(self,inputs,training=None,mask=None):
        emb=self.embedding(inputs)
        conv1s=[]
        for i in range(len(self.kernel_sizes)):
            c=self.conv1s[i](emb)
            c=self.avgpools[i](c)
            conv1s.append(c)
        x=Concatenate()(conv1s)
        x=Dropout(dropout_rate,name='dropout')(x)
        output=self.classifier(x)
        return output     
    
    def build_graph(self,input_shape):
        """自定义函数，在调用model.summary()之前调用
        """
        input_shape_nobatch=input_shape[1:]
        self.build(input_shape)
        inputs=tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)
        

"""
tf.keras.preprocessing.text.Tokenizer
num_words: the maximum number of words to keep, based
    on word frequency. Only the most common `num_words-1` words will
    be kept
"""
def data_process(file,padding_size,tokenizer=None,mode='train'):
    X=[]
    y=[]
    # 有时间看一些tensorflow的懒加载
    with open(file,'r') as fr:
        line=fr.readline()
        while line:
            temp=line.strip().split('\t')
            X.append(temp[0])
            y.append(int(temp[1]))
            line=fr.readline()
    if mode=='train':
        # 定义一个分割方式
        tokenizer=tf.keras.preprocessing.text.Tokenizer(char_level=True,oov_token="<UNK>")
        # 使用一批数据取拟合，可以进行增量拟合
        tokenizer.fit_on_texts(X)
    X=tokenizer.texts_to_sequences(X)
    X_pad=tf.keras.preprocessing.sequence.pad_sequences(X,maxlen=padding_size,padding='post',truncating='post')
    # 此处也可以将词典保存下来然后供后面的使用，而不是使用tokenizer这一套
    
    dataset=tf.data.Dataset.from_tensor_slices((X_pad,y)) #return [(feature_1,label_1),(feature_2,label_2)]
    
    return dataset,tokenizer

#
def decay(epoch):
    if epoch < 3:return 1e-3
    elif epoch >= 3 and epoch < 7:return 1e-4
    else:return 1e-5

#
#如何载入该tokenizer对predict的数据进行处理??? 见下面 tokenizer提供了json的转换接口
#train
def train():
    train_dataset,tokenizer=data_process(train_path,padding_size)
    train_dataset=train_dataset.cache().shuffle(buffer_size).batch(batch_size)
    vocab_size=len(tokenizer.word_index)+1 #加1的原因是因为使用tokenizer的时候默认0为padding 
    with open('tokenizer.json','w') as fw:
        json.dump(tokenizer.to_json(),fw)

    valid_dataset,_=data_process(val_path,padding_size,tokenizer=tokenizer,mode='valid')
    valid_dataset=valid_dataset.batch(batch_size)


    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model=TextCNN(padding_size,vocab_size,embed_size,filter_num,num_classes)
    #     model=TextCNN(vocab_size, padding_size, embed_size, num_classes, num_filters,filter_sizes, regularizers_lambda, dropout_rate)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

        #
        class PrintLR(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print('\nLearning rate for epoch {} is {}'.format(epoch + 1,model.optimizer.lr.numpy()))        

        """
        EarlyStopping：
        monitor keep track of the quantity that is used to decide if the training should be terminated. In this case, we use the validation accuracy.
        min_delta is the threshold that triggers the termination.
        In this case, we require that the accuracy should at least improve 0.0001. 
        patience is the number of "no improvement epochs" to wait until training is stopped. 
        With patience = 1, training terminates immediately after the first epoch with no improvement.
        monitor监控哪个数据的哪个指标，min_delta为该值的差值，patience表示忍受的次数
        """
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
            tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True),
            tf.keras.callbacks.LearningRateScheduler(decay),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0.001,patience=3),
            PrintLR()
        ]

        model.fit(train_dataset, epochs=epoch,validation_data=valid_dataset, callbacks=callbacks)

#
#predict
# strategy.scope方式：https://www.tensorflow.org/tutorials/distribute/keras
def predict():
    with open('tokenizer.json','r') as fr:
        tokenizer=tf.keras.preprocessing.text.tokenizer_from_json(json.load(fr))

    test_dataset,_=data_process(test_path,padding_size,tokenizer=tokenizer,mode='test')
    test_dataset=test_dataset.batch(batch_size)

    # 如何导入验证集下最优的模型ckeckpoint，而不是最近的checkpoint
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    test_loss,test_acc=model.evaluate(test_dataset)
    print('Test loss: {}, Test Accuracy: {}'.format(test_loss, test_acc))

if __name__ == "__main__":
    train()
    # predict()

