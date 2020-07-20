import argparse
import sys
import tensorflow as tf

FLAGS=None

def init_args(parser):
    parser.add_argument('--batch_size',default=512,type=int)
    parser.add_argument('--embed_size',default=128,type=int)
    parser.add_argument('--user_vocab_size',default=0,type=int)
    parser.add_argument('--item_vocab_size',default=0,type=int)
    parser.add_argument('--epochs',default=1,type=int)
    parser.add_argument('--learning_rate',default=0.0001,type=float)
    parser.add_argument('--train_paths',default=None,type=str,required=True)
    parser.add_argument('--eval_paths',default=None,type=str,required=True)
    parser.add_argument('--test_paths',default=None,type=str,required=True)
    return parser.parse_known_args()

def parse_data(line):
    

def train_input_fn(filenames,batch_size,shuffle_buffer_size,epochs=1):
    filenames=tf.data.Dataset.list_files(filenames)
    dataset = tf.data.TextLineDataset(filenames).shuffle(shuffle_buffer_size).map(parse_data, num_parallel_calls=10).prefetch(500000)

    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    user_embedding=tf.get_variable(
        name='user_embedding',
        shape=[params['user_vocab_size'],params['embed_size']],
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001)
    )
    item_embedding=tf.get_variable(
        name='item_embedding',
        shape=[params['item_vocab_size'],params['embed_size']],
        initializer=tf.random_normal_initializer(neam=0.0,stddev=0.001)
    )
    uid_embed=tf.nn.embedding_lookup(user_embedding,features['uid'])
    item_embed=tf.nn.embedding_lookup(item_embedding,features['item'])
    output=tf.math.multiply(user_embed,item_embed) # [batch_size,embed_size]
    output=tf.reduce_sum(output,1)
    output_prob=tf.sigmoid(output)

    predictions={'output_prob':output_prob,'user_embedding':user_embed}
    
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,export_outputs=export_outputs)
    
    #
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=labels))
    eval_metric_ops = {"auc": tf.metrics.auc(labels, output_prob)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,loss=loss,eval_metric_ops=eval_metric_ops)
    
    #
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,loss=loss,train_op=train_op)

def build_estimator():
    config=tf.estimator.RunConfig(
        tf_random_seed=0,
        save_checkpoints_steps=250,
        save_summary_steps=10,
    )
    params={
        'vocab_size':vocab_size,
        'embed_size':embed_size,
    }
    estimator=tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_path+'/persisted', params=params, config=config)
    return estimator

def train():
    model=build_estimator()
    tr_files=[FLAGS.train_paths]
    va_files = FLAGS.eval_paths
    print("va_files:", va_files)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=300, throttle_secs=300)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return model

def predict(model):
    test_files=FLAGS.test_paths
    predictions=model.predict(input_fn=lambda: train_input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))


def main(_):
    model=train()
    predict(model)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    FLAGS,unparsed=init_args(parser)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)