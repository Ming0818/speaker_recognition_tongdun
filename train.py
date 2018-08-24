import numpy as np
import tensorflow as tf
import time

from model import convolutional_model,sigmoid_loss
import utils
import config

x1 = tf.placeholder(dtype=tf.float32, shape=[None,64,13,1])
x2 = tf.placeholder(dtype=tf.float32, shape=[None,64,13,1])

y = tf.placeholder(dtype=tf.float32,shape=[None,1])

embedding_1 = convolutional_model(x1)
embedding_2 = convolutional_model(x2)
s_loss = sigmoid_loss(embedding_1,embedding_2,y)

sigmoid_loss_train_step = tf.train.AdamOptimizer().minimize(s_loss)

train_gen = utils.train_generator(config.TRAINDIR,config.BATCH_SIZE,is_train=True)
val_gen = utils.train_generator(config.TRAINDIR,config.BATCH_SIZE,is_train=False)

saver = tf.train.Saver()
logger = utils.get_logger("./log/info.log")

with tf.Session() as sess:
    sess, step = utils.start_or_restore_training(sess, saver, checkpoint_dir=config.CHECKDIR)

    start_time = time.time()
    print("trainning......")

    while True:
        train_features_1, train_features_2, train_labels = next(train_gen)
        sess.run(sigmoid_loss_train_step,feed_dict={x1:train_features_1, x2:train_features_2, y:train_labels})

        step +=1
        if step%10 == 0:
            train_loss = sess.run(sigmoid_loss_train_step, feed_dict={x1: train_features_1, x2: train_features_2, y: train_labels})
            val_features_1,val_features_2,val_labels = next(val_gen)
            valid_loss = sess.run(s_loss, feed_dict={x1: val_features_1, x2:val_features_2, y: val_labels})

            duration = time.time() - start_time

            logger.info("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (
            step, train_loss, valid_loss, duration))

            print("step %d: trainning loss is %g, validation loss is %g (%0.3f sec)" % (
            step, train_loss, valid_loss, duration))
        if step%100 == 0:
            saver.save(sess, config.CHECKFILE, global_step=step)
            saver.save(sess, config.CHECKFILE_LONGTERM)
            print('writing checkpoint at step %s' % step)

