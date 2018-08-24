import config
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import glob
import random
import numpy as np
import tensorflow as tf
import logging
import os

def get_logger(filepath,level=logging.INFO):
    dir = os.path.dirname(filepath)
    if not os.path.exists(dir):
        os.mkdir(dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # create a file handler
    handler = logging.FileHandler(filepath)
    handler.setLevel(logging.INFO)

    # create a logging format
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    return logger

#初始化sess,或回复保存的sess
def start_or_restore_training(sess,saver,checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
        sess.run(tf.global_variables_initializer())
        step = 1
        print('start training from new state')
    return sess,step

def get_feature(wav_path,num_frame=64):
    rate, sig = wav.read(wav_path)
    #feature = logfbank(sig, rate)
    full_feature = mfcc(sig, rate)
    if num_frame<len(full_feature):
        offset = random.randint(num_frame,len(full_feature))
        select_feature = full_feature[offset-num_frame:offset]
    else:
        select_feature = full_feature

    return np.expand_dims(select_feature,axis=2)

def get_speaker_dict(train_dir):
    wav_paths = glob.glob(train_dir + "/*/*.wav")
    speaker_dict = {}
    for path in wav_paths:
        p_list = path.split("/")
        if not p_list[-2] in speaker_dict.keys():
            speaker_dict[p_list[-2]] = [p_list[-1]]
        else:
            speaker_dict[p_list[-2]].append(p_list[-1])
    return speaker_dict

def constract_batchpaths_labels(train_dir, speaker_dict,speaker_ids, batch_size):
    sample_ids = random.sample(speaker_ids, batch_size)
    first_batch_paths = []
    second_batch_paths = []
    labels = []
    for id in sample_ids:
        if random.random() > 0.5:
            same_id_sounds = random.sample(speaker_dict[id], 2)
            first_batch_paths.append("%s/%s/%s" % (train_dir, id, same_id_sounds[0]))
            second_batch_paths.append("%s/%s/%s" % (train_dir, id, same_id_sounds[1]))
            labels.append(1)
        else:
            same_sound = random.sample(speaker_dict[id], 1)[0]

            diff_id = random.sample(speaker_ids, 1)[0]
            diff_sound = random.sample(speaker_dict[diff_id], 1)[0]

            first_batch_paths.append("%s/%s/%s" % (train_dir, id, same_sound))
            second_batch_paths.append("%s/%s/%s" % (train_dir, diff_id, diff_sound))
            labels.append(0)

    return first_batch_paths,second_batch_paths,labels

def train_generator(train_dir,batch_size,is_train=True):
    speaker_dict = get_speaker_dict(train_dir)
    speaker_ids = list(speaker_dict.keys())
    if is_train:
        speaker_ids = speaker_ids[:-50]
    else:
        speaker_ids = speaker_ids[-50:]

    while True:
        first_batch_paths,second_batch_paths, labels = constract_batchpaths_labels(train_dir, speaker_dict,speaker_ids,batch_size)

        first_batch_features = [get_feature(path) for path in first_batch_paths]
        second_batch_features = [get_feature(path) for path in second_batch_paths]

        yield np.array(first_batch_features), np.array(second_batch_features), np.array(labels)

def random_sample_paths(train_dir,batch_size,speaker_dict,speaker_ids,speaker_label_dict):
    feature_paths = []
    labels = []
    for i in batch_size:
        id = random.sample(speaker_ids, 1)[0]
        feature_file = random.sample(speaker_dict[id])[0]
        feature_paths.append("%s/%s/%s"%(train_dir,id,feature_file))
        labels.append(speaker_label_dict[id])
    return feature_paths,labels

def train_generator_softmax(train_dir,batch_size):
    speaker_dict = get_speaker_dict(train_dir)
    speaker_ids = list(speaker_dict.keys())
    speaker_label_dict = {}
    for i,id in enumerate(speaker_ids):
        speaker_label_dict[id] = i

    while True:
        feature_paths,labels = random_sample_paths(train_dir, batch_size, speaker_dict, speaker_ids, speaker_label_dict)
        features = [get_feature(path) for path in feature_paths]

        yield np.array(features),np.array(labels)
