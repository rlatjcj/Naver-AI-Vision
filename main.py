# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from load_data import *
from metrics import precision, recall, f1
from model import *
from callbacks import *

SEED = 777
np.random.seed(SEED)


def bind_model(model, config):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db, config)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        print('inference start')
        if config.train_mode == 'siamese':
            batch = 256
            sim_matrix = np.zeros((len(query_img), len(reference_img)))
            train_q = np.empty((batch, config.shape, config.shape, 3))
            train_r = np.empty((batch, config.shape, config.shape, 3))
            for q, qi in enumerate(query_img):
                print('**********', q+1, '**********')
                # qi = qi[np.newaxis,...]
                i = 0
                cnt = 0
                for ri in reference_img:
                    train_q[i] = qi
                    train_r[i] = ri
                    # ri = ri[np.newaxis,...]
                    # if i == 0:
                    #     train_q = qi
                    #     train_r = ri
                    # else:
                    #     train_q = np.concatenate((train_q, qi), axis=0)
                    #     train_r = np.concatenate((train_r, ri), axis=0)
                    i += 1

                    if i == batch:
                        score = model.predict_on_batch([train_q, train_r])[:,0]
                        score = score.flatten()
                        sim_matrix[q][cnt:cnt+batch] = score
                        cnt += batch
                        i = 0

                score = model.predict_on_batch([train_q, train_r])[:,0]
                score = score.flatten()
                sim_matrix[q][cnt:] = score[:i]

        elif config.train_mode == 'classification':
            get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])
            # inference
            batch = 10
            for i in range(len(query_img)//batch):
                if i == 0:
                    query_vecs = get_feature_layer([query_img[:batch], 0])[0]
                else:
                    query_vecs = np.concatenate((query_vecs, get_feature_layer([query_img[i*batch:(i+1)*batch], 0])[0]))

            if len(query_img) % batch != 0:
                query_vecs = np.concatenate((query_vecs, get_feature_layer([query_img[(i+1)*batch:], 0])[0]))

            query_vecs = query_vecs.reshape(query_vecs.shape[0], query_vecs.shape[1] * query_vecs.shape[2] * query_vecs.shape[3])

            # caching db output, db inference
            db_output = './db_infer.pkl'       
            if os.path.exists(db_output):
                with open(db_output, 'rb') as f:
                    reference_vecs = pickle.load(f)
            else:
                for i in range(len(reference_img)//batch):
                    if i == 0:
                        reference_vecs = get_feature_layer([reference_img[:batch], 0])[0]
                    else:
                        reference_vecs = np.concatenate((reference_vecs, get_feature_layer([reference_img[i*batch:(i+1)*batch], 0])[0]))

                if len(reference_vecs) % batch != 0:
                    reference_vecs = np.concatenate((reference_vecs, get_feature_layer([reference_img[(i+1)*batch:], 0])[0]))

                reference_vecs = reference_vecs.reshape(reference_vecs.shape[0], reference_vecs.shape[1] * reference_vecs.shape[2] * reference_vecs.shape[3])
                with open(db_output, 'wb') as f:
                    pickle.dump(reference_vecs, f)

            # l2 normalization
            query_vecs = l2_normalize(query_vecs)
            reference_vecs = l2_normalize(reference_vecs)

            # Calculate cosine similarity
            print(query_vecs.shape, reference_vecs.shape, reference_vecs.T.shape)
            sim_matrix = np.dot(query_vecs, reference_vecs.T)


        retrieval_results = {}
        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db, config):
    query_img = []
    reference_img = []
    img_size = (config.shape, config.shape)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--learning_rate', type=float, default=0.00045)
    args.add_argument('--optimizer', type=str, default='rmsprop')
    args.add_argument('--train_mode', type=str, default='classification', metavar="classification / siamese / rmac")
    args.add_argument('--model', type=str, default='vgg', metavar="vgg / xception")
    args.add_argument('--shape', type=int, default=256)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    epochs = config.epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    classes = 1000
    input_shape = (config.shape, config.shape, 3)  # input image shape

    """ Model """
    if config.model == 'vgg':
        if config.train_mode == 'classification':
            model = VGG(input_shape=input_shape, classes=classes)

        elif config.train_mode == 'siamese':
            vgg = VGG(input_shape=input_shape, classes=classes, mode='siamese')
            model = Siamese(input_shape=input_shape, model=vgg)

        elif config.train_mode == 'rmac':
            from get_regions import rmac_regions, get_size_vgg_feat_map
            vgg = VGG(input_shape=input_shape, classes=classes, mode='rmac')
            Wmap, Hmap = get_size_vgg_feat_map(config.shape, config.shape)
            regions = rmac_regions(Wmap, Hmap, 3)
            model = RMAC(input_shape, vgg, len(regions))


    elif config.model == 'xception':
        if config.train_mode == 'classification':
            model = Xception(input_shape=input_shape, classes=classes)

        elif config.train_mode == 'siamese':
            xception = Xception(input_shape=input_shape, classes=classes, mode='siamese')
            model = Siamese(input_shape=input_shape, model=xception)


    bind_model(model, config)

    if config.pause:
        nsml.paused(scope=locals())

    if config.checkpoint:
        model.load_weights(config.checkpoint, by_name=True, skip_mismatch=True)

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True   

        """ Initiate RMSprop optimizer """
        if config.optimizer == 'rmsprop':
            opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)
        elif config.optimizer == 'adam':
            opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
        elif config.optimizer == 'sgd':
            opt = keras.optimizers.sgd(lr=learning_rate, decay=1e-6)

        """ Load data """
        if not nsml.IS_ON_NSML:
            model.summary()
            DATASET_PATH = './dataset'

        print('dataset path', DATASET_PATH)
        # output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        # if nsml.IS_ON_NSML:
        #     # Caching file
        #     nsml.cache(train_load1, data_path=train_dataset_path, img_size=input_shape[:2], output_path=output_path)
        # else:
        #     # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
        #     train_load1(train_dataset_path, input_shape[:2], output_path=output_path)

        # with open(output_path[0], 'rb') as img_f:
        #     img_list = pickle.load(img_f)
        # with open(output_path[1], 'rb') as label_f:
        #     label_list = pickle.load(label_f)

        # x_train = np.asarray(img_list)
        # labels = np.asarray(label_list)
        # y_train = keras.utils.to_categorical(labels, num_classes=classes)
        # x_train = x_train.astype('float32')
        # x_train /= 255
        # print(len(labels), 'train samples')

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        custom_nsml = CustomNSML(epochs)

        callbacks = [reduce_lr, custom_nsml]


        if config.train_mode == 'classification':
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy', precision, recall, f1])

            """ Generator """
            generator = ImageDataGenerator(rescale=1./255,
                                        horizontal_flip=True,
                                        vertical_flip=True)

            train_generator = generator.flow_from_directory(directory=train_dataset_path,
                                                            target_size=input_shape[:2],
                                                            class_mode='categorical',
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            seed=SEED,
                                                            subset='training',
                                                            interpolation='bilinear')

            print("━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Dataset Path       ┃   "+str(train_dataset_path))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Epochs             ┃   "+str(epochs))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Optimizer          ┃   "+config.optimizer)
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Learning Rate      ┃   "+str(learning_rate))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Batch Size         ┃   "+str(batch_size))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Input Shape        ┃   "+str(input_shape))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Train Samples      ┃   "+str(train_generator.samples))
            print("━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            """ Training loop """
            for epoch in range(epochs):
                res = model.fit_generator(train_generator,
                                            steps_per_epoch=int(train_generator.samples//batch_size),
                                            epochs=epoch+1,
                                            initial_epoch=epoch,
                                            max_queue_size=batch_size,
                                            callbacks=callbacks,
                                            verbose=1,
                                            shuffle=True)

                print(res.history)
                train_loss, train_acc, train_precision, train_recall, train_f1 = res.history['loss'][0], res.history['acc'][0], res.history['precision'][0], res.history['recall'][0], res.history['f1'][0]
                nsml.report(summary=True, epoch=epoch, epoch_total=epochs, loss=train_loss, acc=train_acc, precision=train_precision, recall=train_recall, f1=train_f1)
                nsml.save(epoch)
        
        elif config.train_mode == 'siamese':
            model.compile(loss='binary_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy', recall])

            """ Generator """
            datalist = os.listdir(train_dataset_path)
            train_generator = siamese_generator(train_dataset_path, datalist, batch_size, input_shape)

            """ Train Siamese Network """
            print("━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Dataset Path       ┃   "+str(train_dataset_path))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Epochs             ┃   "+str(epochs))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Optimizer          ┃   "+config.optimizer)
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Learning Rate      ┃   "+str(learning_rate))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Input Shape        ┃   "+str(input_shape))
            print("━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("  Batch Size         ┃   "+str(batch_size))
            print("━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # for epoch in range(epochs):
            #     print("---------------------")
            #     print("     ", epoch+1, "Epoch")
            #     print("---------------------")

            res = model.fit_generator(train_generator,
                                        steps_per_epoch=100,
                                        epochs=epochs,
                                        initial_epoch=0,
                                        callbacks=callbacks,
                                        verbose=1,
                                        shuffle=True)

                # train_loss, train_acc, train_recall = res.history['loss'][0], res.history['acc'][0], res.history['recall'][0]
                # nsml.report(summary=True, epoch=epoch, epoch_total=epochs, loss=train_loss, acc=train_acc, recall=train_recall)
                # nsml.save(epoch)

        # elif config.train_mode == 'rmac':




    else:
        print("******************** test ********************")
        if not nsml.IS_ON_NSML:
            def preprocess1(queries, db):
                query_img = []
                reference_img = []
                img_size = (config.shape, config.shape)

                for img_path in queries:
                    img = cv2.imread(os.path.join('./dataset/train/val_data', img_path), 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    query_img.append(img)

                for img_path in db:
                    files = os.listdir('./dataset/train/train_data/{}'.format(img_path))
                    p = np.random.permutation(len(files))
                    img = cv2.imread(os.path.join('./dataset/train/train_data/{}/{}'.format(img_path,files[p[0]])), 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    reference_img.append(img)

                return queries, query_img, db, reference_img

            queries = os.listdir('./dataset/train/val_data')[:50]
            db = os.listdir('./dataset/train/train_data')

            queries, query_img, references, reference_img = preprocess1(queries, db)

            print('test data load queries {} query_img {} references {} reference_img {}'.
                format(len(queries), len(query_img), len(references), len(reference_img)))

            queries = np.asarray(queries)
            query_img = np.asarray(query_img)
            references = np.asarray(references)
            reference_img = np.asarray(reference_img)

            query_img = query_img.astype('float32')
            query_img /= 255
            reference_img = reference_img.astype('float32')
            reference_img /= 255

            print('inference start')

            if config.train_mode == 'siamese':
                batch = 64
                sim_matrix = np.zeros((len(query_img), len(reference_img)))

                import time
                start = time.time()
                train_q = np.empty((batch, config.shape, config.shape, 3))
                train_r = np.empty((batch, config.shape, config.shape, 3))
                for q, qi in enumerate(query_img):
                    print('**********', q+1, '**********')
                    # qi = qi[np.newaxis,...]
                    i = 0
                    cnt = 0
                    for ri in reference_img:
                        train_q[i] = qi
                        train_r[i] = ri
                        # ri = ri[np.newaxis,...]
                        # if i == 0:
                        #     train_q = qi
                        #     train_r = ri
                        # else:
                        #     train_q = np.concatenate((train_q, qi), axis=0)
                        #     train_r = np.concatenate((train_r, ri), axis=0)
                        i += 1

                        if i == batch:
                            score = model.predict_on_batch([train_q, train_r])[:,0]
                            score = score.flatten()
                            sim_matrix[q][cnt:cnt+batch] = score
                            cnt += batch
                            i = 0

                    score = model.predict_on_batch([train_q, train_r])[:,0]
                    print(score.shape)
                    score = score.flatten()
                    sim_matrix[q][cnt:] = score[:i]

                print(time.time() - start)

            elif config.train_mode == 'classification':
                get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-4].output])
                # inference
                batch = 10
                for i in range(len(query_img)//batch):
                    if i == 0:
                        query_vecs = get_feature_layer([query_img[:batch], 0])[0]
                    else:
                        query_vecs = np.concatenate((query_vecs, get_feature_layer([query_img[i*batch:(i+1)*batch], 0])[0]))

                if len(query_img) % batch != 0:
                    query_vecs = np.concatenate((query_vecs, get_feature_layer([query_img[(i+1)*batch:], 0])[0]))

                query_vecs = query_vecs.reshape(query_vecs.shape[0], query_vecs.shape[1] * query_vecs.shape[2] * query_vecs.shape[3])

                # caching db output, db inference
                db_output = './db_infer.pkl'       
                if os.path.exists(db_output):
                    with open(db_output, 'rb') as f:
                        reference_vecs = pickle.load(f)
                else:
                    for i in range(len(reference_img)//batch):
                        if i == 0:
                            reference_vecs = get_feature_layer([reference_img[:batch], 0])[0]
                        else:
                            reference_vecs = np.concatenate((reference_vecs, get_feature_layer([reference_img[i*batch:(i+1)*batch], 0])[0]))

                    if len(reference_vecs) % batch != 0:
                        reference_vecs = np.concatenate((reference_vecs, get_feature_layer([reference_img[(i+1)*batch:], 0])[0]))

                    reference_vecs = reference_vecs.reshape(reference_vecs.shape[0], reference_vecs.shape[1] * reference_vecs.shape[2] * reference_vecs.shape[3])
                    with open(db_output, 'wb') as f:
                        pickle.dump(reference_vecs, f)

                # l2 normalization
                query_vecs = l2_normalize(query_vecs)
                reference_vecs = l2_normalize(reference_vecs)
                print(query_vecs.shape, reference_vecs.shape, reference_vecs.T.shape)

                # Calculate cosine similarity
                qq = [int(q.split('.')[0]) for q in queries]
                sim_matrix = np.dot(query_vecs, reference_vecs.T)

            retrieval_results = {}
            for (i, query) in enumerate(queries):
                query = query.split('/')[-1].split('.')[0]
                # print('query :', query)
                sim_list = zip(references, sim_matrix[i].tolist())
                sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)
                # print(sorted_sim_list)

                ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list
                # print('ranked_list :', ranked_list)

                retrieval_results[query] = ranked_list
            
            print(retrieval_results)
            print('done')

            # print(list(zip(range(len(retrieval_results)), retrieval_results.items())))