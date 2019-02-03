# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import cv2

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
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

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        if config.train_mode == "classification":
            queries, query_vecs, references, reference_vecs = get_feature(model, queries, db, config)
            
            # l2 normalization
            # query_feature1 = l2_normalize(query_vecs[0])
            # query_feature2 = l2_normalize(query_vecs[1])
            # query_feature3 = l2_normalize(query_vecs[2])
            
            # query_feature2 = query_feature2.reshape(query_feature2.shape[0], query_feature2.shape[1] * query_feature2.shape[2] * query_feature2.shape[3])
            # query_feature3 = query_feature3.reshape(query_feature3.shape[0], query_feature3.shape[1] * query_feature3.shape[2] * query_feature3.shape[3])

            # reference_feature1 = l2_normalize(reference_vecs[0])
            # reference_feature2 = l2_normalize(reference_vecs[1])
            # reference_feature3 = l2_normalize(reference_vecs[2])

            # reference_feature2 = reference_feature2.reshape(reference_feature2.shape[0], reference_feature2.shape[1] * reference_feature2.shape[2] * reference_feature2.shape[3])
            # reference_feature3 = reference_feature3.reshape(reference_feature3.shape[0], reference_feature3.shape[1] * reference_feature3.shape[2] * reference_feature3.shape[3])

            # Calculate cosine similarity
            # print(query_feature1.shape, reference_feature1.shape, reference_feature1.T.shape)
            # print(query_feature2.shape, reference_feature2.shape, reference_feature2.T.shape)
            # print(query_feature3.shape, reference_feature3.shape, reference_feature3.T.shape)
            # sim_matrix1 = np.dot(query_feature1, reference_feature1.T)
            # sim_matrix2 = np.dot(query_feature2, reference_feature2.T)
            # sim_matrix3 = np.dot(query_feature3, reference_feature3.T)

            # sim_matrix = sim_matrix1 + sim_matrix2 + sim_matrix3

            query_feature = l2_normalize(query_vecs)
            reference_feature = l2_normalize(reference_vecs)
            sim_matrix = np.dot(query_feature, reference_feature.T)

            indices = np.argsort(sim_matrix, axis=1)
            indices = np.flip(indices, axis=1)

        elif config.train_mode == 'siamese':
            batch = 256
            sim_matrix = np.zeros((len(queries), len(db)))
            train_q = np.empty((batch, config.shape, config.shape, 3))
            train_r = np.empty((batch, config.shape, config.shape, 3))
            for q in len(queries):
                print('**********', q+1, '**********')
                query = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(test_path, q), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255
                
                i = 0
                cnt = 0
                for d in len(db):
                    train_q[i] = qi
                    train_r[i] = ri
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

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

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
def get_feature(model, queries, db, config):
    batch_size = 4
    target_size = (config.shape, config.shape)
    test_path = DATASET_PATH + '/test/test_data'

    test_datagen = ImageDataGenerator(rescale=1. / 255, dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=target_size,
        classes=['query'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=target_size,
        classes=['reference'],
        color_mode="rgb",
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    if config.train_mode == "classification":
        intermediate_layer_model = Model(inputs=model.input, outputs=[model.layers[-1].output, model.layers[-4].output, model.layers[-7].output])
        query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)
        reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator), verbose=1)

    return queries, query_vecs, db, reference_vecs


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=30)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)
    args.add_argument('--checkpoint', type=str, default=None)
    args.add_argument('--learning_rate', type=float, default=0.00045)
    args.add_argument('--optimizer', type=str, default='rmsprop')
    args.add_argument('--train_mode', type=str, default='classification', metavar="classification / siamese / triple")
    args.add_argument('--model', type=str, default='vgg', metavar="vgg / xception")
    args.add_argument('--shape', type=int, default=256)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    epochs = config.epochs
    batch_size = config.batch_size
    num_classes = config.num_classes
    learning_rate = config.learning_rate
    input_shape = (config.shape, config.shape, 3)  # input image shape

    """ Model """
    if config.model == 'vgg':
        if config.train_mode == 'classification':
            model = VGG(input_shape=input_shape, classes=num_classes)

        elif config.train_mode == 'siamese':
            vgg = VGG(input_shape=input_shape, classes=num_classes, mode='siamese')
            model = Siamese(input_shape=input_shape, model=vgg)

        elif config.train_mode == 'triple':
            from get_regions import rmac_regions, get_size_vgg_feat_map
            vgg = VGG(input_shape=input_shape, classes=num_classes, mode='rmac')
            Wmap, Hmap = get_size_vgg_feat_map(config.shape, config.shape)
            regions = rmac_regions(Wmap, Hmap, 3)
            rmac = RMAC(input_shape, vgg, len(regions))
            model = Triple_Siamese((256, 256, 3), rmac, len(regions))


    elif config.model == 'xception':
        if config.train_mode == 'classification':
            model = Xception(input_shape=input_shape, classes=num_classes)

        elif config.train_mode == 'siamese':
            xception = Xception(input_shape=input_shape, classes=num_classes, mode='siamese')
            model = Siamese(input_shape=input_shape, model=xception)

    bind_model(model, config)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        # nsml.load(checkpoint='13', session='GOOOAL/ir_ph2/19')
        # nsml.save('for_test')
        # exit()

        """ Initiate optimizer """
        if config.optimizer == 'rmsprop':
            opt = keras.optimizers.rmsprop(lr=learning_rate, decay=1e-6)
        elif config.optimizer == 'adam':
            opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
        elif config.optimizer == 'sgd':
            opt = keras.optimizers.sgd(lr=learning_rate, decay=1e-6)

        print('dataset path', DATASET_PATH)
        train_dataset_path = DATASET_PATH + '/train/train_data'

        """ Callback """
        monitor = 'acc'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        custom_nsml = CustomNSML(num_classes)

        callbacks = [reduce_lr]

        # train_datagen = ImageDataGenerator(
        #     rescale=1. / 255,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     horizontal_flip=True)

        # train_generator = train_datagen.flow_from_directory(
        #     directory=DATASET_PATH + '/train/train_data',
        #     target_size=input_shape[:2],
        #     color_mode="rgb",
        #     batch_size=batch_size,
        #     class_mode="categorical",
        #     shuffle=True,
        #     seed=42
        # )

        # """ Callback """
        # monitor = 'acc'
        # reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        # """ Training loop """
        # STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        # t0 = time.time()
        # for epoch in range(nb_epoch):
        #     t1 = time.time()
        #     res = model.fit_generator(generator=train_generator,
        #                               steps_per_epoch=STEP_SIZE_TRAIN,
        #                               initial_epoch=epoch,
        #                               epochs=epoch + 1,
        #                               callbacks=[reduce_lr],
        #                               verbose=1,
        #                               shuffle=True)
        #     t2 = time.time()
        #     print(res.history)
        #     print('Training time for one epoch : %.1f' % ((t2 - t1)))
        #     train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
        #     nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
        #     nsml.save(epoch)
        # print('Total training time : %.1f' % (time.time() - t0))

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
            STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
            t0 = time.time()
            for epoch in range(epochs):
                print("---------------------")
                print("     ", epoch+1, "Epoch")
                print("---------------------")
                t1 = time.time()
                res = model.fit_generator(train_generator,
                                            steps_per_epoch=STEP_SIZE_TRAIN,
                                            epochs=epoch+1,
                                            initial_epoch=epoch,
                                            max_queue_size=batch_size,
                                            callbacks=callbacks,
                                            verbose=1,
                                            shuffle=True)
                t2 = time.time()
                print(res.history)
                print('Training time for one epoch : %.1f' % ((t2 - t1)))
                train_loss, train_acc, train_precision, train_recall, train_f1 = res.history['loss'][0], res.history['acc'][0], res.history['precision'][0], res.history['recall'][0], res.history['f1'][0]
                nsml.report(summary=True, epoch=epoch, epoch_total=epochs, loss=train_loss, acc=train_acc, precision=train_precision, recall=train_recall, f1=train_f1)
                nsml.save(epoch)
            print('Total training time : %.1f' % (time.time() - t0))

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

            t0 = time.time()
            for epoch in range(epochs):
                print("---------------------")
                print("     ", epoch+1, "Epoch")
                print("---------------------")
                t1 = time.time()

                res = model.fit_generator(train_generator,
                                            steps_per_epoch=1000,
                                            epochs=epoch+1,
                                            initial_epoch=epoch,
                                            callbacks=callbacks,
                                            verbose=1,
                                            shuffle=True)

                t2 = time.time()
                print(res.history)
                print('Training time for one epoch : %.1f' % ((t2 - t1)))
                train_loss, train_acc, train_precision, train_recall, train_f1 = res.history['loss'][0], res.history['acc'][0], res.history['precision'][0], res.history['recall'][0], res.history['f1'][0]
                nsml.report(summary=True, epoch=epoch, epoch_total=epochs, loss=train_loss, acc=train_acc, precision=train_precision, recall=train_recall, f1=train_f1)
                nsml.save(epoch)
            print('Total training time : %.1f' % (time.time() - t0))