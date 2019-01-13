# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle
import random
import numpy as np

def train_load1(data_path, img_size, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = cv2.imread(img_path, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
            except:
                continue
            label_list.append(label_idx)
            img_list.append(img)
        label_idx += 1

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)


# nsml test_data_loader
def test_data_loader(data_path):
    data_path = os.path.join(data_path, 'test', 'test_data')

    # return full path
    queries_path = [os.path.join(data_path, 'query', path) for path in os.listdir(os.path.join(data_path, 'query'))]
    references_path = [os.path.join(data_path, 'reference', path) for path in
                       os.listdir(os.path.join(data_path, 'reference'))]

    return queries_path, references_path


def siamese_loader(train_dataset_path, data_list, order, input_shape):
    datalist = data_list.copy()
    target_folder = os.path.join(train_dataset_path, datalist[order])
    datalist.pop(order)
    target_list = os.listdir(target_folder)
    random.shuffle(target_list)
    target_name = target_list[0]
    len_same_class = len(target_list[1:])
    
    pair = [np.zeros((len_same_class*2, input_shape[0], input_shape[1], 3)) for i in range(2)]
    target = np.zeros((len_same_class*2, 2))

    target_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(target_folder, target_name), 1), cv2.COLOR_RGB2BGR), input_shape) / 255

    for i in range(len_same_class):
        flag = random.random()
        compare_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(target_folder, target_list[i+1]), 1), cv2.COLOR_RGB2BGR), input_shape) / 255
        if flag > 0.75:
            target_img = cv2.flip(target_img, 0)
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 0)
            compare_img = cv2.flip(compare_img, 1)
        elif flag > 0.5:
            target_img = cv2.flip(target_img, 0)
            compare_img = cv2.flip(compare_img, 0)
        elif flag > 0.25:
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 1)

        pair[0][i] = target_img
        pair[1][i] = compare_img
        target[i][0] = 1
        # print(i, os.path.join(target_folder, target_img), os.path.join(target_folder, target_list[i+1]), target[i])

    random.shuffle(datalist)
    for i in range(len_same_class, len_same_class*2):
        flag = random.random()
        dif_class = os.listdir(os.path.join(train_dataset_path, datalist[i]))
        random.shuffle(dif_class)
        compare_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(train_dataset_path, datalist[i], dif_class[0]), 1), cv2.COLOR_RGB2BGR), input_shape) / 255
        if flag > 0.75:
            target_img = cv2.flip(target_img, 0)
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 0)
            compare_img = cv2.flip(compare_img, 1)
        elif flag > 0.5:
            target_img = cv2.flip(target_img, 0)
            compare_img = cv2.flip(compare_img, 0)
        elif flag > 0.25:
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 1)

        pair[0][i] = target_img
        pair[1][i] = compare_img
        target[i][1] = 1
        # print(i, os.path.join(target_folder, target_img), os.path.join(train_dataset_path, datalist[i], dif_class[0]), target[i])

    p = np.random.permutation(len_same_class*2)
    pair[1] = pair[1][p]
    target = target[p]

    return pair, target


def siamese_generator(train_dataset_path, data_list, batch_size, input_shape):
    def flip_img(target_img, compare_img, flag):
        if flag > 0.75:
            target_img = cv2.flip(target_img, 0)
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 0)
            compare_img = cv2.flip(compare_img, 1)
        elif flag > 0.5:
            target_img = cv2.flip(target_img, 0)
            compare_img = cv2.flip(compare_img, 0)
        elif flag > 0.25:
            target_img = cv2.flip(target_img, 1)
            compare_img = cv2.flip(compare_img, 1)

        return target_img, compare_img

    while True:
        pair = [np.zeros((batch_size,)+input_shape) for i in range(2)]
        target = np.zeros((batch_size, 2))
        p = np.random.permutation(len(data_list))
        for i in range(batch_size//2):
            flag = random.random()
            target_folder = os.path.join(train_dataset_path, data_list[p[i]])
            target_list = os.listdir(target_folder)
            random.shuffle(target_list)

            target_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(target_folder, target_list[0]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255
            compare_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(target_folder, target_list[1]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255

            pair[0][i], pair[1][i] = flip_img(target_img, compare_img, flag)
            target[i][0] = 1

        for i in range(batch_size//2, batch_size):
            flag = random.random()
            target_folder = os.path.join(train_dataset_path, data_list[p[i]])
            target_list = os.listdir(target_folder)
            compare_folder = os.path.join(train_dataset_path, data_list[p[i+batch_size]])
            compare_list = os.listdir(compare_folder)
            random.shuffle(target_list)
            random.shuffle(compare_list)

            target_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(target_folder, target_list[0]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255
            compare_img = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(compare_folder, compare_list[0]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255

            pair[0][i], pair[1][i] = flip_img(target_img, compare_img, flag)
            target[i][1] = 1

        p = np.random.permutation(batch_size)
        pair[1] = pair[1][p]
        target = target[p]

        yield pair, target
        # yield pair[0].shape, pair[1].shape, target.shape


def triple_generator(train_dataset_path, data_list, batch_size, input_shape, regions):
    def flip_img(query, relevant, irrelevant, flag):
        if flag > 0.75:
            query = cv2.flip(query, 0)
            query = cv2.flip(query, 1)
            relevant = cv2.flip(relevant, 0)
            relevant = cv2.flip(relevant, 1)
            irrelevant = cv2.flip(irrelevant, 0)
            irrelevant = cv2.flip(irrelevant, 1)
        elif flag > 0.5:
            query = cv2.flip(query, 0)
            relevant = cv2.flip(relevant, 0)
            irrelevant = cv2.flip(irrelevant, 0)
        elif flag > 0.25:
            query = cv2.flip(query, 1)
            relevant = cv2.flip(relevant, 1)
            irrelevant = cv2.flip(irrelevant, 1)

        return query, relevant, irrelevant

    while True:
        pair = [np.zeros((batch_size,)+input_shape) for i in range(3)]
        target = np.zeros((batch_size, 1))
        p = np.random.permutation(len(data_list))
        for i in range(batch_size):
            flag = random.random()
            query_folder = os.path.join(train_dataset_path, data_list[p[i]])
            irrelevant_folder = os.path.join(train_dataset_path, data_list[p[i+batch_size]])
            query_list = os.listdir(query_folder)
            irrelevant_list = os.listdir(irrelevant_folder)
            random.shuffle(query_list)

            query = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(query_folder, query_list[0]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255
            relevant = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(query_folder, query_list[1]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255
            irrelevant = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(irrelevant_folder, irrelevant_list[0]), 1), cv2.COLOR_RGB2BGR), input_shape[:2]) / 255

            pair[0][i], pair[1][i], pair[2][i] = flip_img(query, relevant, irrelevant, flag)
            pair.append(regions)
            target[i][0] = 0

            # print(i, os.path.join(query_folder, query_list[0]), os.path.join(query_folder, query_list[1]), os.path.join(irrelevant_folder, irrelevant_list[0]))

        # yield pair, target
        yield pair[0].shape, pair[1].shape, pair[2].shape, len(pair[3]), len(target)


if __name__ == '__main__':
    from get_regions import rmac_regions, get_size_vgg_feat_map
    Wmap, Hmap = get_size_vgg_feat_map(512, 512)
    regions = rmac_regions(Wmap, Hmap, 3)
    train_dataset_path = './dataset/train/train_data'
    datalist = os.listdir(train_dataset_path)
    gen = triple_generator(train_dataset_path, datalist, 16, (512, 512, 3), regions)
    for i in range(10):
        print(next(gen))