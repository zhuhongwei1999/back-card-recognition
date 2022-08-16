'''
训练部分
'''
import datetime

import keras.backend as K
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Lambda
from keras.models import Model
from card_recognize.yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from util.image_handler import get_data
from util.model_data_handler import get_classes, get_anchors
from card_recognize.kmeans import Kmeans
from card_recognize.data_make import data_split, add_path
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

os.environ["PATH"] += os.pathsep + 'F:/Graphviz/bin'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
batch_size = 1
iterations = 1


def main():
    classes_path = 'model_data/classes.txt'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'number']
    write_classes(classes_path, classes)

    data_split()
    add_path()

    cluster_number = 9
    annotation_path = '../dataset/label/train_label.txt'
    Kmeans(cluster_number, annotation_path)

    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (544, 544)
    model = create_model(input_shape, anchors, len(class_names))

    log_dir = ('weights/weights_%s' % nowTime + '.h5')

    train(model, annotation_path, input_shape, anchors, len(class_names), log_dir=log_dir)


def write_classes(classes_path, classes):
    with open(classes_path, 'w') as f:
        for one in classes:
            f.write(one + '\n')


def train(model, annotation_path, input_shape, anchors, num_classes, log_dir):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, validate on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    history = model.fit_generator(
        generator=data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                       num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=iterations)

    model.save_weights(log_dir)

    print('model has been trained!\n')


def create_model(input_shape, anchors, num_classes):
    K.clear_session() 
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], \
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])
    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        return None
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_data(annotation_lines[i], input_shape)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':
    main()
