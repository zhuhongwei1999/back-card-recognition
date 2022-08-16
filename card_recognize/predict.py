'''
预测部分
'''
import keras
from PIL import Image
from keras import backend as K, Input
import os
from util.model_data_handler import get_anchors, get_classes
from timeit import default_timer as timer
import numpy as np
from card_recognize.yolo3.model import yolo_eval, yolo_body
from util.xml_handler import read_one_xml
from util.image_handler import letterbox_image
from util.model_data_handler import compute_iou
from util.image_handler import draw_box

score_threshold = 0.6
iou = 0.45
image_size = (544, 544)
anchors_path = 'C:\\Users\cyw35\Desktop\Bank_Card_OCR\card_recognize\model_data\yolo_anchors.txt'
classes_path = 'C:\\Users\cyw35\Desktop\Bank_Card_OCR\card_recognize\model_data\classes.txt'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)


def get_predict_result(image_set_path, result_set_path, model_path):
    all_result_list = []
    sess = K.get_session()
    num_anchors = len(anchors)
    model_path = os.path.expanduser(model_path)

    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    yolo_model.load_weights(model_path)
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, num_classes, image_size, score_threshold, iou)
    for file in os.listdir(image_set_path):
        file_path = os.path.join(image_set_path, file)
        image = Image.open(file_path)
        boxed_image, scale, padding = letterbox_image(image, tuple(reversed(image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes], feed_dict={
                yolo_model.input: image_data,
                K.learning_phase(): 0
            })
        # 取出预测结果
        pre_rec, pre_trust, pre_class = [], [], []
        for i, _ in sorted(list(enumerate(out_boxes[:, 1])), key=lambda e: e[1]):
            c = out_classes[i]
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            pre_rec.append((top, left, bottom, right))
            pre_trust.append(score)
            pre_class.append(predicted_class)
        one_result_list, one_result_rec, one_result_trust = nms_inter_class(pre_rec, pre_trust, pre_class)
        if 'number' in one_result_list:
            index = one_result_list.index('number')
            del one_result_list[index]
            del one_result_rec[index]
            del one_result_trust[index]

        tempArr = []
        for i in range(len(one_result_list)):
            tempArr.append(one_result_list[i])

        one_result_list_blank = get_blank_space(one_result_list, one_result_rec)

        one_result_str = file_path.split('\\')[-1][0:-4] + ':'

        for i in range(len(one_result_list_blank)):
            one_result_str += one_result_list_blank[i]

        all_result_list.append(one_result_str)

    return all_result_list


def nms_inter_class(pre_rec, pre_trust, pre_class):
    result_rec = []
    result_trust = []
    result_class = []

    number_score = 0
    number_index = -1
    for i in range(len(pre_class)):
        if pre_class[i] == 'number' and pre_trust[i] > number_score:
            number_index = i
            number_score = pre_trust[i]

    for i in range(len(pre_class)):
        if pre_class[i] == 'number':
            if i == number_index:
                result_rec.append(pre_rec[i])
                result_trust.append(pre_trust[i])
                result_class.append(pre_class[i])
            else:
                continue
        else:
            if len(result_rec) == 0:
                result_rec.append(pre_rec[i])
                result_trust.append(pre_trust[i])
                result_class.append(pre_class[i])
            else:
                flag = True
                for j in range(len(result_rec)):
                    if compute_iou(result_rec[j], pre_rec[i]) > iou:
                        flag = False
                        if pre_trust[i] > result_trust[j]:
                            result_rec[j] = pre_rec[i]
                            result_trust[j] = pre_trust[i]
                            result_class[j] = pre_class[i]
                        break
                if flag == True:
                    result_rec.append(pre_rec[i])
                    result_trust.append(pre_trust[i])
                    result_class.append(pre_class[i])
    return result_class, result_rec, result_trust


model_path = 'C:\\Users\cyw35\Desktop\Bank_Card_OCR\card_recognize\weights\weights_2019_07_08_15_06_27.h5'
sess = K.get_session()
num_anchors = len(anchors)
model_path = os.path.expanduser(model_path)

yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
yolo_model.load_weights(model_path)

boxes, scores, classes = yolo_eval(yolo_model.output, anchors, num_classes, image_size, score_threshold, iou)


def get_one_result(image, model_path):
    boxed_image, scale, padding = letterbox_image(image, tuple(reversed(image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    start = timer()
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes], feed_dict={
            yolo_model.input: image_data,
            K.learning_phase(): 0
        })
    end = timer()
    print(str(end - start))
    pre_rec = []
    pre_trust = []
    pre_class = []
    for i, _ in sorted(list(enumerate(out_boxes[:, 1])), key=lambda e: e[1]):
        c = out_classes[i]
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        pre_rec.append((top, left, bottom, right))
        pre_trust.append(score)
        pre_class.append(predicted_class)

    # print(pre_class)
    # print(pre_trust)
    # 做类间极大值抑制
    one_result_list, one_result_rec, one_result_trust = nms_inter_class(pre_rec, pre_trust, pre_class)

    # 删除number
    if 'number' in one_result_list:
        index = one_result_list.index('number')
        del one_result_list[index]
        del one_result_rec[index]
        del one_result_trust[index]

    # 返回没有空格的predNum
    tempArr = []
    for i in range(len(one_result_list)):
        tempArr.append(one_result_list[i])

    one_result_list_blank = get_blank_space(one_result_list, one_result_rec)

    return tempArr, one_result_list_blank


def get_blank_space(one_result_list, one_result_rec):
    dif_arr = []
    for i in range(len(one_result_rec)):
        if i > 0:
            dif_arr.append(one_result_rec[i][1] - one_result_rec[i - 1][1])
    dif_arr = (dif_arr - np.mean(dif_arr)) / np.std(dif_arr)
    print(dif_arr)

    if max(dif_arr) > 3:
        deta = 1.5
    else:
        deta = 1.0

    incrementValue = 1
    for i in range(len(dif_arr)):
        if dif_arr[i] > deta and i > 2:
            blankIndex = i + incrementValue
            one_result_list.insert(blankIndex, '_')
            incrementValue += 1
    return one_result_list


def get_test_label_accuracy(test_label_path, model_path):
    non_digital_label = ['number', 'up', 'left', 'right', 'up', 'down', 'bottom']
    pre_digital_num = 0
    real_digital_num = 0
    pre_whole_num = 0
    real_whole_num = 0
    sess = K.get_session()
    num_anchors = len(anchors)
    model_path = os.path.expanduser(model_path)

    yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
    yolo_model.load_weights(model_path)
    boxes, scores, classes = yolo_eval(yolo_model.output, anchors, num_classes, image_size, score_threshold, iou)

    with open(test_label_path, 'r') as f:
        line = f.readline()
        while line:

            image_path = line.split(' ')[0]
            print(image_path)
            image = Image.open(image_path)
            boxed_image, scale, padding = letterbox_image(image, tuple(reversed(image_size)))
            image_data = np.array(boxed_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes], feed_dict={
                    yolo_model.input: image_data,
                    K.learning_phase(): 0
                })
            pre_rec = []
            pre_trust = []
            pre_class = []
            for i, _ in sorted(list(enumerate(out_boxes[:, 1])), key=lambda e: e[1]):
                c = out_classes[i]
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                pre_rec.append((top, left, bottom, right))
                pre_trust.append(score)
                pre_class.append(predicted_class)
            one_result_list, one_result_rec, one_result_trust = nms_inter_class(pre_rec, pre_trust, pre_class)
            # ['1','2','3']
            name = image_path.split('/')[-1][:-4]
            pre_digital_str = ''
            for one in one_result_list:
                if one not in non_digital_label:
                    pre_digital_str += one
            xml_path = '../dataset/annotation/' + name + '.xml'
            real_digital_str = []
            node_list = read_one_xml(xml_path)
            for node in node_list:
                if node.name not in non_digital_label:
                    real_digital_str.append(node.name)
            flag = True
            length = min(len(pre_digital_str), len(real_digital_str))
            for i in range(length):
                if real_digital_str[i] == pre_digital_str[i]:
                    pre_digital_num += 1
                else:
                    flag = False
            if flag == True:
                pre_whole_num += 1

            line = f.readline()
            real_whole_num += 1
            real_digital_num += len(real_digital_str)

    print('按数字算准确率为：{}'.format(float(pre_digital_num) / float(real_digital_num)))
    print('按整张图片算准确率为：{}'.format(float(pre_whole_num) / float(real_whole_num)))


if __name__ == '__main__':
    get_test_label_accuracy('../dataset/label/test_label.txt',
                            '../card_recognize/weights/weights_2019_06_05_18_16_40.h5')
    # image = Image.open('./../demo/test_images/nkp(51).png')
    # image = Image.open('./../dataset/data_19_6_27/img/ccc0001.jpg')
    # start = timer()
    # print(get_one_result(image, '../card_recognize/weights/weights_2019_06_05_18_16_40.h5'))
    # end = timer()
    # print(str(end - start) + 's')
    # 10.3s
