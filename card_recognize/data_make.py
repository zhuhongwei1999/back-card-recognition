'''
Making dataset

First, divide all the data into train/test/val set, then put all the xml of each set with all the box
of each set in a txt file with vertex coordinates and types
'''

import os
import random
import xml.etree.ElementTree as ET

sets = ['train', 'test']

root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_classes():
    class_file = open("model_data/classes.txt", "r")
    classes = class_file.read().splitlines()
    class_file.close()
    return classes

def data_split():
    train_val_percent = 0.2
    test_percent = 0.8
    xml_filepath = '../dataset/annotation'
    total_xml = os.listdir(xml_filepath)

    num = len(total_xml)
    list = range(num)

    train_num = int(num * test_percent)

    train = random.sample(list, train_num)

    test_file = open('../dataset/name/test.txt', 'w')
    train_file = open('../dataset/name/train.txt', 'w')

    for file in list:
        name = total_xml[file][:-4] + '\n'
        if file not in train:
            test_file.write(name)
        else:
            train_file.write(name)

    train_file.close()
    test_file.close()


def convert_annotation(image_id, list_file):
    '''
    Take coordinates of the vertices (top left/bottom right) and the type of each box of the xml file and save them.
    '''
    in_file = open('../dataset/annotation/%s.xml' % image_id)
    tree = ET.parse(in_file)
    root = tree.getroot()
    classes = get_classes()

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        cor_list = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text),
                    int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
        list_file.write(" " + ",".join([str(cor) for cor in cor_list]) + ',' + str(cls_id))


def add_path():
    for image_set in sets:
        image_ids = open('../dataset/name/%s.txt' % image_set).read().strip().split()
        list_file = open('../dataset/label/%s_label.txt' % image_set, 'w')
        for image_id in image_ids:
            list_file.write('%s/dataset/images/%s.jpg' % (root_path, image_id))
            convert_annotation(image_id, list_file)
            list_file.write('\n')
        list_file.close()


if __name__ == '__main__':
    data_split()
    add_path()
