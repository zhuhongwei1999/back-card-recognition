'''
Data Augmentation
'''
from PIL import Image
import numpy as np
import os
import xml.etree.ElementTree as ET
import util.node
import util.xml_handler
import math

def rand(a = 0, b = 1):
    return np.random.rand() * (b - a) + a


def label_rotate(xmin, ymin, xmax, ymax, angle, image_x, image_y):
    '''
    :param xmin: x-axis of top left
    :param ymin: y-axis of top left
    :param xmax: x-axis of bottom right
    :param ymax: y-axis of bottom right
    :param angle: rotation angle
    :param image_x: image width
    :param image_y: image length
    :return: (xmin, ymin, xmax, ymax) tuple after rotation
    '''
    angle = -angle * math.pi / 180.0
    center_x = image_x / 2.0
    center_y = image_y / 2.0
    x = [xmin, xmax, xmin, xmax]
    y = [ymin, ymin, ymax, ymax]

    # Rotation Formula
    # x = (x1 - x2)cosθ - (y1 - y2)sinθ + x2
    # y = (x1 - x2)sinθ + (y1 - y2)cosθ + y2

    # Counter-clockwise
    if angle < 0:
        nxmin = (x[0] - center_x) * math.cos(angle) - (y[0] - center_y) * math.sin(angle) + center_x
        nymin = (x[1] - center_x) * math.sin(angle) + (y[1] - center_y) * math.cos(angle) + center_y
        nymax = (x[2] - center_x) * math.sin(angle) + (y[2] - center_y) * math.cos(angle) + center_y
        nxmax = (x[3] - center_x) * math.cos(angle) - (y[3] - center_y) * math.sin(angle) + center_x
    # Clockwise
    else:
        nymin = (x[0] - center_x) * math.sin(angle) + (y[0] - center_y) * math.cos(angle) + center_y
        nxmax = (x[1] - center_x) * math.cos(angle) - (y[1] - center_y) * math.sin(angle) + center_x
        nxmin = (x[2] - center_x) * math.cos(angle) - (y[2] - center_y) * math.sin(angle) + center_x
        nymax = (x[3] - center_x) * math.sin(angle) + (y[3] - center_y) * math.cos(angle) + center_y

    return (str(int(nxmin)), str(int(nymin)), str(int(nxmax)), str(int(nymax)))


def augment(src_image_path, dest_image_path, src_xml_path, dest_xml_path):
    '''
    Data Augmentation
    :param src_image_path: Source image path
    :param dest_image_path: Destination image path
    :param src_xml_path: source xml path
    :param dest_xml_path: dest xml path
    :return: None
    '''
    for file in os.listdir(src_image_path):
        file_path = os.path.join(src_image_path, file)
        image = Image.open(file_path)
        tmp = image
        for i in range(20):
            image = tmp
            # Random scaling generation
            # scale = rand(.25, 2)
            # image = image.resize((int(scale * w), int(scale * h)), Image.BICUBIC)
            # Angle range [-10,-1] & [1,10]
            scale = i - 10
            if scale == 0:
                continue
            # Change the degree of rotation of the image
            image = image.rotate(scale)
            # Randomly adjust the light and dark of the picture
            degree = rand(0.8, 1.2)
            image = image.point(lambda p: p * degree)
            image.save(dest_image_path + '\\' + file[0:7] + '_' + str(i + 1) + '.jpg')
            # Operation on xml files
            my_src_xml_path = os.path.join(src_xml_path, file[0:7] + '.xml')
            my_dest_xml_path = os.path.join(dest_xml_path, file[0:7] + '_' + str(i + 1) + '.xml')

            node_list = []
            in_file = open(my_src_xml_path)
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                cls = obj.find('name').text
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymax').text))
                # Change labeling position based on rotation.
                nb = label_rotate(b[0], b[1], b[2], b[3], scale, image.size[0], image.size[1])
                one = util.node.Node(cls, '0', nb[0], nb[1], nb[2], nb[3])
                node_list.append(one)
            util.xml_handler.write_xml(node_list, my_dest_xml_path)


if __name__ == "__main__":
    src_image_path = 'src_image'
    dest_image_path = 'dest_image'
    src_xml_path = 'src_xml'
    dest_xml_path = 'dest_xml'
    augment(src_image_path, dest_image_path, src_xml_path, dest_xml_path)
