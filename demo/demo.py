'''
Prediction
'''
from card_recognize.predict import get_predict_result


def recognize_image_set(image_set_path, result_set_path, result_path):
    '''
    :param image_set_path:
    :param result_set_path:
    :param result_path:
    :return: None
    '''
    model_path = '../card_recognize/weights/weights_2019_07_08_15_06_27.h5'
    all_result_list = get_predict_result(image_set_path, result_set_path, model_path)
    with open(result_path, 'w') as f:
        for one in all_result_list:
            f.write(one + '\n')


if __name__ == '__main__':
    recognize_image_set('test_images', 'test_result', 'result.txt')
