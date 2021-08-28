from covid_classifier import *

import os
import cv2
import numpy as np


def covid_samples(covid_image_bytes_list, deep_model):
    """
    To determine the number of True positives and false positives procured after predicting on a list of CT scan images belonging to the class "Covid"

    Args:
    covid_image_bytes_list: A list of images in the form of thier corresponding bytes. Note: we actually receive these bytes from our fast api,
    deep_model: tensor object our classifier model

    Returns:
    tp: true positives count, fp: False positives count, false_positives_list: A list of names of files classified incorrectly as "Non_Covid"
    """
    tp = fp = 0
    false_positives_list = []
    for image_info in covid_image_bytes_list:

        img = cv2.imdecode(np.frombuffer(image_info[1], dtype="uint8"), 1)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        final_tensor = convert_image_arr_to_tensor(rgb_img)

        img_class = classify_image(deep_model, final_tensor)

        if img_class == "Covid":
            tp += 1
        else:
            fp += 1
            false_positives_list.append(image_info[0])

    return tp, fp, false_positives_list


def non_covid_samples(non_covid_image_bytes_list, deep_model):
    """
    To determine the number of True negatives and false negatives procured after predicting on a list of CT scan images belonging to the class "Non_Covid"

    Args:
    non_covid_image_bytes_list: A list of images in the form of thier corresponding bytes. Note: we actually receive these bytes from our fast api,
    deep_model: tensor object our classifier model

    Returns:
    tn: true negatives count, fn: False negatives count, false_negatives_list: A list of names of files classified incorrectly as "Covid"
    """
    tn = fn = 0
    false_negatives_list = []
    for image_info in non_covid_image_bytes_list:

        img = cv2.imdecode(np.frombuffer(image_info[1], dtype="uint8"), 1)

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        final_tensor = convert_image_arr_to_tensor(rgb_img)

        img_class = classify_image(deep_model, final_tensor)

        if img_class == "Non_Covid":
            tn += 1
        else:
            fn += 1
            false_negatives_list.append(image_info[0])

    return tn, fn, false_negatives_list
