"""
Created on Sun Jan 09 2022

utils

@author: Ray
"""
import os, glob, sys
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing import image


def load_path(folder):
    paths = []
    if len(glob.glob(folder + r'\**.png')) != 0:
        paths = glob.glob(folder + r'\**.png')

        if len(glob.glob(folder + r'\**.jpg')) != 0:
            paths.extend(glob.glob(folder + r'\**.jpg'))

    elif len(glob.glob(folder + r'\**.png')) == 0:
        if len(glob.glob(folder + r'\**.jpg')) != 0:
            paths = glob.glob(folder + r'\**.jpg')
    else: 
        pass

    return paths


def load_data(images, input_size):
    g_imgs = []
    d_imgs = []
    for i in range(len(images)):
        g_train, d_train = images[i].split(', ')
        g_img = np.array(image.load_img(g_train, target_size=(input_size, input_size)))
        d_img = np.array(image.load_img(d_train, target_size=(input_size, input_size)))
        g_img = (g_img / 127.5) - 1
        d_img = (d_img / 127.5) - 1
        g_imgs.append(g_img)
        d_imgs.append(d_img)

    return g_imgs, d_imgs, 


def load_test_data(A_path, input_size):

    g_img = np.array(image.load_img(A_path, target_size=(input_size, input_size)))
    g_img = (g_img / 127.5) - 1
    g_img = np.expand_dims(g_img, axis=0)

    return list(g_img)


def save_data(im, images, save_path=None):
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
        for i in range(len(images)):
            im[i].save(save_path + '\\' + os.path.basename(images[i]))
                
    else:
        for i in range(len(images)):
            im[i].save(save_path + '\\' + os.path.basename(images[i]))

    return save_path

