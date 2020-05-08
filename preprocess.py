import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import gc

##Load images, labels, masks
def load_data():
    '''
    the output sequence is: train_x, test_x, train_y, test_y
    '''
    labels = np.load('brain_tumor_dataset/labels.npy')
    images = np.clip( (np.load('brain_tumor_dataset/images.npy')/12728),0,1)
    masks = np.load('brain_tumor_dataset/masks.npy')*1
    print("labels shape: ",labels.shape)
    print("images shape: ",images.shape)
    print("masks shape: ",masks.shape)

    img_size_ori = 512
    img_size_target = 128

    images = np.expand_dims(images,axis=-1)
    masks = np.expand_dims(masks,axis=-1)

    def downsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True,)
        
    def upsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    
    images = np.array([ downsample(image) for image in images ])
    masks = (np.array([ downsample(mask) for mask in masks ])>0)*1

    print("images shape after reshape: ", images.shape)
    print("masks shape: ", masks.shape)  

    X,X_v,Y,Y_v = train_test_split( images,masks,test_size=0.2,stratify=labels)
    del images
    del masks
    del labels
    gc.collect()
    print("train x shape: ", X.shape)
    print("test x shape: ", X_v.shape) 

    X = np.append( X, [ np.fliplr(x) for x in X], axis=0 )
    Y = np.append( Y, [ np.fliplr(y) for y in Y], axis=0 )
    print("final train x shape: ", X.shape)
    print("final train y shape: ", Y.shape) 

    return X,X_v,Y,Y_v