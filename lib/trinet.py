#!/usr/bin/env python3
from importlib import import_module
from itertools import count

import cv2
import numpy as np
import tensorflow as tf

from aggregators import AGGREGATORS
from lib.utils import compare


class TriNetReID:
    def __init__(self, model="resnet_v1_50", head="fc1024", checkpoints="checkpoints", aggregator="mean",
                 batch_size=16, crop_augment=None, flip=None, embed_dim=128):
        self.crop_augment = crop_augment
        self.flip = flip
        self.embed_dim = embed_dim
        self.model = model
        self.head = head
        self.aggregator = aggregator
        self.checkpoints = checkpoints
        self.batch_size = batch_size

        # Create the model and an embedding head.
        self.model = import_module('nets.' + self.model)
        self.head = import_module('heads.' + self.head)

        self.sess = tf.Session()
        # Initialize the network/load the checkpoint.
        self.checkpoint = tf.train.latest_checkpoint(self.checkpoints)
        print('Using checkpoint: {}'.format(self.checkpoint))

    @staticmethod
    def flip_augment(image, pid):
        """ Returns both the original and the horizontal flip of an image. """
        images = tf.stack([image, tf.reverse(image, [1])])
        return images, [pid] * 2

    @staticmethod
    def five_crops(image, crop_size):
        """ Returns the central and four corner crops of `crop_size` from `image`. """
        image_size = tf.shape(image)[:2]
        crop_margin = tf.subtract(image_size, crop_size)
        assert_size = tf.assert_non_negative(crop_margin,
                                             message='Crop size must be smaller or equal to the image size.')
        with tf.control_dependencies([assert_size]):
            top_left = tf.floor_div(crop_margin, 2)
            bottom_right = tf.add(top_left, crop_size)
        center = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        top_left = image[:-crop_margin[0], :-crop_margin[1]]
        top_right = image[:-crop_margin[0], crop_margin[1]:]
        bottom_left = image[crop_margin[0]:, :-crop_margin[1]]
        bottom_right = image[crop_margin[0]:, crop_margin[1]:]
        return center, top_left, top_right, bottom_left, bottom_right

    @staticmethod
    def resize(image, image_size):
        img_str = cv2.imencode('.jpg', image)[1].tostring()
        image_decoded = tf.image.decode_jpeg(img_str, channels=3)
        image_resized = tf.image.resize_images(image_decoded, image_size)
        return image_resized

    def embed(self, crops):
        # Load the data from the CSV file.
        net_input_size = (256, 128)
        pre_crop_size = (288, 144)

        crops = [self.resize(img, pre_crop_size if self.crop_augment else net_input_size) for img in crops]
        dataset = tf.data.Dataset.from_tensor_slices((crops, [x for x in range(len(crops))]))

        # Augment the data if specified by the arguments.
        # `modifiers` is a list of strings that keeps track of which augmentations
        # have been applied, so that a human can understand it later on.
        modifiers = ['original']
        if self.flip:
            dataset = dataset.map(lambda img, index: self.flip_augment(img, index))
            dataset = dataset.apply(tf.contrib.data.unbatch())
            modifiers = [o + m for m in ['', '_flip'] for o in modifiers]

        if self.crop_augment == 'center':
            dataset = dataset.map(lambda im, index: (self.five_crops(im, net_input_size)[0], index))
            modifiers = [o + '_center' for o in modifiers]
        elif self.crop_augment == 'five':
            dataset = dataset.map(lambda im, index: (tf.stack(self.five_crops(im, net_input_size)), [index] * 5))
            dataset = dataset.apply(tf.contrib.data.unbatch())
            modifiers = [o + m for o in modifiers for m in
                         ['_center', '_top_left', '_top_right', '_bottom_left', '_bottom_right']]
        elif self.crop_augment == 'avgpool':
            modifiers = [o + '_avgpool' for o in modifiers]
        else:
            modifiers = [o + '_resize' for o in modifiers]

        # Group it back into PK batches.
        dataset = dataset.batch(self.batch_size)

        # Overlap producing and consuming.
        dataset = dataset.prefetch(1)

        images, _ = dataset.make_one_shot_iterator().get_next()

        endpoints, body_prefix = self.model.endpoints(images, is_training=False)
        with tf.name_scope('head'):
            endpoints = self.head.head(endpoints, self.embed_dim, is_training=False)

        tf.train.Saver().restore(self.sess, self.checkpoint)

        # Go ahead and embed the whole dataset, with all augmented versions too.
        emb_storage = np.zeros((len(crops) * len(modifiers), self.embed_dim), np.float32)
        for start_idx in count(step=self.batch_size):
            try:
                emb = self.sess.run(endpoints['emb'])
                print('\rEmbedded batch {}-{}/{}'.format(start_idx, start_idx + len(emb), len(emb_storage)),
                      flush=True, end='')
                emb_storage[start_idx:start_idx + len(emb)] = emb
            except tf.errors.OutOfRangeError:
                break  # This just indicates the end of the dataset.

        print("Done with embedding, aggregating augmentations...", flush=True)

        if len(modifiers) > 1:
            # Pull out the augmentations into a separate first dimension.
            emb_storage = emb_storage.reshape(len(crops), len(modifiers), -1)
            emb_storage = emb_storage.transpose((1, 0, 2))  # (Aug,FID,128D)

            # Aggregate according to the specified parameter.
            emb_storage = AGGREGATORS[self.aggregator](emb_storage)
        # print(emb_storage)
        print(emb_storage.shape)

        return emb_storage


if __name__ == '__main__':
    import os

    images = []
    dir = "/home/imesha/Documents/Projects/FYP/Tests/re-id-dataset/output"
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        images.append(img)

    imgs = images[:100]

    re_id = TriNetReID(batch_size=32)
    print("Embedding {} images".format(len(imgs)))
    embeddings = re_id.embed(imgs)

    print("Comparing embeddings")
    for i in range(1):
        img = imgs[i]
        similar = []
        for j in range(len(imgs)):
            similarity = compare(embeddings[j], embeddings[i])
            if similarity < 10:
                similar.append(imgs[j])

        print("Found {} similar images".format(len(similar)))
        frame = cv2.resize(img, (25, 50))
        for s in similar:
            frame = np.hstack((frame, cv2.resize(s, (25, 50))))
        cv2.imshow("Similar", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    imgs = images[100:200]

    re_id = TriNetReID(batch_size=32)
    print("Embedding {} images".format(len(imgs)))
    embeddings = re_id.embed(imgs)

    print("Comparing embeddings")
    for i in range(1):
        img = imgs[i]
        similar = []
        for j in range(len(imgs)):
            similarity = compare(embeddings[j], embeddings[i])
            if similarity < 10:
                similar.append(imgs[j])

        print("Found {} similar images".format(len(similar)))
        frame = cv2.resize(img, (25, 50))
        for s in similar:
            frame = np.hstack((frame, cv2.resize(s, (25, 50))))
        cv2.imshow("Similar", frame)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
