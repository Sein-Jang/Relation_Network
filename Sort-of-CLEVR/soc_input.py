"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import functools


class ImageInput(object):
    """Generates Image input_fn for training or evaluation."""

    def __init__(self, is_training, data_dir, use_bfloat16, transpose_input=True):
        self.is_training = is_training
        self.data_dir = data_dir
        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input

    def set_shapes(self, batch_size, images, questions, answers):
        """Statically set the batch_size dimension."""

        features = {}

        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, None, None, None])))
        answers.set_shape(answers.get_shape().merge_with(
            tf.TensorShape([batch_size, 10])))
        questions.set_shape(questions.get_shape().merge_with(
            tf.TensorShape([batch_size, 11])))

        features['image'] = images
        features['question'] = questions

        return features, answers

    def dataset_parser(self, value):
        """Parse an Imagedata record from a serialized string Tensor."""
        keys_to_features = {
            'question':tf.VarLenFeature(tf.float32),
            'answer':tf.VarLenFeature(tf.float32),
            'image':tf.VarLenFeature(tf.float32)
            }

        parsed = tf.parse_single_example(value, keys_to_features)
        question = tf.sparse_tensor_to_dense(parsed['question'])
        answer = tf.sparse_tensor_to_dense(parsed['answer'])
        image = tf.reshape(tf.sparse_tensor_to_dense(parsed['image']), [128, 128, 3])

        return image, question, answer


    features = {}

    def input_fn(self,params):
        """Input function which provides a single batch for train or eval."""
        if self.data_dir == None:
            raise ValueError('No Data Dir')
            return self.input_fn_null(params)

        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # tf.contfib.tpu.RunConfig for details.
        batch_size = params['batch_size']

        # Shuffle the filenames to ensure better randomization.
        file_pattern = os.path.join(
            self.data_dir, 'train-*' if self.is_training else 'validation-*')

        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

        if self.is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(filename)
            return dataset

        # Read the data from disk in parallel
        dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=32, sloppy=True))

        # Parse, preprocess, and batch the data in parallel
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                self.dataset_parser, batch_size=batch_size,
                num_parallel_batches=8,
                drop_remainder=True))



        # Assign static batch size dimension
        dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        # Perfetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

        return dataset
