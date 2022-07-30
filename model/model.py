# DECAF training and predicting model with parallelization
# Created by Renhao Liu and Yu Sun, CIG, WUSTL, 2021

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.client import device_lib
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import skimage
from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging

# Model parameters
flags.DEFINE_string("tf_summary_dir", "log", "directory for tf summary log")
flags.DEFINE_enum(
    "positional_encoding_type", "exp_diag", ["exp_diag", "exp", "fourier_fixed_xy"], "positional_encoding_type",)
flags.DEFINE_float("dia_digree", 45, "degrees per each encoding in exp_diag")
flags.DEFINE_enum(
    "mlp_activation", "leaky_relu", ["leaky_relu"], "Activation functions for mlp",)
flags.DEFINE_integer("mlp_layer_num", 10, "number of layers in mlp network")
flags.DEFINE_integer("mlp_kernel_size", 208, "width of mlp")
flags.DEFINE_integer('fourier_encoding_size', 256, "number of rows in fourier matrix")
flags.DEFINE_float("sig_xy", 26.0, "Fourier encoding sig_xy")
flags.DEFINE_float("sig_z", 1.0, "Fourier encoding sig_z")
flags.DEFINE_integer(
    "xy_encoding_num", 6, "number of frequecncies expanded in the spatial dimensions"
)
flags.DEFINE_integer(
    "z_encoding_num", 5, "number of frequecncies expanded in the depth dimension"
)
flags.DEFINE_multi_integer("mlp_skip_layer", [5], "skip layers in the mlp network")
flags.DEFINE_float("output_scale", 5, "neural network out put scale")

# Regularization parameters
flags.DEFINE_enum("regularize_type", "dncnn2d", ["dncnn2d"], "type of the network",)
flags.DEFINE_float("regularize_weight", 0.0, "Weight for regularizer")
flags.DEFINE_float(
    "tv3d_z_reg_weight", 3.079699, "Reg weight scaling for z axis in 3dtv"
)
flags.DEFINE_string(
    "DnCNN_model_path",
    "/export/project/sun.yu/projects/DnCNN/cnn_trained/DnCNN_sigma=5.0/models/final/model",
    "model path of pre-trained DnCNN",
)
flags.DEFINE_float("DnCNN_normalization_min", -0.05, "DnCNN normalization min")
flags.DEFINE_float("DnCNN_normalization_max", 0.05, "DnCNN normalization max")

# Training parameter
flags.DEFINE_integer("start_epoch", 0, "start epoch, useful for continue training")
flags.DEFINE_integer("iters_per_epoch", 1, "num of iters for each resampling")
flags.DEFINE_integer("image_save_epoch", 5000, "number of iteration to save one image")
flags.DEFINE_integer(
    "intermediate_result_save_epoch",
    100,
    "number of iterations to save intermediate result",
)
flags.DEFINE_integer("log_iter", 25, "number of iteration to log to console")
flags.DEFINE_integer("model_save_epoch", 5000, "epoch per intermediate model")
flags.DEFINE_integer(
    "num_measurements_per_batch",
    -1,
    "number of measurements per batch. negative value for all measurements",
)

# Prediction parameters
flags.DEFINE_integer("prediction_batch_size", 1, "Batch size for prediction")

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"

# get total number of visible gpus
local_device_protos = device_lib.list_local_devices()
NUM_GPUS = np.size([x.name for x in local_device_protos if x.device_type == "GPU"])


########################################
###       Tensorboard & Helper       ###
########################################


def record_summary(writer, name, value, step):
    summary = tf.compat.v1.Summary()
    summary.value.add(tag=name, simple_value=value)
    writer.add_summary(summary, step)
    writer.flush()


def reshape_image(image):
    if tf.size(input=tf.shape(input=image)) == 2:
        image_reshaped = tf.expand_dims(image, axis=0)
        image_reshaped = tf.expand_dims(image_reshaped, axis=-1)
    elif tf.size(input=tf.shape(input=image)) == 3:
        image_reshaped = tf.expand_dims(image, axis=-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = tf.expand_dims(image, axis=0)
    image_reshaped = tf.expand_dims(image_reshaped, axis=-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = tf.expand_dims(image, axis=-1)
    return image_reshaped


def reshape_image_5(image):
    shape = tf.shape(input=image)
    image_reshaped = tf.reshape(image, [-1, shape[2], shape[3], 1])
    return image_reshaped


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ["Variable", "VariableV2", "AutoReloadVariable"]


def assign_to_device(device, ps_device="/cpu:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.compat.v1.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


class Model:
    def __init__(self, net_kargs=None, name="model_summary"):
        # Setup parameters
        self.name = name
        self.tf_summary_dir = "{}/{}".format(FLAGS.tf_summary_dir, name)
        tf.keras.backend.clear_session()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
        if net_kargs is None:
            self.net_kargs = {
                "skip_layers": FLAGS.mlp_skip_layer,
                "mlp_layer_num": FLAGS.mlp_layer_num,
                "kernel_size": FLAGS.mlp_kernel_size,
                "L_xy": FLAGS.xy_encoding_num,
                "L_z": FLAGS.z_encoding_num,
            }
        else:
            self.net_kargs = net_kargs

    ###########################
    ###     Neural Nets     ###
    ###########################

    def inference(self, coordinates, Hreal, Himag, padding, mask, reuse=False):
        # MLP network
        with tf.compat.v1.variable_scope("infer_y", reuse=reuse):
            xhat = self.__neural_repres(
                coordinates, tf.shape(input=Hreal), **self.net_kargs
            )
            mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(mask, -1), 0), 0)
            Hxhat = self.__forward_op(xhat * mask, Hreal, Himag, padding)
        return Hxhat, xhat

    def __neural_repres(
        self,
        in_node,
        x_shape,
        skip_layers=[],
        mlp_layer_num=10,
        kernel_size=256,
        L_xy=6,
        L_z=5,
    ):
        # positional encoding
        with tf.compat.v1.variable_scope(NET_SCOPE):
            if FLAGS.positional_encoding_type == "exp_diag":
                s = np.sin(np.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                    :, np.newaxis
                ]
                c = np.cos(np.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[
                    :, np.newaxis
                ]
                fourier_mapping = np.concatenate((s, c), axis=1).T

                xy_freq = tf.matmul(in_node[:, :2], fourier_mapping)

                for l in range(L_xy):
                    cur_freq = tf.concat(
                        [
                            tf.sin(2 ** l * np.pi * xy_freq),
                            tf.cos(2 ** l * np.pi * xy_freq),
                        ],
                        axis=-1,
                    )
                    if l == 0:
                        tot_freq = cur_freq
                    else:
                        tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)

                for l in range(L_z):
                    cur_freq = tf.concat(
                        [
                            tf.sin(2 ** l * np.pi * in_node[:, 2][:, None]),
                            tf.cos(2 ** l * np.pi * in_node[:, 2][:, None]),
                        ],
                        axis=-1,
                    )
                    tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)
            elif FLAGS.positional_encoding_type == 'exp':
                for l in range(L_xy):  # fourier feature map
                    indicator = np.array([1., 1., 1. if l < L_z else 0.])
                    cur_freq = tf.concat([tf.sin(indicator * 2 ** l * np.pi * in_node),
                                          tf.cos(indicator * 2 ** l * np.pi * in_node)], axis=-1)
                    if l is 0:
                        tot_freq = cur_freq
                    else:
                        tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)
            elif FLAGS.positional_encoding_type == 'fourier_fixed_xy':
                np.random.seed(10)
                fourier_mapping = np.random.normal(0, FLAGS.sig_xy, (FLAGS.fourier_encoding_size, 2)).astype('float32')
                
                xy_freq = tf.matmul(in_node[:, :2], fourier_mapping.T)
                xy_freq = tf.concat([tf.sin(2 * np.pi * xy_freq),
                          tf.cos(2 * np.pi * xy_freq)], axis=-1)
                
                tot_freq = xy_freq        
                for l in range(L_z):
                    cur_freq = tf.concat([tf.sin(2 ** l * np.pi * in_node[:, 2][:, None]),
                                          tf.cos(2 ** l * np.pi * in_node[:, 2][:, None])], axis=-1)   
                    tot_freq = tf.concat([tot_freq, cur_freq], axis=-1)
            else:
                raise NotImplementedError(FLAGS.positional_encoding_type)
            # input to MLP
            in_node = tot_freq
            # input encoder
            if FLAGS.task_type == "aidt":
                kernel_initializer = None
            elif FLAGS.task_type == "idt":
                kernel_initializer = tf.compat.v1.random_uniform_initializer(
                    -0.05, 0.05
                )
            elif FLAGS.task_type == "midt":
                kernel_initializer = None
            else:
                raise NotImplementedError

            for layer in range(mlp_layer_num):
                if layer in skip_layers:
                    in_node = tf.concat([in_node, tot_freq], -1)

                if FLAGS.mlp_activation == "relu":
                    activation = tf.nn.relu
                elif FLAGS.mlp_activation == "leaky_relu":
                    activation = None  # tf.nn.leaky_relu
                elif FLAGS.mlp_activation == "elu":
                    activation = tf.nn.elu
                elif FLAGS.mlp_activation == "tanh":
                    activation = tf.nn.tanh
                in_node = tf.compat.v1.layers.dense(
                    in_node,
                    kernel_size,
                    kernel_initializer=kernel_initializer,
                    activation=activation,
                )

                if FLAGS.mlp_activation == "leaky_relu":
                    in_node = tf.maximum(0.2 * in_node, in_node)

            # final layer
            output = tf.compat.v1.layers.dense(
                in_node, 2, kernel_initializer=kernel_initializer, activation=None
            )
            output = output / FLAGS.output_scale
        # reshape output to x
        xhat = tf.reshape(
            output, (x_shape[1], x_shape[2], FLAGS.view_size, FLAGS.view_size, 2)
        )  # [1, Z, X, Y, Real/Imagenary]
        return xhat

    def __forward_op(
        self,
        x,
        Hreal,
        Himag,
        padding,
    ):
        padded_field = tf.pad(tensor=x, paddings=padding)
        padded_phase = padded_field[:, :, :, :, 0]
        padded_absorption = padded_field[:, :, :, :, 1]
        transferred_field = tf.signal.ifft2d(
            tf.multiply(Hreal, tf.signal.fft2d(tf.cast(padded_phase, tf.complex64)))
            + tf.multiply(
                Himag, tf.signal.fft2d(tf.cast(padded_absorption, tf.complex64))
            )
        )
        Hxhat = tf.reduce_sum(input_tensor=tf.math.real(transferred_field), axis=(1, 2))
        return Hxhat

    def save(self, sess, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        save_path = self.saver.save(sess, path)
        print("saved to {}".format(save_path))
        return save_path

    def restore(self, sess, model_path):
        self.saver.restore(sess, model_path)

    ##############################
    ###     Loss Functions     ###
    ##############################

    def __tower_loss(self, tower_idx, Hreal, Himag, reuse=False):
        # get input coordinates & measurements & padding
        x = self.Xs[tower_idx, ...]
        y = self.Ys[tower_idx, ...]
        padding = self.Ps[tower_idx, ...]
        mask = self.Ms[tower_idx, ...]

        # inference
        Hxhat, xhat = self.inference(x, Hreal, Himag, padding, mask, reuse=reuse)
        # data fidelity
        if FLAGS.loss == "l1":
            mse = tf.reduce_mean(input_tensor=tf.abs(Hxhat - y))
        elif FLAGS.loss == "l2":
            mse = tf.reduce_mean(input_tensor=tf.square(Hxhat - y)) / 2
        else:
            raise NotImplementedError
        # regularizer
        if FLAGS.regularize_type == "dncnn2d":
            xhat_trans = tf.transpose(
                a=tf.squeeze(xhat), perm=[3, 0, 1, 2]
            )  # [1, Z, X, Y, Real/Imagenary]
            xhat_concat = tf.concat([xhat_trans[0, ...], xhat_trans[1, ...]], 0)
            xhat_expand = tf.expand_dims(xhat_concat, 3)
            phase_regularize_value = self.__dncnn_2d(xhat_expand, reuse=reuse)
            absorption_regularize_value = tf.constant(0.0)
        else:
            raise NotImplementedError

        if FLAGS.tv3d_z_reg_weight != 0:
            tv_z = self.__total_variation_z(xhat[..., 0])
            tv_z += self.__total_variation_z(xhat[..., 1])
        else:
            tv_z = tf.constant(0.0)
            
        # final loss
        loss = (
            mse
            + FLAGS.regularize_weight
            * (absorption_regularize_value + phase_regularize_value)
            + FLAGS.tv3d_z_reg_weight * tv_z
        )

        return (
            loss,
            mse,
            phase_regularize_value,
            absorption_regularize_value,
            xhat,
            Hxhat,
            y,
        )

    def __total_variation_2d(self, images):
        pixel_dif2 = tf.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
        pixel_dif3 = tf.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
        total_var = tf.reduce_sum(input_tensor=pixel_dif2) + tf.reduce_sum(
            input_tensor=pixel_dif3
        )
        return total_var

    def __total_variation_z(self, images):
        """
        Normalized total variation 3d
        :param images: Images should have 4 dims: batch_size, z, x, y
        :return:
        """
        pixel_dif1 = tf.abs(images[:, 1:, :, :] - images[:, :-1, :, :])
        total_var = tf.reduce_sum(input_tensor=pixel_dif1)
        return total_var

    def __dncnn_inference(
        self,
        input,
        reuse,
        output_channel=1,
        layer_num=10,
        filter_size=3,
        feature_root=64,
    ):
        # input layer
        with tf.compat.v1.variable_scope("DnCNN", reuse=reuse):
            with tf.compat.v1.variable_scope("layer_1"):
                in_node = tf.compat.v1.layers.conv2d(
                    input,
                    feature_root,
                    filter_size,
                    padding="same",
                    activation=tf.nn.relu,
                    trainable=False,
                )
            # composite convolutional layers
            for layer in range(2, layer_num):
                with tf.compat.v1.variable_scope("layer_{}".format(layer)):
                    in_node = tf.compat.v1.layers.conv2d(
                        in_node,
                        feature_root,
                        filter_size,
                        padding="same",
                        name="conv2d_{}".format(layer),
                        use_bias=False,
                        trainable=False,
                    )
                    in_node = tf.nn.relu(
                        tf.compat.v1.layers.batch_normalization(
                            in_node, trainable=False
                        )
                    )
            # output layer and residual learning
            with tf.compat.v1.variable_scope("layer_{}".format(layer_num)):
                in_node = tf.compat.v1.layers.conv2d(
                    in_node,
                    output_channel,
                    filter_size,
                    padding="same",
                    trainable=False,
                )
                output = input - in_node
        return output

    def __dncnn_2d(self, images, reuse=True):  # [N, H, W, C]
        """
        DnCNN as 2.5 dimensional denoiser based on l-2 norm
        """
        a_min = FLAGS.DnCNN_normalization_min
        a_max = FLAGS.DnCNN_normalization_max
        normalized = (images - a_min) / (a_max - a_min)
        denoised = self.__dncnn_inference(tf.clip_by_value(normalized, 0, 1), reuse)
        denormalized = denoised * (a_max - a_min) + a_min
        dncnn_res = tf.reduce_sum(tf.square(denormalized))
        return dncnn_res

    #########################################
    ###    Parallel & Serial Training     ###
    #########################################

    def train(self, output_path, train_provider, learning_rate=1e-5, epochs=80):
        # set default graph
        with tf.Graph().as_default(), tf.device("/cpu:0"):
            # set up training accross all gpus
            (
                H_op,
                train_op,
                statistics_op,
                summary_op,
                xhat_op,
                Hxhat_op,
            ) = self.__parallelization()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                )
            )
            # initialize trainable variables
            # z = np.zeros((1,1,1,1,1)) + 1j * np.zeros((1,1,1,1,1))

            if FLAGS.num_measurements_per_batch <= 0:
                sess.run(
                    tf.compat.v1.global_variables_initializer(),
                    feed_dict={
                        self.Himag_update_placeholder: train_provider.Himag,
                        self.Hreal_update_placeholder: train_provider.Hreal,
                    },
                )
            else:
                Hreal, Himag = train_provider.sample_partial_tf(
                    FLAGS.num_measurements_per_batch
                )
                sess.run(
                    tf.compat.v1.global_variables_initializer(),
                    feed_dict={
                        self.Himag_update_placeholder: Hreal,
                        self.Hreal_update_placeholder: Himag,
                    },
                )

            print("********************")
            # Global Steps
            iters_per_epoch = FLAGS.iters_per_epoch
            global_step = FLAGS.start_epoch * iters_per_epoch
            # start the queue runners
            tf.compat.v1.train.start_queue_runners(sess=sess)

            # initialize summary writer
            summary_writer = tf.compat.v1.summary.FileWriter(
                self.tf_summary_dir, graph=sess.graph
            )

            # load previous model if start epoch > 0
            if FLAGS.start_epoch > 0:
                train_provider.restore("{}/latest/".format(output_path))
                self.restore(sess, os.path.join(output_path, "latest/model"))

            # load pre-trained dncnn
            if "dncnn" in FLAGS.regularize_type:
                saver = tf.compat.v1.train.Saver(
                    var_list=[
                        v for v in tf.compat.v1.global_variables(scope=DNCNN_SCOPE)
                    ]
                )
                saver.restore(sess, FLAGS.DnCNN_model_path)
                tf.compat.v1.logging.info(
                    "DnCNN restored from file: %s" % FLAGS.DnCNN_model_path
                )

            # main loop
            tf.compat.v1.logging.info("Training Started")
            total_time = 0
            for epoch in range(FLAGS.start_epoch, epochs):
                current_time = time.perf_counter()

                # extract learning rate
                if type(learning_rate) is np.ndarray or type(learning_rate) is list:
                    lr = learning_rate[epoch]
                elif type(learning_rate) is float:
                    lr = learning_rate
                else:
                    tf.compat.v1.logging.info(
                        "Learning rate should be a list of double or a double scalar."
                    )
                    quit()
                record_summary(summary_writer, "learning_rate", lr, global_step)

                # load data
                Xs, Ys, Ps, Ms = train_provider.next_batch_multigpu(NUM_GPUS)

                # iteration
                for iter in range(iters_per_epoch):
                    # training
                    _, xhats, Hxhats, loss, mse, ph_reg, ab_reg = sess.run(
                        [train_op, xhat_op, Hxhat_op] + statistics_op,
                        feed_dict={
                            self.lr: lr,
                            self.Xs: Xs,
                            self.Ys: Ys,
                            self.Ps: Ps,
                            self.Ms: Ms,
                        },
                    )
                    if iter % FLAGS.log_iter == 0:
                        tf.compat.v1.logging.info(
                            "[Global Step {}] [Epoch {}: {}/{}] [Total = {}] [MSE = {}] "
                            "[Ph_Reg = {}] [Ab_Reg = {}]".format(
                                global_step,
                                epoch + 1,
                                iter + 1,
                                iters_per_epoch,
                                loss,
                                mse,
                                ph_reg,
                                ab_reg,
                            )
                        )

                    # record total loss, mse, ph_reg & ab_reg
                    record_summary(summary_writer, "total_loss", loss, global_step)
                    record_summary(summary_writer, "mse", mse, global_step)
                    record_summary(summary_writer, "phase_reg", ph_reg, global_step)
                    record_summary(summary_writer, "abs_reg", ab_reg, global_step)
                    # record summary
                    if (epoch + 1) % FLAGS.image_save_epoch == 0:
                        summaries = sess.run(
                            summary_op,
                            feed_dict={
                                self.lr: lr,
                                self.Xs: Xs,
                                self.Ys: Ys,
                                self.Ps: Ps,
                                self.Ms: Ms,
                            },
                        )
                        for summary in summaries:
                            summary_writer.add_summary(summary, global_step + 1)

                    # global_step ++
                    global_step += 1

                # collect memory
                gc.collect()

                # update provider
                idx = 0
                for xhat, Hxhat in zip(xhats, Hxhats):
                    current_error, new_error = train_provider.update(
                        xhat, partial_estimate=Hxhat, tower_idx=idx
                    )
                    idx += 1
                record_summary(
                    summary_writer, "current_error", current_error, epoch + 1
                )
                record_summary(summary_writer, "new_error", new_error, epoch + 1)

                # logs
                execution_time = time.perf_counter() - current_time
                total_time += execution_time
                print(
                    "**** [Total epoch time: {:0.4f} seconds] "
                    "[Average iteration time: {:0.4f} seconds] ****".format(
                        execution_time, total_time / (epoch - FLAGS.start_epoch + 1)
                    )
                )

                # save model
                if (epoch + 1) % FLAGS.model_save_epoch == 0:
                    self.save(sess, output_path, epoch=epoch + 1)
                if (epoch + 1) % FLAGS.intermediate_result_save_epoch == 0:
                    self.save(sess, output_path, train_provider=train_provider)
                else:
                    self.save(sess, output_path)

                if (
                    FLAGS.num_measurements_per_batch > 0
                    and (epoch + 1) % train_provider.scheduler.get_total_blocks() == 0
                ):
                    Hreal, Himag = train_provider.sample_partial_tf(
                        FLAGS.num_measurements_per_batch
                    )
                    sess.run(
                        H_op,
                        feed_dict={
                            self.Himag_update_placeholder: Himag,
                            self.Hreal_update_placeholder: Hreal,
                        },
                    )
        tf.compat.v1.logging.info("Training Ends")

    def __parallelization(self):
        # set up placeholder
        self.lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        # Setup placeholder and variables
        self.Hreal_update_placeholder = tf.compat.v1.placeholder(
            tf.complex64, shape=[None, None, None, None, None], name="Hreal_placeholder"
        )
        self.Himag_update_placeholder = tf.compat.v1.placeholder(
            tf.complex64, shape=[None, None, None, None, None], name="Himag_placeholder"
        )
        self.Xs = tf.compat.v1.placeholder(
            tf.float32, shape=[NUM_GPUS, None, 3], name="coordinate_placeholder"
        )
        self.Ys = tf.compat.v1.placeholder(
            tf.float32,
            shape=[NUM_GPUS, None, None, None],
            name="measurement_placeholder",
        )
        self.Ps = tf.compat.v1.placeholder(
            tf.int32, shape=[NUM_GPUS, 5, 2], name="padding"
        )
        self.Ms = tf.compat.v1.placeholder(
            tf.float32, shape=[NUM_GPUS, None, None], name="margin"
        )

        # set up optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # calculate the gradients for each model tower.
        tower_xhat = []
        tower_Hxhat = []
        tower_H = []
        tower_grads = []
        total_loss = tf.constant(0, dtype=tf.float32)
        total_mse = tf.constant(0, dtype=tf.float32)
        total_phase = tf.constant(0, dtype=tf.float32)
        total_absor = tf.constant(0, dtype=tf.float32)
        reuse_var = False
        for i in range(NUM_GPUS):
            with tf.device("/GPU:{}".format(i)):
                # load H for each tower
                load_H_op, Hreal, Himag = self.__load_H(reuse_var)
                # keep track of load_op_H accross all towers
                tower_H.append(load_H_op)
                # define the loss for each tower.
                (
                    loss,
                    mse,
                    phase_regularize_value,
                    absorption_regularize_value,
                    xhat,
                    Hxhat,
                    y,
                ) = self.__tower_loss(i, Hreal, Himag, reuse=reuse_var)
                total_loss = total_loss + loss
                total_mse = total_mse + mse
                total_phase = total_phase + phase_regularize_value
                total_absor = total_absor + absorption_regularize_value
                tower_xhat.append(xhat)
                tower_Hxhat.append(Hxhat)
                # calculate the gradients for the batch of data on this tower.
                grads = opt.compute_gradients(loss)
                # keep track of the gradients across all towers.
                tower_grads.append(grads)
                # reuse variable
                reuse_var = True
        # load H operation
        H_op = tf.group(tower_H)
        # we must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = self.__average_gradients(tower_grads)
        # apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads)
        # collect statistics
        statistics_op = [
            tf.multiply(total_loss, 1 / NUM_GPUS),
            tf.multiply(total_mse, 1 / NUM_GPUS),
            tf.multiply(total_phase, 1 / NUM_GPUS),
            tf.multiply(total_absor, 1 / NUM_GPUS),
        ]
        # keep track of xhat & Hxhat obtained by the last tower
        Hxhat_summary = tf.compat.v1.summary.image(
            "Hxhat", reshape_image_3(Hxhat[:2]), max_outputs=2
        )
        chopped_ground_truth_summary = tf.compat.v1.summary.image(
            "chopped_ground_truth", reshape_image_3(y[:2]), max_outputs=2
        )
        # summary operation
        summary_op = tf.tuple(tensors=[Hxhat_summary, chopped_ground_truth_summary])
        # xhat operation
        xhat_op = tf.tuple(tensors=tower_xhat)
        # Hxhat operation
        Hxhat_op = tf.tuple(tensors=tower_Hxhat)
        # create a saver
        self.saver = tf.compat.v1.train.Saver(
            [
                v
                for v in tf.compat.v1.global_variables(
                    scope="{}/{}".format("infer_y", NET_SCOPE)
                )
            ]
        )

        return H_op, train_op, statistics_op, summary_op, xhat_op, Hxhat_op

    def __load_H(self, reuse):
        # build a graph for loading H (this is due to efficiency)
        with tf.compat.v1.variable_scope("load_H", reuse=reuse):
            # set up operation for loading H
            z = np.zeros((24, 1, 52, 2, 2))
            Hreal = tf.Variable(
                self.Hreal_update_placeholder,
                validate_shape=False,
                trainable=False,
                dtype=tf.complex64,
                name="Hreal",
            )
            self.H_debug = Hreal
            Himag = tf.Variable(
                self.Himag_update_placeholder,
                validate_shape=False,
                trainable=False,
                dtype=tf.complex64,
                name="Himag",
            )
            Hreal_update_op = tf.compat.v1.assign(
                Hreal, self.Hreal_update_placeholder, validate_shape=False
            )
            Himag_update_op = tf.compat.v1.assign(
                Himag, self.Himag_update_placeholder, validate_shape=False
            )
            load_H_op = tf.group(Hreal_update_op, Himag_update_op)
        return load_H_op, Hreal, Himag

    def __average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        # single GPU collect gradients to cpu (inefficient)
        if len(tower_grads) == 1:
            return tower_grads[0]

        # muliple GPU
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(input_tensor=grad, axis=0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    #########################
    ###    Prediction     ###
    #########################

    def predict(self, model_path, mesh_grid, Hreal=None, Himag=None):
        """Perform the inference of MLP
        Args:
            model_path: path to the saved model.
            mesh_grid:
            Hreal: phase light transfer function.
            Himag: absorption light transfer function.
        Returns:
            xhat: final reconstruction.
        """
        z, x, y, _ = mesh_grid.shape

        # placeholder
        xhat = np.zeros((z, x, y, 2))
        Hxhat = np.zeros((2, x, y))

        # Fourier transform
        F = lambda x: np.fft.fft2(x)
        iF = lambda x: np.fft.ifft2(x)

        with tf.device("/cpu:0"):
            with tf.compat.v1.Session() as sess:
                # get corresponding operators
                _, _, _, _, xhat_op, _ = self.__parallelization()
                # Initialize variables
                z = np.zeros((1, 1, 1, 1, 1)) + 1j * np.zeros((1, 1, 1, 1, 1))
                sess.run(
                    tf.compat.v1.global_variables_initializer(),
                    feed_dict={
                        self.Himag_update_placeholder: z,
                        self.Hreal_update_placeholder: z,
                    },
                )
                # Restore model weights from previously saved model
                self.restore(sess, model_path)
                # Start
                for start_layer in range(
                    0, mesh_grid.shape[0], FLAGS.prediction_batch_size
                ):
                    # input mesh grid
                    partial_mesh_grid = mesh_grid[
                        start_layer : start_layer + FLAGS.prediction_batch_size
                    ]
                    reshaped_mesh_grid = np.expand_dims(
                        np.reshape(partial_mesh_grid, (-1, 3)), 0
                    )
                    # switch based on Hreal & Himag
                    if Hreal is not None and Himag is not None:
                        # extract H's
                        partial_Hreal = Hreal[
                            :, start_layer : start_layer + FLAGS.prediction_batch_size
                        ]
                        partial_Himag = Himag[
                            :, start_layer : start_layer + FLAGS.prediction_batch_size
                        ]
                        partial_xhat = sess.run(
                            [xhat_op], feed_dict={self.Xs: reshaped_mesh_grid}
                        )
                        Hxhat += np.sum(
                            np.real(
                                iF(
                                    np.multiply(partial_Hreal, F(partial_xhat[..., 0]))
                                    + np.multiply(
                                        partial_Himag, F(partial_xhat[..., 1])
                                    )
                                )
                            ),
                            axis=1,
                        )
                    else:
                        partial_xhat = sess.run(
                            [xhat_op],
                            feed_dict={self.Xs: reshaped_mesh_grid, self.lr: 8e-4},
                        )
                    xhat[
                        start_layer : start_layer + FLAGS.prediction_batch_size
                    ] = partial_xhat[0][0]
                return Hxhat, xhat
