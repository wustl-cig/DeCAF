"""
Training main function for DECAF.
Created by Renhao Liu, CIG, WUSTL, 2021.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import h5py
from absl import app
from absl import flags

from model.model import Model
from model.provider import DecafEndToEndProvider

FLAGS = flags.FLAGS

# Version parameters
flags.DEFINE_string("name", "DECAF", "model name")

# Replace current parameters.
flags.DEFINE_string("input_dir", "", "input_file")
flags.DEFINE_string("model_save_dir", "saved_model", "directory for saving model")

# Training hyper parameters
flags.DEFINE_integer("epochs", 40000, "number of training epochs")
flags.DEFINE_float("start_lr", 8e-4, "number of training epochs")
flags.DEFINE_float("end_lr", 3e-4, "number of training epochs")

def main(argv):
    """
        DECAF main function
    """
    lr_start = FLAGS.start_lr
    lr_end = FLAGS.end_lr
    epochs = FLAGS.epochs
    multiplier = (lr_end / lr_start) ** (1 / epochs)
    decayed_lr = [lr_start * (multiplier ** x) for x in range(epochs)]

    train_kargs = {
        "epochs": epochs,
        "learning_rate": decayed_lr,
    }
    directory = FLAGS.model_save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    config_file = open(directory + "/config.txt", "w")
    config_file.write(FLAGS.flags_into_string())
    config_file.close()

    h5_file = h5py.File(FLAGS.input_dir, "r")
    train_provider = DecafEndToEndProvider(h5_file)
    model = Model(name=FLAGS.name)
    model.train(FLAGS.model_save_dir, train_provider, **train_kargs)

if __name__ == "__main__":
    app.run(main)
