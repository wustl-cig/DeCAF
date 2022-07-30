"""
This file runs inference on DECAF model and generates readable results.
Created by Renhao Liu, CIG, WUSTL, 2021.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import h5py
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from absl import app, flags
import imageio
from PIL import Image

from model.model import Model
from model.provider import DecafEndToEndProvider

FLAGS = flags.FLAGS
# input, output dirs.
flags.DEFINE_string("input_dir", "", "directory for Hreal & Himag")
flags.DEFINE_string("model_save_dir", "saved_model", "directory for saving model")
flags.DEFINE_string("result_save_dir", "result", "directory for saving results")

# Prediction config.
flags.DEFINE_float("z_min", -10, "minimum depth in micrometer")
flags.DEFINE_float("z_max", 16, "maximum depth in micrometer")
flags.DEFINE_float("z_train_delta", 0.5, "z delta in training data")
flags.DEFINE_float("z_delta", 0.1, "depth for each layer in micrometer")

flags.DEFINE_boolean(
    "partial_render",
    False,
    "Whether to render a subset of z. z_render_min, z_render_max only"
    "works if this is True",
)
flags.DEFINE_float("z_render_min", -20, "minimum depth to render in micrometer")
flags.DEFINE_float("z_render_max", 60, "maximum depth to render in micrometer")

flags.DEFINE_integer("row_render_min", 0, "minimum row to render in pixel")
flags.DEFINE_integer("row_render_max", 100, "maximum row to render in pixel")

flags.DEFINE_integer("col_render_min", 0, "minimum col to render in pixel")
flags.DEFINE_integer("col_render_max", 100, "maximum col to render in pixel")
flags.DEFINE_float("super_resolution_scale", 1, "super resolution scale")

# Render config.
flags.DEFINE_float("n0", 1.33, "n0 of the medium.")
flags.DEFINE_float("render_max", 0.02, "Range above average in rendering.")
flags.DEFINE_float("render_min", 0.02, "Range below average in rendering.")

#Permittivity to RI conversion
def perm2RI(er, ei, n0):
    """
    Description: This function converts the recovered object's permittivity contrast into refractive index values more
                 commonly found in the literature.
    :param er:  Scalar, 2D, or 3D matrix containing object's real permittivity contrast
    :param ei:  scalar, 2D, or 3D matrix containing object's imaginary permittivity contrast
    :param n0:  Scalar value containing value of imaging medium's refractive index value. in Air n0 = 1, in water n0 = 1.33
    :return: nr: Scalar, 2D, or 3D matrix of object's real refractive index value.
             ni: scalar, 2D, or 3D matrix of object's imaginary refractive index value.
    """
    print("er max: {}, er min:{}".format(er.max(), er.min()))
    nr = np.sqrt(0.5 * ((n0**2 + er) + np.sqrt((n0**2 + er)**2 + ei**2)))
    ni = np.divide(ei, 2 * nr)
    return nr, ni

def main(argv):
    """
    DECAF prediction main function.
    """
    print("DECAF prediction started. Loading files.")
    data = h5py.File(FLAGS.input_dir, 'r')
    provider = DecafEndToEndProvider(data, [0, 1])
    
    print("Inference started.")
    tic = time.perf_counter()
    
    rows = int(provider.measurement_size)
    cols = int(provider.measurement_size)

    assert FLAGS.z_min < FLAGS.z_min + FLAGS.z_delta < FLAGS.z_max
    key_zs = np.ceil((FLAGS.z_max + 1e-8 - FLAGS.z_min) / FLAGS.z_train_delta)
    zs = np.ceil((FLAGS.z_max + 1e-8 - FLAGS.z_min) / FLAGS.z_delta)

    if FLAGS.partial_render:
        scale = FLAGS.super_resolution_scale
        adjustment = 0.5 * (scale - 1) / scale
        rows_idx = np.linspace(
            FLAGS.row_render_min - adjustment,
            FLAGS.row_render_max - 1 + adjustment,
            num=int(
                (FLAGS.row_render_max - FLAGS.row_render_min)
                * FLAGS.super_resolution_scale
            ),
        )
        cols_idx = np.linspace(
            FLAGS.col_render_min - adjustment,
            FLAGS.col_render_max - 1 + adjustment,
            num=int(
                (FLAGS.col_render_max - FLAGS.col_render_min)
                * FLAGS.super_resolution_scale
            ),
        )

        assert FLAGS.z_min <= FLAGS.z_render_min <= FLAGS.z_render_max <= FLAGS.z_max

        key_z_min = (FLAGS.z_render_min - FLAGS.z_min) / FLAGS.z_train_delta
        partial_zs = np.ceil(
            (FLAGS.z_render_max + 1e-8 - FLAGS.z_render_min) / FLAGS.z_delta
        )
        key_z_max = (
            FLAGS.z_render_min + (partial_zs - 1) * FLAGS.z_delta - FLAGS.z_min
        ) / FLAGS.z_train_delta
        print(key_z_max)
        zs_idx = np.linspace(key_z_min, key_z_max, num=int(partial_zs))
    else:
        rows_idx = np.arange(0, rows)
        cols_idx = np.arange(0, cols)
        zs_idx = np.linspace(0, key_zs - 1, num=int(zs))
    r_mesh, z_mesh, c_mesh = np.meshgrid(cols_idx, zs_idx, rows_idx)

    r_mesh = (r_mesh / rows)[..., np.newaxis] - 0.5
    c_mesh = (c_mesh / cols)[..., np.newaxis] - 0.5
    z_mesh = (z_mesh / key_zs)[..., np.newaxis] - 0.5
    mesh_grid = np.concatenate((r_mesh, c_mesh, z_mesh), axis=-1) * 2

    FLAGS.view_size = rows_idx.size
    model = Model()
    output_dir = FLAGS.result_save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _, recon = model.predict(FLAGS.model_save_dir, mesh_grid)
    if FLAGS.partial_render:
        save_name = "prediction_result_zmax{}_zmin{}_zdelta{}_{}_{}_{}_{}_{}_to_{}_x{}".format(
            output_dir,
            FLAGS.z_max,
            FLAGS.z_min,
            FLAGS.z_delta,
            FLAGS.row_render_min,
            FLAGS.row_render_max,
            FLAGS.col_render_min,
            FLAGS.col_render_max,
            FLAGS.z_render_min,
            FLAGS.z_render_max,
            FLAGS.super_resolution_scale,
        )
        save_path = "{}/{}.mat".format(output_dir, save_name)
    else:
        save_name = "prediction_result_zmax{}_zmin{}_zdelta{}".format(
            FLAGS.z_max,
            FLAGS.z_min,
            FLAGS.z_delta,
        )
        save_path = "{}/{}.mat".format(
            output_dir,
            save_name
        )
    toc = time.perf_counter()
    print("Inference ended in {:4} seconds.".format(toc - tic))
    with h5py.File(save_path, "w") as h5_file:
        h5_file.create_dataset("recon", data=recon)
    
    print("Prediction saved to {}".format(save_path))

    ab = recon[:, :, :, 1]
    ph = recon[:, :, :, 0]

    visual = "n_re"
    n_re, n_im = perm2RI(ph, ab, FLAGS.n0)
    result = n_re
    
    if visual == 'n_re':
        up = FLAGS.n0 + FLAGS.render_max;
        low = FLAGS.n0 + FLAGS.render_min;
    else:
        up = FLAGS.render_max
        low = FLAGS.render_min
    mu = (up + low) / 2;
    w = up - low
    result = np.clip(result, low, up)
    result -= np.min(result)
    result /= np.max(result)
    result *= 255
    result = result.astype(np.uint8)

    video_frames =[]
    image_dir = '{}/{}/'.format(output_dir, save_name)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    for idx, img in enumerate(result):
        im = Image.fromarray(img.T)
        im.save(image_dir + 'img_{}.tif'.format(idx))
        video_frames.append(img.T)

    f = '{}/{}.mp4'.format(output_dir, save_name)
    imageio.mimwrite(f, video_frames, fps=8, quality=7)

if __name__ == "__main__":
    app.run(main)
