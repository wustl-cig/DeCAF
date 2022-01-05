"""
Inputs provider class.
Created by Yu Sun, CIG, WUSTL, 2019.
Last modified by Renhao Liu, CIG, WUSTL, 2021.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from absl import flags
from scipy.io import savemat, loadmat
import gc
import h5py

flags.DEFINE_enum("task_type", "idt", ["idt", "aidt", "midt"], "different input type")
flags.DEFINE_integer("view_size", 128, "view size at each iteration")
flags.DEFINE_integer("block_padding", 5, "block padding size")
flags.DEFINE_enum("loss", "l2", ["l1", "l2"], "data fidelity type")
flags.DEFINE_integer(
    "force_update_count", 1, "A block must be updated at least once per this count"
)
FLAGS = flags.FLAGS

def get_meshgrid(z, rows, cols):
    """
    Generate meshgrid of shape (z, r, w, 3) with range from -1 to 1 on each axis
    :param z: dim of z
    :param rows: dim of rows
    :param cols: dim of cols
    :return: meshgrid with shape (z, x, y, 3)
    """
    rows_idx = np.arange(0, rows)
    cols_idx = np.arange(0, cols)
    zs_idx = np.arange(0, z)
    rr, zz, cc = np.meshgrid(rows_idx, zs_idx, cols_idx)

    rr = (rr / rows)[..., np.newaxis] - 0.5
    cc = (cc / cols)[..., np.newaxis] - 0.5
    zz = (zz / z)[..., np.newaxis] - 0.5
    mesh_grid = np.concatenate((rr, cc, zz), axis=-1) * 2
    return mesh_grid

EQUAL_PAD_MODE = "equal"
FULL_PAD_MODE = "full"

class DecafEndToEndProvider(object):
    """
    input: radon transformed data: N x h x w
    """

    def __init__(self, layer_1_h5, slices=None):
        measurements, self.num_measurements, self.Hreal, self.Himag = self.extract_data(
            layer_1_h5, slices
        )

        self.view_size = FLAGS.view_size
        self.measurement_size = measurements.shape[-1]
        print(
            "measurement shape: {}, Hreal shape: {}, Himag shape {}".format(
                measurements.shape, self.Hreal.shape, self.Himag.shape
            )
        )
        self.H_size = self.Hreal.shape[-1]
        self.num_layers = self.Hreal.shape[2]
        assert self.view_size <= self.H_size <= self.measurement_size
        self.scheduler = Scheduler(
            self.measurement_size, self.view_size, min_padding=FLAGS.block_padding
        )
        total_blocks = self.scheduler.get_total_blocks()
        self.sample_order = []

        # Pad measurements
        if measurements.shape[-1] > self.H_size:
            print("Using equal padding")
            self.pad_mode = EQUAL_PAD_MODE
            pad = (self.H_size - self.view_size) // 2
            assert (self.H_size - self.view_size) % 2 == 0
            measurements = np.pad(measurements, [[0, 0], [pad, pad], [pad, pad]])
        else:
            print("Using full padding")
            self.pad_mode = FULL_PAD_MODE

        self.measurements = measurements
        self.estimation = np.zeros(
            (self.num_layers, self.measurement_size, self.measurement_size, 2)
        )
        self.current_Ax = np.zeros(
            (total_blocks, self.num_measurements, self.H_size, self.H_size)
        )
        self.raw_sub_measurement = self.measurements.copy()
        if FLAGS.loss == "l1":
            self.loss = self.l1_loss
        elif FLAGS.loss == "l2":
            self.loss = self.l2_loss
        else:
            raise NotImplementedError
        self.current_error = self.loss(self.raw_sub_measurement)
        self.original_error = self.current_error
        self.mesh_grid = None
        self.measurement_samples = np.array(list(range(self.num_measurements)))
        self.padding = None

        self.tower_to_grid = {}
        self.grid_update_counter = {}
        self.mes_picked_rows = {}
        self.mes_picked_cols = {}

    def l1_loss(self, data):
        return np.mean(np.abs(data))

    def l2_loss(self, data):
        return np.mean(np.square(data)) / 2

    def sample_partial_tf(self, count):
        if count <= 0:
            raise ValueError
        self.measurement_samples = np.random.choice(
            self.num_measurements, count, replace=False
        )
        return (
            self.Hreal[self.measurement_samples],
            self.Himag[self.measurement_samples],
        )

    def get_measurement_view_ratio(self):
        """
        Gets the ratio between measurement size and view size.
        Roughly the number of epochs which each pixel is expected to get train once.
        Rounded to a lower integer
        :return:
        """
        return self.measurement_size ** 2 // self.view_size ** 2

    def extract_data(self, file, slices=None):
        """
        :param slices: which measurements to read. None for all measurements.
        :return:
        """
        if FLAGS.task_type == "idt":
            MEASUREMNT_KEY = "I"
            HREAL_KEY = "Hreal"
            HIMAG_KEY = "Himag"
        elif FLAGS.task_type == "aidt":
            MEASUREMNT_KEY = "I_Raw"
            HREAL_KEY = "PTF_4D"
            HIMAG_KEY = "ATF_4D"
        elif FLAGS.task_type == "midt":
            MEASUREMNT_KEY = "dat"
            HREAL_KEY = "Hreal_py"
            HIMAG_KEY = "Himag_py"

        if slices is None:
            measurements = np.squeeze(np.array(file[MEASUREMNT_KEY]))
            num_measurements = measurements.shape[0]
            slices = list(range(num_measurements))
        else:
            measurements = np.squeeze(np.array(file[MEASUREMNT_KEY][slices]))
            num_measurements = measurements.shape[0]
        Hreal = np.array(file[HREAL_KEY]["real"][slices]) + 1j * np.array(
            file[HREAL_KEY]["imag"][slices]
        )
        Himag = np.array(file[HIMAG_KEY][slices])
        Himag = Himag["real"] + 1j * Himag["imag"]
        Hreal = Hreal[:, np.newaxis]
        Himag = Himag[:, np.newaxis]
        return measurements, num_measurements, Hreal, Himag

    def norm_idx(self, zs, rows, cols):
        """
        input three arrays representing the index on three axis
        :param cols:
        :param rows:
        :param zs:
        :return: a normalized meshgrid between 0 and 1
        """
        rr, zz, cc = np.meshgrid(rows, zs, cols)
        rr = (rr / self.measurement_size)[..., np.newaxis] - 0.5
        cc = (cc / self.measurement_size)[..., np.newaxis] - 0.5
        zz = (zz / self.num_layers)[..., np.newaxis] - 0.5
        mesh_grid = np.concatenate((rr, cc, zz), axis=-1) * 2
        return mesh_grid

    def next_batch(self, tower_idx=0):
        if len(self.sample_order) == 0:
            self.sample_order = list(
                np.random.choice(
                    self.scheduler.get_total_blocks(),
                    self.scheduler.get_total_blocks(),
                    replace=False,
                )
            )
        current_block_index = self.sample_order.pop()
        #         print("current_block_index: {}".format(current_block_index))
        self.tower_to_grid[tower_idx] = current_block_index

        start_row, start_col = self.scheduler.get_offset(current_block_index)
        self.picked_rows = list(range(start_row, start_row + self.view_size))
        self.picked_cols = list(range(start_col, start_col + self.view_size))
        zs = list(range(self.num_layers))
        mesh = self.norm_idx(zs, self.picked_rows, self.picked_cols)

        if self.pad_mode == EQUAL_PAD_MODE:
            pad = (self.H_size - self.view_size) // 2
        # z, x, y, ab/ph
        if self.pad_mode == FULL_PAD_MODE:
            self.padding = np.array(
                [
                    [0, 0],
                    [start_row, self.measurement_size - (start_row + self.view_size)],
                    [start_col, self.measurement_size - (start_col + self.view_size)],
                    [0, 0],
                ]
            )
        else:
            self.padding = np.array([[0, 0], [pad, pad], [pad, pad], [0, 0]])

        if self.pad_mode == FULL_PAD_MODE:
            self.mes_picked_rows[tower_idx] = list(
                range(0, self.measurements.shape[-1])
            )
            self.mes_picked_cols[tower_idx] = self.mes_picked_rows[tower_idx]
        else:
            self.mes_picked_rows[tower_idx] = list(
                range(start_row, start_row + self.view_size + 2 * pad)
            )
            self.mes_picked_cols[tower_idx] = list(
                range(start_col, start_col + self.view_size + 2 * pad)
            )

        sub_measurement = (
            self.raw_sub_measurement[
                self.measurement_samples,
                self.mes_picked_rows[tower_idx][0] : self.mes_picked_rows[tower_idx][-1]
                + 1,
                self.mes_picked_cols[tower_idx][0] : self.mes_picked_cols[tower_idx][-1]
                + 1,
            ]
        ).copy()
        complement = self.current_Ax[current_block_index, self.measurement_samples]
        sub_measurement += complement

        outlier_idx = np.abs(sub_measurement) > 10
        sub_measurement[outlier_idx] = self.measurements[
            self.measurement_samples,
            self.mes_picked_rows[tower_idx][0] : self.mes_picked_rows[tower_idx][-1]
            + 1,
            self.mes_picked_cols[tower_idx][0] : self.mes_picked_cols[tower_idx][-1]
            + 1,
        ][outlier_idx]

        top, right, down, left = self.scheduler.get_margin(current_block_index)
        mask = np.ones((self.view_size, self.view_size))
        mask[:top] *= 1 / 2
        if right > 0:
            mask[:, -right:] *= 1 / 2
        if down > 0:
            mask[-down:] *= 1 / 2
        mask[:, :left] *= 1 / 2

        return (
            mesh.reshape(-1, 3),
            sub_measurement,
            np.vstack(([0, 0], self.padding)),
            mask,
        )

    def next_batch_multigpu(self, num_gpus):
        Xs = []
        Ys = []
        Ps = []
        Ms = []
        for i in range(num_gpus):
            X, Y, P, M = self.next_batch(i)
            Xs.append(X)
            Ys.append(Y)
            Ps.append(P)
            Ms.append(M)
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        Ps = np.array(Ps)
        return Xs, Ys, Ps, Ms

    def update(self, data, partial_estimate=None, tower_idx=0):
        """
        :param data: output from network [angle, batch_size, z, x, y]
        :return:
        """
        if partial_estimate is None:
            raise NotImplementedError
        grid_idx = self.tower_to_grid[tower_idx]

        if grid_idx not in self.grid_update_counter:
            self.grid_update_counter[grid_idx] = 0
        update_residual = (
            self.current_Ax[grid_idx, self.measurement_samples] - partial_estimate
        )

        self.raw_sub_measurement[
            self.measurement_samples,
            self.mes_picked_rows[tower_idx][0] : self.mes_picked_rows[tower_idx][-1]
            + 1,
            self.mes_picked_cols[tower_idx][0] : self.mes_picked_cols[tower_idx][-1]
            + 1,
        ] += update_residual

        new_error = self.loss(self.raw_sub_measurement)
        if new_error <= self.current_error or (
            self.grid_update_counter[grid_idx] >= FLAGS.force_update_count
            and new_error <= self.original_error
        ):
            self.current_Ax[grid_idx, self.measurement_samples] = partial_estimate
            self.current_error = new_error
            self.grid_update_counter[grid_idx] = 0
        else:
            self.raw_sub_measurement[
                self.measurement_samples,
                self.mes_picked_rows[tower_idx][0] : self.mes_picked_rows[tower_idx][-1]
                + 1,
                self.mes_picked_cols[tower_idx][0] : self.mes_picked_cols[tower_idx][-1]
                + 1,
            ] -= update_residual
            self.grid_update_counter[grid_idx] += 1
        gc.collect()
        return self.current_error, new_error

    def save(self, dir):
        with h5py.File("{}/intermediate_results.mat".format(dir), "w") as hf:
            hf.create_dataset("current_Ax", data=self.current_Ax)
            hf.create_dataset("raw_sub_measurement", data=self.raw_sub_measurement)
        print("intermediate results saved")

    def restore(self, dir):
        with h5py.File(
            "{}/intermediate_results.mat".format(dir), "r", swmr=True
        ) as data:
            self.raw_sub_measurement = np.array(data["raw_sub_measurement"])
            if "current_Ax" in data:
                self.current_Ax = np.array(data["current_Ax"])
            else:
                raise NotImplementedError
            print("intermediate results restored")


class Scheduler:
    def __init__(self, measurement_size, view_size, min_padding=5):
        self.view_size = view_size
        num_blocks = measurement_size / (view_size - min_padding)
        if not num_blocks.is_integer():
            num_blocks = int(num_blocks) + 1
        else:
            num_blocks = int(num_blocks)
        pixels_to_cover = measurement_size - view_size
        stride = pixels_to_cover // (num_blocks - 1)

        residual = pixels_to_cover % (num_blocks - 1)
        self.indexes = [0]
        for _ in range(num_blocks - 2):
            current_stride = stride
            if residual > 0:
                current_stride += 1
                residual -= 1
            self.indexes.append(self.indexes[-1] + current_stride)

        self.indexes.append(measurement_size - view_size - 1)
        self.num_blocks = num_blocks

    def get_total_blocks(self):
        return self.num_blocks ** 2

    def get_offset_by_row_col(self, row, col):
        return (self.indexes[row], self.indexes[col])

    def get_offset(self, index):
        return self.get_offset_by_row_col(
            index // self.num_blocks, index % self.num_blocks
        )

    def get_margin(self, index):

        row_offset, col_offset = self.get_offset(index)
        row_index = index // self.num_blocks
        col_index = index % self.num_blocks

        if row_index == 0:
            top = 0
        else:
            top_offset, _ = self.get_offset_by_row_col(row_index - 1, col_index)
            top = top_offset + self.view_size - row_offset
        if row_index == self.num_blocks - 1:
            down = 0
        else:
            down_offset, _ = self.get_offset_by_row_col(row_index + 1, col_index)
            down = row_offset + self.view_size - down_offset

        if col_index == 0:
            left = 0
        else:
            _, left_offset = self.get_offset_by_row_col(row_index, col_index - 1)
            left = left_offset + self.view_size - col_offset
        if col_index == self.num_blocks - 1:
            right = 0
        else:
            _, right_offset = self.get_offset_by_row_col(row_index, col_index + 1)
            right = col_offset + self.view_size - right_offset

        return top, right, down, left
