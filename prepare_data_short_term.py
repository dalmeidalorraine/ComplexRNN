# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import errno
import zipfile
import numpy as np
import csv
import sys
import re
from urllib.request import urlretrieve
from glob import glob
from common.quaternion import expmap_to_quaternion, qfix
from shutil import rmtree

if __name__ == '__main__':
    def read_file(path):
        '''
        Read an individual file in expmap format,
        and return a NumPy tensor with shape (sequence length, number of joints, 3).
        '''
        data = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(row)
        data = np.array(data, dtype='float64')
        return data.reshape(data.shape[0], -1, 3)

    out_pos = []
    out_rot = []
    out_subjects = []
    out_actions = []
    output_directory = "../Datasets"
    print('Converting dataset...')
    output_filename = 'dataset_h36m'
    output_file_path = output_directory + '/' + output_filename
    subjects = sorted(glob(output_directory + '/h3.6m/dataset/*'))
    for subject in subjects:
        actions = sorted(glob(subject + '/*'))
        result_ = {}
        for action_filename in actions:
            data = read_file(action_filename)

            # Discard the first joint, which represents a corrupted translation
            data = data[:, 1:]

            # Convert to quaternion and fix antipodal representations
            quat = expmap_to_quaternion(-data)
            quat = qfix(quat)

            out_pos.append(np.zeros((quat.shape[0], 3))) # No trajectory for H3.6M
            out_rot.append(quat)
            tokens = re.split('\/|\.', action_filename.replace('\\', '/'))
            subject_name = tokens[-3]
            out_subjects.append(subject_name)
            action_name = tokens[-2]
            out_actions.append(action_name)

    print('Saving...')
    np.savez_compressed(output_file_path,
            trajectories=out_pos,
            rotations=out_rot,
            subjects=out_subjects,
            actions=out_actions)

    rmtree(output_directory + '/h3.6m') # Clean up
    print('Done.')
