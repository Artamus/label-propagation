# TODO: Make this script more nice and less hard coded
import numpy as np
import os


def read_velo_to_cam_transform(file_path):
    data = dict()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            key, vals = line.split(': ')
            data[key] = vals.split(' ')

    translation = np.array(list(map(float, data['T'])))
    rotation = np.array(list(map(float, data['R']))).reshape(3, 3)

    return np.c_[rotation, translation].flatten()


def read_cam_to_cam_transforms(file_path):
    data = dict()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            key, vals = line.split(': ')
            data[key] = vals.split(' ')

    r0_rect = np.array(list(map(float, data['R_rect_00'])))
    p2 = np.array(list(map(float, data['P_rect_02'])))

    return r0_rect, p2


def to_scientific(x):
    return '{:.12e}'.format(x)


with open('/home/markus/KITTI/semantic/training/matches.txt', 'r') as f:
    matches = []
    for line in f:
        line = line.rstrip()
        match = line.split(' ')
        matches.append((match[0], match[1], match[2]))

raw_calib_dir = '/home/markus/KITTI/raw_data_synced/calib'
output_calib_dir = '/home/markus/KITTI/semantic/training/calib'

for match in matches:
    frame_id = match[0][:-4]
    date = match[1][:10]

    calib_path = os.path.join(raw_calib_dir, date + '_calib', date)

    tr_velo_to_cam = read_velo_to_cam_transform(
        os.path.join(calib_path, 'calib_velo_to_cam.txt'))
    r0_rect, p2 = read_cam_to_cam_transforms(
        os.path.join(calib_path, 'calib_cam_to_cam.txt'))

    final_calibration_data = 'P2: {}\n'.format(
        ' '.join(map(to_scientific, p2)))
    final_calibration_data += 'R0_rect: {}\n'.format(
        ' '.join(map(to_scientific, r0_rect)))
    final_calibration_data += 'Tr_velo_to_cam: {}\n'.format(
        ' '.join(map(to_scientific, tr_velo_to_cam)))

    target_calib_path = os.path.join(output_calib_dir, frame_id + '.txt')
    with open(target_calib_path, 'w') as f:
        f.write(final_calibration_data)
