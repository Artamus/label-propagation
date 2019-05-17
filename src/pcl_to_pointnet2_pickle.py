import pypcd
import os
import glob
import argparse
import pickle
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def read_pointcloud_file(pointcloud_file_path):
    return pypcd.PointCloud.from_path(pointcloud_file_path)


def read_pointcloud(input_dir):
    pointcloud_file_paths = sorted(glob.glob(os.path.join(input_dir, '*.pcd')))

    filenames_list = []
    pointclouds_list = []
    labels_list = []

    for pointcloud_file_path in tqdm.tqdm(pointcloud_file_paths):
        pointcloud = read_pointcloud_file(pointcloud_file_path)

        x = pointcloud.pc_data['x']
        y = pointcloud.pc_data['y']
        z = pointcloud.pc_data['z']

        pointcloud_matrix = np.c_[x, y, z]
        labels = pointcloud.pc_data['label']
        labels[labels == -1] = 0
        labels = labels.astype(np.uint8)

        filenames_list.append(os.path.basename(pointcloud_file_path)[:-4])
        pointclouds_list.append(pointcloud_matrix)
        labels_list.append(labels)

    return np.array(filenames_list), np.array(pointclouds_list), np.array(labels_list)


def save_pointcloud_data(output_dir, filename, pointclouds, labels):

    output_file_path = os.path.join(
        output_dir, '{name:s}.pickle'.format(name=filename))

    with open(output_file_path, 'wb') as f:
        pickle.dump(pointclouds, f)
        pickle.dump(labels, f)


def save_split_filenames(output_dir, output_filename, filenames):
    filenames = sorted(filenames)
    output_data = '\n'.join(filenames)
    
    output_file = os.path.join(output_dir, '{name:s}.txt'.format(name=output_filename))
    with open(output_file, 'w') as f:
        f.write(output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing labelled pointcloud in .pcd format'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to directory that will contain the processed data'
    )

    parser.add_argument(
        '-n', '--name', type=str,
        default='',
        help='Base name to give to the resulting pickles'
    )

    parser.add_argument(
        '--test', action='store_true',
        help='Make a triple split of train-val-test instead of just train-val'
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    output_name = args.name
    test_split = args.test

    if len(output_name) > 0:
        output_name += '_'

    if not os.path.isdir(input_dir):
        exit('Input directory does not exist')

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    filenames, pointclouds, labels = read_pointcloud(input_dir)

    indices = np.arange(len(filenames))
    indices_train, indices_val = train_test_split(indices, test_size=0.2, random_state=42)

    filenames_train = filenames[indices_train]
    pointclouds_train = pointclouds[indices_train]
    labels_train = labels[indices_train]

    print('Saving training data pickle')
    save_pointcloud_data(output_dir, output_name + 'train', pointclouds_train, labels_train)

    if test_split:
        indices_val, indices_test = train_test_split(indices_val, test_size=0.5, random_state=21)

    filenames_val = filenames[indices_val]
    pointclouds_val = pointclouds[indices_val]
    labels_val = labels[indices_val]

    print('Saving validation data pickle')
    save_pointcloud_data(output_dir, output_name + 'val', pointclouds_val, labels_val)

    if test_split:
        filenames_test = filenames[indices_test]
        pointclouds_test = pointclouds[indices_test]
        labels_test = labels[indices_test]

        print('Saving test data pickle')
        save_pointcloud_data(output_dir, output_name + 'test', pointclouds_test, labels_test)
    
    save_split_filenames(output_dir, output_name + 'train', filenames_train)
    save_split_filenames(output_dir, output_name + 'val', filenames_val)
    save_split_filenames(output_dir, output_name + 'trainval', np.concatenate((filenames_train, filenames_val)))

    if test_split:
        save_split_filenames(output_dir, output_name + 'test', filenames_test)