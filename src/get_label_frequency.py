import argparse
import pypcd
import os
import numpy as np
from cityscapes_classes import id_to_name


def read_pointcloud_file(pointcloud_file_path):
    return pypcd.PointCloud.from_path(pointcloud_file_path)


def input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing labelled pointcloud files'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = input_arguments()

    input_dir = args.input

    pointcloud_files = os.listdir(input_dir)
    pointcloud_files = [
        filename for filename in pointcloud_files if filename.endswith('.pcd')]

    labels = []

    for filename in pointcloud_files:
        pointcloud_file_path = os.path.join(input_dir, filename)
        pointcloud = read_pointcloud_file(pointcloud_file_path)

        label = pointcloud.pc_data['label'].astype(np.int32)
        labels.extend(label)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    unique_names = map(id_to_name, unique)
    print('Unique labels: {:s}'.format(', '.join(unique_names)))
    print('Counts: ')
    print(np.asarray((unique, counts)).T)
