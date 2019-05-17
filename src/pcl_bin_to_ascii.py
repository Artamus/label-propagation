import argparse
import os
import pypcd
import tqdm
import numpy as np


def read_pcl(pointcloud_file_path):
    return pypcd.PointCloud.from_path(pointcloud_file_path)


def save_pcl(pointcloud_file, pointcloud_target_path):
    pointcloud_file.save_pcd(pointcloud_target_path, compression='ascii')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input', type=str,
        help='Directory to read binary PCL files from'
    )

    parser.add_argument(
        'output', type=str,
        help='Directory to put ASCII PCL files'
    )

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    input_dir_files = os.listdir(input_dir)
    input_pcl_files = [
        filename for filename in input_dir_files if filename.endswith('.pcd')]

    for filename in tqdm.tqdm(input_pcl_files):
        filepath = os.path.join(input_dir, filename)

        pcd = read_pcl(filepath)

        x = pcd.pc_data['x']
        y = pcd.pc_data['y']
        z = pcd.pc_data['z']
        label = pcd.pc_data['label']

        label[label < 0] = 0

        pcd = pypcd.make_xyz_label_point_cloud(np.c_[x, y, z, label])

        target_filepath = os.path.join(output_dir, filename)

        save_pcl(pcd, target_filepath)
