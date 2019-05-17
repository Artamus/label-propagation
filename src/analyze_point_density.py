import argparse
import os
import tqdm
import pypcd
import itertools
import pickle
import numpy as np


def read_pointcloud_file(pointcloud_file_path):
    return pypcd.PointCloud.from_path(pointcloud_file_path)


def read_pickle_file(pickle_file_path):
    with open(pickle_file_path, 'rb') as fp:
        scene_points_list = pickle.load(fp)
        semantic_labels_list = pickle.load(fp)

    del semantic_labels_list

    return scene_points_list


def get_nr_points_per_voxel(points, voxel_size):

    voxel_big_delta = voxel_size * 0.2 / 1.5

    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)

    # TODO: Add z-axis
    nsubvolume_x = np.ceil(
        (coord_max[0] - coord_min[0]) / voxel_size).astype(np.int32)
    nsubvolume_y = np.ceil(
        (coord_max[1] - coord_min[1]) / voxel_size).astype(np.int32)

    nr_points = []

    for i, j in itertools.product(range(nsubvolume_x), range(nsubvolume_y)):

        current_min = coord_min + [i * voxel_size, j * voxel_size, 0]
        current_max = coord_min + \
            [(i + 1) * voxel_size, (j + 1)
             * voxel_size, coord_max[2] - coord_min[2]]

        curchoice = np.sum((points >= (current_min - voxel_big_delta)) *
                           (points <= (current_max + voxel_big_delta)), axis=1) == 3

        current_points = points[curchoice]
        nr_points.append(len(current_points))

    return nr_points


def input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input', type=str,
        help='Path to directory containing labelled pointcloud files'
    )

    parser.add_argument(
        '-v', '--vox_size', type=float,
        default=1.5,
        help='Voxel size for voxelizing the scene [default: 1.5]'
    )

    parser.add_argument(
        '--pickle', action='store_true',
        help='If there\'s a pickle file with the data instead of .pcd files'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = input_arguments()

    input_dir = args.input
    voxel_size = args.vox_size
    is_pickle = args.pickle

    print('Input directory: %s' % input_dir)

    point_densities = []

    if is_pickle:

        files = os.listdir(input_dir)
        files = [
            filename for filename in files if filename.endswith('.pickle')]

        all_points = []

        for filename in files:
            file_path = os.path.join(input_dir, filename)

            points = read_pickle_file(file_path)
            all_points.extend(points)

        print('Number of scenes: %d' % len(all_points))

        for points in tqdm.tqdm(all_points):
            pointcloud_points_per_voxel = get_nr_points_per_voxel(
                points, voxel_size)
            point_densities.extend(pointcloud_points_per_voxel)

    else:
        pointcloud_files = os.listdir(input_dir)
        pointcloud_files = [
            filename for filename in pointcloud_files if filename.endswith('.pcd')]

        print('Number of scenes: %d' % len(pointcloud_files))

        for pointcloud_file in tqdm.tqdm(pointcloud_files):
            pointcloud_file_path = os.path.join(input_dir, filename)
            pointcloud = read_pointcloud_file(pointcloud_file_path)

            x = pointcloud.pc_data['x']
            y = pointcloud.pc_data['y']
            z = pointcloud.pc_data['z']
            points = np.c_[x, y, z]

            pointcloud_points_per_voxel = get_nr_points_per_voxel(
                points, voxel_size)
            point_densities.extend(pointcloud_points_per_voxel)

    point_densities = np.array(point_densities)

    print('Total amount of voxels: %d' % len(point_densities))
    print('Number of non-empty voxels: %d' % np.count_nonzero(point_densities))

    print('Fraction of non-empty voxels: %.2f' % (1.0 * np.count_nonzero(point_densities) / len(point_densities)))

    print('Mean amount of points in a voxel (incl empty): %.3f' %
          point_densities.mean())

    point_densities_nonzero = point_densities[point_densities.nonzero()]
    print('Mean amount of points in a voxel: %.3f' %
          point_densities_nonzero.mean())
    print('Median amount of points in a voxel: %d' %
          np.median(point_densities_nonzero))
    print('75th percentile amount of points in a voxel: %d' %
          np.percentile(point_densities_nonzero, 75))
    print('90th percentile amount of points in a voxel: %d' %
          np.percentile(point_densities_nonzero, 90))
    print('95th percentile amount of points in a voxel: %d' %
          np.percentile(point_densities_nonzero, 95))
    print('99th percentile amount of points in a voxel: %d' %
          np.percentile(point_densities_nonzero, 99))
    print('Max amount of points in a voxel: %d' % point_densities.max())
