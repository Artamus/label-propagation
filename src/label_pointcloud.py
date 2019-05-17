import argparse
import os
import glob
import tqdm
import cv2
import pypcd
import numpy as np

from cityscapes_classes import color_to_id, color_to_train_id, get_number_of_labels, get_number_of_train_labels


def get_point_file_list(pointcloud_dir, pc_type):
    paths = glob.glob(os.path.join(pointcloud_dir,
                                   '*.{file_type:s}'.format(file_type=pc_type)))
    files = [os.path.basename(path) for path in paths]
    return sorted(files)


def read_pointcloud(pointcloud_path, pc_type):
    return POINTCLOUD_TYPES[pc_type](pointcloud_path)


def _read_pointcloud_bin(pointcloud_path):
    points = np.fromfile(pointcloud_path, dtype=np.float32)
    return points.reshape(-1, 4)


def _read_pointcloud_pcd(pointcloud_path):
    raise NotImplementedError


def read_image(image_path):
    return cv2.imread(image_path)


def read_calibration(calibration_path):
    data = dict()
    with open(calibration_path, 'r') as f:
        for line in f:
            if len(line) < 10:
                continue
            line = line.rstrip()
            key, vals = line.split(': ')

            data[key] = np.array(list(map(float, vals.split(' '))))

    return {
        'P2': data['P2'].reshape(3, 4),
        'R0_rect': data['R0_rect'].reshape(3, 3),
        'Tr_velo_to_cam': data['Tr_velo_to_cam'].reshape(3, 4)
    }


def project_points(points, calibration):
    original_points = points.copy()
    points[:, 3] = 1  # We don't use reflectance

    points_transformed = calibration['Tr_velo_to_cam'].dot(points.T)
    points_rectified = calibration['R0_rect'].dot(points_transformed).T
    points_rectified = np.c_[points_rectified,
                             np.ones(points_rectified.shape[0])]

    front_points_ids = points_rectified[:, 2] >= 0
    points_rectified = points_rectified[front_points_ids, :]

    points_projected = points_rectified.dot(calibration['P2'].T)
    points_projected /= points_projected[:, 2:]

    return points_projected[:, :2], original_points[front_points_ids, :]


def remove_points_outside_image(projected_points_pixels, points, image):
    ind = (projected_points_pixels[:, 1] >= 0) & \
        (projected_points_pixels[:, 1] < image.shape[0]) & \
        (projected_points_pixels[:, 0] >= 0) & \
        (projected_points_pixels[:, 0] < image.shape[1])

    return projected_points_pixels[ind, :], points[ind, :]


def get_class_ids(colours, reduced):

    if reduced:
        class_ids = map(color_to_train_id, colours)
    else:
        class_ids = map(color_to_id, colours)

    return np.array(list(class_ids), dtype=np.float32)


def filter_closer_than(points, distance):
    assert distance > 0.0, 'Distance to filter must be positive'

    mask = points[:, 0] <= distance
    return mask


def label_pointcloud_file(point_file_name, pointcloud_dir, image_dir, calibration_dir, pointcloud_type, output_dir, colour=False, reduced=False, distance=0.0):
    frame_id = point_file_name[:-4]

    point_file_path = os.path.join(pointcloud_dir, point_file_name)
    points = read_pointcloud(point_file_path, pointcloud_type)

    labelled_image_path = os.path.join(
        image_dir, '{frame_id:s}.png'.format(frame_id=frame_id))
    labelled_image = read_image(labelled_image_path)

    calibration_path = os.path.join(
        calibration_dir, '{frame_id:s}.txt'.format(frame_id=frame_id))
    calibration = read_calibration(calibration_path)

    projected_points, filtered_points = project_points(points, calibration)
    projected_points_pixels = np.rint(projected_points).astype(np.int64)

    projected_points_pixels, filtered_points = remove_points_outside_image(
        projected_points_pixels, filtered_points, labelled_image)

    if distance > 0.0:
        distance_mask = filter_closer_than(filtered_points, distance)

        projected_points_pixels = projected_points_pixels[distance_mask]
        filtered_points = filtered_points[distance_mask]

    point_labels_colours = labelled_image[projected_points_pixels[:, 1],
                                          projected_points_pixels[:, 0]]

    point_labels_colours[:, [2, 0]] = point_labels_colours[:, [0, 2]]  # BGR to RGB

    if colour:
        rgb = pypcd.encode_rgb_for_pcl(point_labels_colours)
        new_points = np.c_[filtered_points[:, :3], rgb]

        pcd_file = pypcd.make_xyz_rgb_point_cloud(new_points)

    else:
        class_ids = get_class_ids(point_labels_colours, reduced)
        new_points = np.c_[filtered_points[:, :3], class_ids]

        pcd_file = pypcd.make_xyz_label_point_cloud(new_points)

    pcd_file_path = os.path.join(output_dir, frame_id + '.pcd')
    pcd_file.save_pcd(pcd_file_path, compression='binary_compressed')


POINTCLOUD_TYPES = {
    'bin': _read_pointcloud_bin,
    'pcd': _read_pointcloud_pcd
}


def input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'points', type=str,
        help='Path to directory containing pointcloud files'
    )

    parser.add_argument(
        'image', type=str,
        help='Path to directory containing RGB segmented image data'
    )

    parser.add_argument(
        'calibration', type=str,
        help='Path to directory containing calibration data'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to directory that will contain the labelled pointcloud'
    )

    parser.add_argument(
        '-t', '--type',
        choices=['bin', 'pcd'],
        default='bin',
        help='Pointcloud file type'
    )

    parser.add_argument(
        '-d', '--dist', type=float,
        default=0.0,
        help='Distance up to which points are used (anything beyond is discarded). By default, all points will be used.'
    )

    parser.add_argument(
        '-c', '--colour', action='store_true',
        help='If this flag is set, the point cloud won\'t be labelled, but instead the colours will be projected'
    )

    parser.add_argument(
        '--reduced', action='store_true',
        help='If this flag is set, the reduced label space is used instead'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = input_arguments()

    pointcloud_dir = args.points
    image_dir = args.image
    calibration_dir = args.calibration
    output_dir = args.output

    points_type = args.type
    distance = args.dist
    colour = args.colour
    use_reduced_labels = args.reduced

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    point_file_list = get_point_file_list(pointcloud_dir, points_type)
    for point_file_name in tqdm.tqdm(point_file_list):
        label_pointcloud_file(point_file_name, pointcloud_dir,
                              image_dir, calibration_dir, points_type, output_dir, colour, use_reduced_labels, distance)

    if use_reduced_labels:
        nr_labels = get_number_of_train_labels()
    else:
        nr_labels = get_number_of_labels()

    print('The total amount of classes for training is {:d}'.format(nr_labels))
