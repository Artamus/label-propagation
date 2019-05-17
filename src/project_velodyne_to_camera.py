import argparse
import os
import numpy as np
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input', type=str,
        default='/home/markus/KITTI/object',
        help='KITTI data input directory'
    )

    parser.add_argument(
        '-n', '--number', type=int,
        default=100,
        help='Number of images from the KITTI dataset to project'
    )

    parser.add_argument(
        '-d', '--distance', type=float,
        default=0.05,
        help='Distance to add for marking closer points larger than faraway points'
    )

    parser.add_argument(
        '-o', '--output', type=str,
        default='/home/markus/KITTI/projection',
        help='Output directory of projected points'
    )

    return parser.parse_args()


def read_calibration(frame_id):
    calibration_path = os.path.join(
        SOURCE_PATH, 'calib', '{:06d}.txt'.format(frame_id))

    raw_calibrations = dict()

    with open(calibration_path, 'r') as f:
        for line in f:
            line = line.rstrip()

            if len(line) == 0:
                continue

            key, val = line.split(': ', 1)
            raw_calibrations[key] = np.array(val.split(' '), dtype=np.float32)

    return {
        'P2': raw_calibrations['P2'].reshape(3, 4),
        'R0': raw_calibrations['R0_rect'].reshape(3, 3),
        'Tr_velo_to_cam': raw_calibrations['Tr_velo_to_cam'].reshape(3, 4)
    }


def project_points(points, calibration):
    points[:, 3] = 1  # We don't use reflectance

    points_transformed = calibration['Tr_velo_to_cam'].dot(points.T)
    points_rectified = calibration['R0'].dot(points_transformed).T
    points_rectified = np.c_[points_rectified,
                             np.ones(points_rectified.shape[0])]

    front_points_ids = points_rectified[:, 2] >= 0
    points_rectified = points_rectified[front_points_ids, :]

    points_projected = points_rectified.dot(calibration['P2'].T)
    points_projected /= points_projected[:, 2:]

    return points_projected[:, :2]


def project_points_with_distance(points, shifted_points, calibration):
    points[:, 3] = 1  # We don't use reflectance
    shifted_points[:, 3] = 1

    points_transformed = calibration['Tr_velo_to_cam'].dot(points.T)
    points_rectified = calibration['R0'].dot(points_transformed).T
    points_rectified = np.c_[points_rectified,
                             np.ones(points_rectified.shape[0])]

    shifted_points_transformed = calibration['Tr_velo_to_cam'].dot(
        shifted_points.T)
    shifted_points_rectified = calibration['R0'].dot(
        shifted_points_transformed).T
    shifted_points_rectified = np.c_[
        shifted_points_rectified, np.ones(shifted_points_rectified.shape[0])]

    front_points_ids = points_rectified[:, 2] >= 0
    points_rectified = points_rectified[front_points_ids, :]
    shifted_points_rectified = shifted_points_rectified[front_points_ids, :]

    points_projected = points_rectified.dot(calibration['P2'].T)
    points_projected /= points_projected[:, 2:]

    shifted_points_projected = shifted_points_rectified.dot(
        calibration['P2'].T)
    shifted_points_projected /= shifted_points_projected[:, 2:]

    return points_projected[:, :2], shifted_points_projected[:, :2]


def project_image(frame_id):
    # Read image
    image_path = os.path.join(SOURCE_PATH, 'image_2',
                              '{:06d}.png'.format(frame_id))
    image = cv2.imread(image_path)

    # Read points
    velodyne_path = os.path.join(
        SOURCE_PATH, 'velodyne', '{:06d}.bin'.format(frame_id))
    points = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4)
    comparison_points = points.copy()
    comparison_points[:, 0] += DISTANCE_SHIFT

    calibration = read_calibration(frame_id)

    points_projected, shifted_points = project_points_with_distance(
        points, comparison_points, calibration)

    # Remove points that are not on the image?
    ind = (points_projected[:, 1] >= 0) & \
        (points_projected[:, 1] <= image.shape[0]) & \
        (points_projected[:, 0] >= 0) & \
        (points_projected[:, 0] <= image.shape[1])

    points_projected = points_projected[ind, :]
    shifted_points = shifted_points[ind, :]

    radiuses = np.linalg.norm(points_projected - shifted_points, axis=1)

    pointcloud_pixels = points_projected.astype(
        np.int32)  # DOES NOT ROUND PROPERLY

    for i, point in enumerate(pointcloud_pixels):
        radius = radiuses[i]
        cv2.circle(image, tuple(point), int(radius), (0, 0, 255), -1)

    return image


args = parse_arguments()

SOURCE_PATH = args.input
SOURCE_PATH = os.path.join(SOURCE_PATH, 'training')
TARGET_PATH = args.output

if not os.path.isdir(TARGET_PATH):
    os.mkdir(TARGET_PATH)

NUM_IMAGES = int(args.number)

DISTANCE_SHIFT = float(args.distance)

for frame_id in range(NUM_IMAGES):
    projected_image = project_image(frame_id)

    projected_image_path = os.path.join(
        TARGET_PATH, '{:06d}.png'.format(frame_id))
    cv2.imwrite(projected_image_path, projected_image)
