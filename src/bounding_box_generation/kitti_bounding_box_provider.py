import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def read_kitti_calibration_file(file_path):
    calib = dict()

    with open(file_path, 'r') as f:
        for line in f:
            if len(line) < 5:
                continue

            key, val = line.rstrip().split(': ')
            calib[key] = np.array(map(float, val.split(' ')))

    return {
        'Tr_velo_to_cam': calib['Tr_velo_to_cam'].reshape((3, 4)),
        'Tr_cam_to_velo': inverse_rigid_transform(calib['Tr_velo_to_cam'].reshape((3, 4))),
        'R0_rect': calib['R0_rect'].reshape((3, 3)),
        'P2': calib['P2'].reshape((3, 4))
    }


# Transformation code inspired by Frustum-PointNet code at https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
def inverse_rigid_transform(transform):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_transform = np.zeros_like(transform)  # 3x4
    inv_transform[0:3, 0:3] = np.transpose(transform[0:3, 0:3])
    inv_transform[0:3,
                  3] = np.dot(-np.transpose(transform[0:3, 0:3]), transform[0:3, 3])
    return inv_transform


def project_rect_to_ref(r0, pts_3d_rect):
    ''' Input and output are nx3 '''
    return np.transpose(np.dot(np.linalg.inv(r0), np.transpose(pts_3d_rect)))


def cartesian_to_homogenous(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_ref_to_velo(c2v, pts_3d_ref):
    pts_3d_ref = cartesian_to_homogenous(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(c2v))


def project_rect_to_velo(r0, c2v, pts_3d_rect):
    pts_3d_ref = project_rect_to_ref(r0, pts_3d_rect)
    return project_ref_to_velo(c2v, pts_3d_ref)


def project_velo_to_ref(v2c, pts_3d_velo):
    pts_3d_velo = cartesian_to_homogenous(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, np.transpose(v2c))


def project_ref_to_rect(r0, pts_3d_ref):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(r0, np.transpose(pts_3d_ref)))


def project_velo_to_rect(r0, v2c, pts_3d_velo):
    pts_3d_ref = project_velo_to_ref(v2c, pts_3d_velo)
    return project_ref_to_rect(r0, pts_3d_ref)


def project_rect_to_image(p, pts_3d_rect):
    pts_3d_rect = cartesian_to_homogenous(pts_3d_rect)
    pts_2d = np.dot(pts_3d_rect, np.transpose(p)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:,0:2]

def project_velo_to_image(p, r0, v2c, pts_3d_velo):
    pts_3d_rect = project_velo_to_rect(r0, v2c, pts_3d_velo)
    return project_rect_to_image(p, pts_3d_rect)


def rotate_points_around(all_points, around, angle):
    rotation = R.from_euler('z', -1.0 * angle)

    transformed_points = np.zeros_like(all_points)

    for ind in range(len(all_points)):
        points = all_points[ind]

        centered_points = points - around
        rotated_points = rotation.apply(centered_points)
        final_points = rotated_points + around

        transformed_points[ind] = final_points

    return transformed_points


def rotate_points_around_another_point(points, around, angle):
    rotation = R.from_euler('z', [-1.0 * angle])

    centered_points = points - around
    rotated_points = rotation.apply(centered_points)

    return (rotated_points + around)[:, :2]


def rotate_points_around_another_point_3d(points, around, angle):
    rotation = R.from_euler('z', [-1.0 * angle])

    centered_points = points - around
    rotated_points = rotation.apply(centered_points)

    return (rotated_points + around)


def get_bounding_boxes(bounding_box_file_path, allowed_classes, calibration_file_path):
    """Read in KITTI ground truth bounding boxes for a given scene

    Arguments:
        bounding_box_file_path string -- path to the file containing the bounding boxes
        allowed_classes set[string] -- a set containing the allowed classes, all other classes are filtered out
        calibration_file_path string -- path to the file containing the camera calibrations

    Returns:
        np.ndarray -- matrix of bounding box parameters with rows being bounding boxes for the scene and columns being values
                      columns: x, y, z, height, width, length, angle
    """
    transform = read_kitti_calibration_file(calibration_file_path)

    label_data = list()
    with open(bounding_box_file_path, 'r') as f:
        for line in f:
            elements = line.rstrip().split(' ')

            if elements[0] not in allowed_classes:
                continue

            label_data.append(np.array([elements[11], elements[12], elements[13], elements[8], elements[9], elements[10], elements[14]], dtype=np.float32))

    label_data = np.array(label_data)

    if len(label_data) == 0:
        return label_data

    label_data[:, 0:3] = project_rect_to_velo(transform['R0_rect'], transform['Tr_cam_to_velo'], label_data[:, 0:3])
    return label_data


def count_points_bboxes_matching(testable_points, bounding_boxes):
    # TODO
    # Match cluster centers to bounding boxes
    # One option is to calculate distances between the centerpoints and bbox centerpoints to match
    # and only then to do the "inside" check
    points = np.array(testable_points.values())[:, :2]  # If there are more than 2 dimensions included

    points_in_boxes = np.zeros((len(points)), dtype=bool)

    for bbox in bounding_boxes:
        xyz = bbox[:3]
        length, width = bbox[3:5]
        angle = bbox[6]

        p1 = xyz.copy()
        p1[0] -= width / 2
        p1[1] -= length / 2

        p2 = xyz.copy()
        p2[0] += width / 2
        p2[1] -= length / 2

        p3 = xyz.copy()
        p3[0] -= width / 2
        p3[1] += length / 2

        # Only for debugging
        p4 = xyz.copy()
        p4[0] += width / 2
        p4[1] += length / 2

        vertices = np.asarray((p1, p2, p3, p4))
        p1, p2, p3, p4 = rotate_points_around_another_point(vertices, xyz, angle)

        i = p2 - p1
        ii = i.dot(i)
        j = p3 - p1
        jj = j.dot(j)

        v = points - p1

        vi = v.dot(i)
        vj = v.dot(j)

        points_inside_box = (0 < vi) & (vi < ii) & (0 < vj) & (vj < jj)

        points_in_boxes |= points_inside_box

    nr_points_in_boxes = points_in_boxes.sum()
    nr_points_outside_boxes = len(points) - nr_points_in_boxes
    nr_empty_boxes = len(bounding_boxes) - nr_points_in_boxes

    # Band aid to deal with the fact that if multiple centerpoints fall into a box, this is not taken into account
    if nr_empty_boxes < 0:
        nr_empty_boxes = 0

    #TP, FP, FN
    return nr_points_in_boxes, nr_points_outside_boxes, nr_empty_boxes


def transform_points_velo_to_rect(points, calibration_file_path):
    if len(points) == 0:
        return points

    transform = read_kitti_calibration_file(calibration_file_path)
    transformed_points = project_velo_to_rect(transform['R0_rect'], transform['Tr_velo_to_cam'], points)
    return transformed_points


def get_2d_bbox_from_3d_alt(bounding_boxes, calibration_file_path):
    calibration = read_kitti_calibration_file(calibration_file_path)

    boxes_2d = np.zeros((len(bounding_boxes), 4))

    for ind, bbox in enumerate(bounding_boxes):

        image_points = project_velo_to_image(calibration['P2'], calibration['R0_rect'], calibration['Tr_velo_to_cam'], bbox)

        image_left, image_right = image_points[:, 0].min(), image_points[:, 0].max()
        image_top, image_bottom = image_points[:, 1].min(), image_points[:, 1].max()

        boxes_2d[ind] = [image_left, image_top, image_right, image_bottom]

    return np.array(boxes_2d)
