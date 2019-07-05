import argparse
import os
import pypcd
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from cityscapes_classes import get_train_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict


KITTI_CLASSES_MAP = OrderedDict([
    ('Pedestrian', 12),
    ('Car', 14),
    ('Cyclist', 19)
])
KITTI_CLASSES = KITTI_CLASSES_MAP.keys()
KITTI_CLASSES_VALUES = KITTI_CLASSES_MAP.values()

CITYSCAPES_CLASSES_NAMES = ['unlabelled'] + get_train_labels()
KITTI_CLASSES_NAMES = ['unlabelled'] + get_train_labels()

# TODO: Transformations have moved into their own file, use those


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


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


def read_kitti_label_file(file_path):
    labels = []

    with open(file_path, 'r') as f:
        for line in f:
            labels.append(line.rstrip().split(' '))

    return labels


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
        'Tr_cam_to_velo': inverse_rigid_trans(calib['Tr_velo_to_cam'].reshape((3, 4))),
        'R0_rect': calib['R0_rect'].reshape((3, 3))
    }


def read_kitti_ground_truth(ground_truth_dir, prediction_files):
    filenames = [filename[:-4] + '.txt' for filename in prediction_files]

    scenes_labels = []
    scenes_bboxes = []

    for filename in filenames:
        labels = read_kitti_label_file(
            os.path.join(ground_truth_dir, filename))
        labels = [label for label in labels if label[0] in KITTI_CLASSES]

        scene_labels = [KITTI_CLASSES_MAP[label[0]] for label in labels]
        scene_bboxes = [map(float, label[8:15]) for label in labels]

        scenes_labels.append(scene_labels)
        scenes_bboxes.append(scene_bboxes)

    return scenes_labels, np.array(scenes_bboxes)


def read_kitti_transformations(calib_dir, prediction_files):
    filenames = [filename[:-4] + '.txt' for filename in prediction_files]

    transformations = []

    for filename in filenames:
        transformation_path = os.path.join(calib_dir, filename)
        transform = read_kitti_calibration_file(transformation_path)

        transformations.append(transform)

    return transformations


def read_pointcloud_file(file_path):
    return pypcd.PointCloud.from_path(file_path)


def read_pointcloud_files(directory, filenames):

    all_points = []
    ground_truth_labels = []
    predicted_labels = []

    for filename in tqdm.tqdm(filenames, desc='Reading point cloud files'):
        pointcloud_path = os.path.join(directory, filename)
        pointcloud = read_pointcloud_file(pointcloud_path)

        x = pointcloud.pc_data['x']
        y = pointcloud.pc_data['y']
        z = pointcloud.pc_data['z']

        points = np.c_[x, y, z]

        ground_truth_label = pointcloud.pc_data['ground_truth'].astype(
            np.int32)
        predicted_label = pointcloud.pc_data['label'].astype(np.int32)

        all_points.append(points)
        ground_truth_labels.append(ground_truth_label)
        predicted_labels.append(predicted_label)

    return np.array(all_points), np.array(ground_truth_labels), np.array(predicted_labels)


def to_lidar_frame(scenes_bboxes, transforms):

    new_scenes_bboxes = []

    for scene_bboxes, transform in zip(scenes_bboxes, transforms):
        scene_bboxes = np.array(scene_bboxes)
        points = scene_bboxes[:, 3:6]

        # Transform
        points = project_rect_to_velo(
            transform['R0_rect'], transform['Tr_cam_to_velo'], points)

        # Put back together
        scene_bboxes[:, 3:6] = points
        new_scenes_bboxes.append(scene_bboxes)

    return new_scenes_bboxes


def plot_confusion_matrix(ground_truth, prediction, classes, is_3d, cmap=plt.cm.Blues):
    classes = np.array(classes)
    new_classes = classes[unique_labels(ground_truth, prediction)]

    old_to_new_ind = {np.argwhere(classes == label)[
        0][0]: i for i, label in enumerate(new_classes)}

    cm = confusion_matrix(ground_truth, prediction)

    if is_3d:
        existing_gt_classes_ind = [old_to_new_ind[ind]
                                   for ind in unique_labels(ground_truth)]
        existing_pred_classes_ind = [old_to_new_ind[ind]
                                     for ind in unique_labels(prediction)]

        xtick_labels = new_classes[existing_pred_classes_ind]
        # ytick_labels = new_classes[existing_gt_classes_ind]
        ytick_labels = ['Unlabelled'] + KITTI_CLASSES
    else:
        xtick_labels = new_classes
        ytick_labels = new_classes

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if is_3d:
        cm = cm[existing_gt_classes_ind]
        cm = cm[:, existing_pred_classes_ind]

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # ax.figure.colorbar(im, ax=ax)
    plt.colorbar(im, cax=cax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=xtick_labels, yticklabels=ytick_labels,
           # title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def calculate_accuracy(ground_truths, predictions, is_3d, exclude_nan=False):

    nr_classes = np.max(np.hstack(ground_truths)) + 1

    # True positives, false negatives, false positives, true negatives, amount in ground truth, amount in predictions
    raw_metrics = np.zeros((nr_classes, 6))

    total_points = 0
    total_correct = 0

    for ground_truth, prediction in zip(ground_truths, predictions):
        assert len(ground_truth) == len(
            prediction), 'Different amount of points between ground truth and prediction'

        total_points += len(ground_truth)
        total_correct += np.sum(ground_truth == prediction)

        for class_index in range(nr_classes):
            # True positives
            raw_metrics[class_index, 0] += np.sum(
                (ground_truth == class_index) & (prediction == class_index))

            # False negatives
            raw_metrics[class_index, 1] += np.sum(
                (ground_truth == class_index) & (prediction != class_index))

            # False positives
            raw_metrics[class_index, 2] += np.sum(
                (ground_truth != class_index) & (prediction == class_index))

            # True negatives
            raw_metrics[class_index, 3] += np.sum(
                (ground_truth != class_index) & (prediction != class_index))

            # Amount of class in ground truth
            raw_metrics[class_index, 4] += np.sum(ground_truth == class_index)

            # Amount of class in predictions
            raw_metrics[class_index, 5] += np.sum(prediction == class_index)

    print('Accuracy: {:f}'.format(1.0 * total_correct / total_points))
    print('Total correct: {:d}'.format(total_correct))
    print('Total points: {:d}'.format(total_points))

    class_accuracy = (
        raw_metrics[:, 0] + raw_metrics[:, 3]) / np.sum(raw_metrics[:, :4], axis=1)
    class_recall = raw_metrics[:, 0] / (raw_metrics[:, 0] + raw_metrics[:, 1])
    class_precision = raw_metrics[:, 0] / \
        (raw_metrics[:, 0] + raw_metrics[:, 2])
    class_f1 = 2 * class_recall * class_precision / \
        (class_recall + class_precision)
    class_iou = raw_metrics[:, 0] / np.sum(raw_metrics[:, :3], axis=1)
    class_weight_gt = raw_metrics[:, 4] / total_points
    class_weight_pred = raw_metrics[:, 5] / total_points

    class_ids = np.arange(nr_classes)

    np.set_printoptions(suppress=True)
    print('Class ID, amount of instances, fraction of total points in ground truth, fraction of total points in predictions:')
    print(np.asarray(
        (class_ids, raw_metrics[:, 4], class_weight_gt, class_weight_pred)).T)

    print('Class ID, accuracy, recall, precision, f1-score by class, iou:')
    main_metrics = np.asarray(
        (class_ids, class_accuracy, class_recall, class_precision, class_f1, class_iou)).T
    if exclude_nan:
        main_metrics = main_metrics[~np.isnan(main_metrics).any(axis=1)]
    print(main_metrics)

    flat_gt = np.hstack(ground_truths)
    flat_pred = np.hstack(predictions)
    assert len(flat_gt) == len(
        flat_pred), 'Different number of points in ground truth and predictions'

    conf_matrix = confusion_matrix(flat_gt, flat_pred)
    print('Confusion matrix')
    print(conf_matrix)

    if is_3d:
        classes = KITTI_CLASSES_NAMES
        title = 'Confusion matrix, normalized'
        output_name = 'kitti_3d_confusion_matrix.pdf'
    else:
        classes = CITYSCAPES_CLASSES_NAMES
        title = 'Confusion matrix'
        output_name = 'pointnet_confusion_matrix.pdf'

    cm_plot = plot_confusion_matrix(flat_gt, flat_pred, classes, is_3d)
    plt.savefig(output_name, bbox_inches='tight', pad_inches=0.0)

    print('Class ID, true positives, false negatives, false positives, true negatives, amount of points in gt, amount of points in pred')
    print(np.c_[np.arange(nr_classes), raw_metrics])


def points_between(a, b):
    x = np.linspace(a[0], b[0], 15)
    y = np.linspace(a[1], b[1], 15)
    z = np.linspace(a[2], b[2], 15)

    return np.c_[x, y, z, np.zeros_like(x)]


def rotate_points_around_another_point(points, around, angle):
    rotation = R.from_euler('z', [-1.0 * angle])

    centered_points = points - around
    rotated_points = rotation.apply(centered_points)

    return rotated_points + around


def draw_3d_bboxes_for_scene(labelled_points, bboxes, filename, output_dir):
    '''
    This is for sanity checking. This method draws all of the 3D bounding boxes onto the pointcloud 
    and also marks any points inside the bounding boxes as unlabelled (label 0)
    '''

    for bbox in bboxes:
        height, width, length = bbox[:3]
        xyz = bbox[3:6]
        rotation_angle = bbox[6]

        p1 = xyz.copy()
        p1[0] -= width / 2
        p1[1] -= length / 2

        p2 = xyz.copy()
        p2[0] += width / 2
        p2[1] -= length / 2

        p3 = xyz.copy()
        p3[0] += width / 2
        p3[1] += length / 2

        p4 = xyz.copy()
        p4[0] -= width / 2
        p4[1] += length / 2

        p5 = xyz.copy()
        p5[0] -= width / 2
        p5[1] -= length / 2
        p5[2] += height

        p6 = xyz.copy()
        p6[0] += width / 2
        p6[1] -= length / 2
        p6[2] += height

        p7 = xyz.copy()
        p7[0] += width / 2
        p7[1] += length / 2
        p7[2] += height

        p8 = xyz.copy()
        p8[0] -= width / 2
        p8[1] += length / 2
        p8[2] += height

        vertices = np.asarray((p1, p2, p3, p4, p5, p6, p7, p8))
        p1, p2, p3, p4, p5, p6, p7, p8 = rotate_points_around_another_point(
            vertices, xyz, rotation_angle)

        i = p2 - p1
        ii = i.dot(i)
        j = p4 - p1
        jj = j.dot(j)
        k = p5 - p1
        kk = k.dot(k)

        v = labelled_points[:, :3] - p1

        vi = v.dot(i)
        vj = v.dot(j)
        vk = v.dot(k)

        points_inside = (0 < vi) & (vi < ii) & (
            0 < vj) & (vj < jj) & (0 < vk) & (vk < kk)
        # labelled_points[points_inside, 3] = 0

        p1p2 = points_between(p1, p2)
        p1p4 = points_between(p1, p4)
        p1p5 = points_between(p1, p5)
        p2p3 = points_between(p2, p3)
        p2p6 = points_between(p2, p6)
        p3p4 = points_between(p3, p4)
        p3p7 = points_between(p3, p7)
        p4p8 = points_between(p4, p8)
        p5p6 = points_between(p5, p6)
        p5p8 = points_between(p5, p8)
        p6p7 = points_between(p6, p7)
        p7p8 = points_between(p7, p8)

        labelled_points = np.concatenate(
            (labelled_points, p1p2, p1p4, p1p5, p2p3, p2p6, p3p4, p3p7, p4p8, p5p6, p5p8, p6p7, p7p8), axis=0)

    labelled_points = labelled_points[labelled_points[:, 2].argsort()]
    labelled_points = labelled_points[labelled_points[:, 1].argsort(
        kind='mergesort')]
    labelled_points = labelled_points[labelled_points[:, 0].argsort(
        kind='mergesort')]

    pcl = pypcd.make_xyz_label_point_cloud(labelled_points)
    new_filepath = os.path.join(output_dir, filename)
    pcl.save_pcd(new_filepath, compression='ascii')


def calculate_kitti(all_labels, all_bboxes, all_points, all_predictions, all_gt_points, scene_filenames):

    calculated_ground_truths = []
    calculated_predictions = []

    for scene_labels, scene_bboxes, scene_points, scene_predictions, scene_gt, scene_filename in zip(all_labels, all_bboxes, all_points, all_predictions, all_gt_points, scene_filenames):
        if len(scene_labels) == 0:
            continue

        # Get bounding boxes in some format that allows for testing if other points are in there
        # Discard ground truth boxes that have no points in them

        labelled_points = np.c_[scene_points, scene_predictions]
        # labelled_points = np.c_[scene_points, scene_gt]
        ground_truth_labelled_points = labelled_points.copy()

        points_to_keep = np.full(len(labelled_points), False)

        # draw_3d_bboxes_for_scene(labelled_points, scene_bboxes, scene_filename, '/home/markus/testlol')  # TODO: Perhaps read path from somewhere

        for scene_label, scene_bbox in zip(scene_labels, scene_bboxes):
            height, width, length = scene_bbox[:3]
            xyz = scene_bbox[3:6]
            rotation_angle = scene_bbox[6]

            p1 = xyz.copy()
            p1[0] -= width / 2
            p1[1] -= length / 2

            p2 = xyz.copy()
            p2[0] += width / 2
            p2[1] -= length / 2

            p4 = xyz.copy()
            p4[0] -= width / 2
            p4[1] += length / 2

            p5 = xyz.copy()
            p5[0] -= width / 2
            p5[1] -= length / 2
            p5[2] += height

            vertices = np.asarray((p1, p2, p4, p5))
            p1, p2, p4, p5 = rotate_points_around_another_point(
                vertices, xyz, rotation_angle)

            i = p2 - p1
            ii = i.dot(i)
            j = p4 - p1
            jj = j.dot(j)
            k = p5 - p1
            kk = k.dot(k)

            v = ground_truth_labelled_points[:, :3] - p1

            vi = v.dot(i)
            vj = v.dot(j)
            vk = v.dot(k)

            points_inside = (0 < vi) & (vi < ii) & (
                0 < vj) & (vj < jj) & (0 < vk) & (vk < kk)

            ground_truth_labelled_points[points_inside, 3] = scene_label

            points_to_keep = points_to_keep | points_inside

        allowed_class_predicted_points = np.isin(
            labelled_points[:, 3], KITTI_CLASSES_VALUES)

        outside_boxes = allowed_class_predicted_points & ~points_to_keep
        ground_truth_labelled_points[outside_boxes, 3] = 0

        points_to_keep = points_to_keep | allowed_class_predicted_points

        ground_truth_labels = ground_truth_labelled_points[points_to_keep, 3].astype(
            np.int32)
        predicted_labels = labelled_points[points_to_keep, 3].astype(np.int32)

        assert len(predicted_labels) == len(ground_truth_labels)

        calculated_ground_truths.append(ground_truth_labels)
        calculated_predictions.append(predicted_labels)

    return calculated_ground_truths, calculated_predictions


def get_input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'prediction', type=str,
        help='Path to directory containing output pointcloud files'
    )

    parser.add_argument(
        '--ground_truth_3d', type=str,
        help='Path to directory containing 3D ground truth in KITTI format'
    )

    parser.add_argument(
        '--calib', type=str,
        help='Path to directory containing KITTI transformations'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_input_arguments()

    prediction_dir = args.prediction
    ground_truth_3d_dir = args.ground_truth_3d
    calib_dir = args.calib

    prediction_files = os.listdir(prediction_dir)
    prediction_files = [
        filename for filename in prediction_files if filename.endswith('.pcd')]

    all_points, ground_truths, predictions = read_pointcloud_files(
        prediction_dir, prediction_files)

    kitti_labels, kitti_bboxes = read_kitti_ground_truth(
        ground_truth_3d_dir, prediction_files)

    transforms = read_kitti_transformations(calib_dir, prediction_files)
    assert len(transforms) == len(kitti_labels)

    kitti_bboxes = to_lidar_frame(kitti_bboxes, transforms)
    assert len(kitti_labels) == len(kitti_bboxes)

    print('##### Network training metrics #####')
    calculate_accuracy(ground_truths, predictions, False)

    kitti_ground_truths, kitti_predictions = calculate_kitti(
        kitti_labels, kitti_bboxes, all_points, predictions, ground_truths, prediction_files)

    print('##### 3D ground truth metrics #####')
    calculate_accuracy(kitti_ground_truths,
                       kitti_predictions, True, exclude_nan=True)
