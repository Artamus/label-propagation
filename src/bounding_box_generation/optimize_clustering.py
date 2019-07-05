import argparse
import os
import pypcd
import sklearn.cluster
import itertools
import tqdm
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from kitti_bounding_box_provider import get_bounding_boxes, count_points_bboxes_matching, transform_points_velo_to_rect, get_2d_bbox_from_3d_alt
from bounding_box_creator import minimum_bounding_rectangle
from scipy.spatial.transform import Rotation as R
from MinimumBoundingBox import MinimumBoundingBox
from visualization import visualize_clustering, visualize_bounding_box_2d, visualize_pointcloud_with_bounding_boxes_KITTI, visualize_pointcloud_with_bounding_boxes

max_cluster_size = 0

def read_pointcloud(pointcloud_file_path):
    pointcloud = pypcd.PointCloud.from_path(pointcloud_file_path)

    x = pointcloud.pc_data['x']
    y = pointcloud.pc_data['y']
    z = pointcloud.pc_data['z']
    labels = pointcloud.pc_data['label']

    return np.c_[x, y, z], labels


def filter_pedestrian_points(points, labels):
    filter_mask = labels == 12.  # Pedestrian is class 12
    return points[filter_mask]


def get_points(pointcloud_file_path):

    points, labels = read_pointcloud(pointcloud_file_path)
    filtered_points = filter_pedestrian_points(points, labels)

    return filtered_points


def cluster_points(points, min_cluster_size, min_samples, eps, method='dbscan'):
    if len(points) == 0:
        return []
    
    if len(points) < min_samples:
        return np.array([-1] * len(points))

    if method == 'dbscan':
        return sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    elif method == 'hdbscan':
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(points)

    raise RuntimeError('Invalid clustering method chosen')


def read_bbox_data(bbox_file_path, calibration_file_path):
    # TODO: Fix data structure

    allowed_classes = {'Pedestrian', 'Person_sitting'}
    return get_bounding_boxes(bbox_file_path, allowed_classes, calibration_file_path)


def get_cluster_centers(points, point_cluster_labels):
    global max_cluster_size
    unique_labels = set(point_cluster_labels)
    unique_labels.discard(-1)

    cluster_centers = list()
    for cluster_index in unique_labels:
        current_cluster_ind = point_cluster_labels == cluster_index
        current_cluster_points = points[current_cluster_ind]

        if len(current_cluster_points) > max_cluster_size:
            max_cluster_size = len(current_cluster_points)

        cluster_centers.append(current_cluster_points.mean(axis=0))

    return np.array(cluster_centers)


def generate_bounding_boxes_for_scene(points, cluster_labels, calibration_file_path):
    # Generate bounding boxes for the clusters
    # Use those to get metrics using the original bounding boxes, probably use the same metric as KITTI
    # Some viable methods
    # 1) Simply min/max for each axis and get axis-aligned bounding boxes
    # 2) Find an average bounding box from either training or validation data and fit this to either the center of the cluster
    #    or find some other points on the edge to fit it with
    # 3) Use MVBB to calculate orientated bounding boxes

    # print('Points in scene: {:s}'.format(points.shape))

    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)

    bounding_boxes = list()
    bounding_box_corners = np.zeros((len(unique_labels), 8, 3))
    scores = list()

    for ind, cluster_label in enumerate(unique_labels):
        cluster_points = points[cluster_labels == cluster_label]
        score = 1.0 * len(cluster_points) / 1200  # 780
        score = np.clip(score, 0.0, 1.0)
        scores.append(score)
        # print('Points in current cluster: {:s}'.format(cluster_points.shape))

        min_bbox = MinimumBoundingBox(cluster_points[:, :2])

        min_z, max_z = cluster_points.min(axis=0)[2], cluster_points.max(axis=0)[2]

        corners_2d = np.array(list(min_bbox.corner_points))
        bottom_corners = np.c_[corners_2d, [min_z] * 4]
        top_corners = np.c_[corners_2d, [max_z] * 4]
        bounding_box_corners[ind] = np.concatenate((bottom_corners, top_corners), axis=0)

        len_z = max_z - min_z

        # TODO: These might be swapped, y might be parallel instead
        len_y = min_bbox.length_orthogonal
        len_x = min_bbox.length_parallel
        angle = min_bbox.unit_vector_angle

        # len_z, len_x, len_y, x, y, z, angle
        # height, width, length, x, y, z, angle in KITTI format
        bounding_boxes.append([len_z, len_x, len_y, min_bbox.rectangle_center[0], min_bbox.rectangle_center[1], min_z, angle])

    bounding_boxes = np.array(bounding_boxes)
    # print(bounding_boxes)

    # Transform centerpoints of bboxes
    bounding_boxes[:, 3:6] = transform_points_velo_to_rect(bounding_boxes[:, 3:6], calibration_file_path)
    # print(bounding_boxes)

    # boxes_2d = get_2d_bbox_from_3d(bounding_boxes, calibration_file_path)
    boxes_2d = get_2d_bbox_from_3d_alt(bounding_box_corners, calibration_file_path)

    labels = list()
    for ind in range(len(bounding_boxes)):
        # TODO: Third value, alpha, should be computed somehow probably
        # TODO: Calculate the score, if possible

        labels.append(['Pedestrian', -1, -1, bounding_boxes[ind, -1]] + boxes_2d[ind].tolist() + bounding_boxes[ind].tolist() + [scores[ind]])
    
    return labels, bounding_box_corners


def save_point_estimates(output_file_path, point_estimates):
    if len(point_estimates) == 0:
        return

    output_point_estimates = ['{:.5f} {:.5f}'.format(estimate[0], estimate[1]) for estimate in point_estimates]
    raw_output = '\n'.join(output_point_estimates)

    with open(output_file_path, 'w') as f:
        f.write(raw_output)


def save_labels(output_file_path, labels):
    if len(labels) == 0:
        return

    formatted_labels = list()
    for label in labels:
        formatted_labels.append([label[0], str(label[1]), str(label[2])] + ['{:.2f}'.format(elements) for elements in label[3:]])
    output_labels = [' '.join(label) for label in formatted_labels]
    raw_output = '\n'.join(output_labels)

    with open(output_file_path, 'w') as f:
        f.write(raw_output)


def save_box_corners(output_file_path, box_corners):
    if len(box_corners) == 0:
        return

    output_corners = []
    for corners in box_corners:
        output_corners.append(' '.join(corners.flatten()))
    
    raw_output = '\n'.join(output_corners)
    
    with open(output_file_path, 'w') as f:
        f.write(raw_output)


def save_empty_labels(output_file_path):
    with open(output_file_path, 'w') as f:
        f.write('')


def input_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'points', type=str,
        help='Path to directory containing labelled pointcloud files'
    )

    parser.add_argument(
        'data', type=str,
        help='Path to directory containing supplementary data, this folder should contain labels, calibrations, etc.'
    )

    parser.add_argument(
        'output', type=str,
        help='Path to directory where output will be written'
    )

    parser.add_argument(
        '-c', '--minclust', type=int, default=100,
        help='Min cluster size for HDBSCAN'
    )

    parser.add_argument(
        '-p', '--minpts', type=int, default=50,
        help='Min points for clustering'
    )

    parser.add_argument(
        '-e', '--eps', type=float, default=0.15,
        help='Epsilon for clustering'
    )

    parser.add_argument(
        '-m', '--method',
        choices=['dbscan', 'hdbscan', 'optics'],
        default='dbscan',
        help='Clustering algorithm to use'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = input_arguments()

    # For each scene
    # - Clustering (outputs cluster centerpoints to some output folder, each file is for a scene), returns cluster-labels for every input point
    #   clustering is done in 2D
    # - Optional metrics calculation which doubles as visualization, or the two can be separate.

    # For each cluster in the scene:
    # - Get bounding rectangle in 2 dimensions and build 3D bounding box, also create 2D image plane bounding box based on this
    # TODO: How to validate this?
    # Take image and draw the rectangle onto it
    # Draw 3D bounding boxes onto the point cloud, possibly in a separate script, can live in a separate file and just be called.

    # Setup
    data_path = args.data
    min_clust = args.minclust
    min_samples = args.minpts
    eps = args.eps
    print('Minimum samples for clustering: {:d}, epsilon: {:2f}, min cluster size: {:d}'.format(min_samples, eps, min_clust))
    metrics = False
    visualization = False
    debug = False

    total_tp = 0
    total_fp = 0
    total_fn = 0

    max_calculations = 5
    nr_calculations = 0

    # Create appropriate directories
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    point_estimate_output_dir = os.path.join(args.output, 'point_estimates')
    if not debug and not os.path.isdir(point_estimate_output_dir):
        os.mkdir(point_estimate_output_dir)

    clustering_visualization_dir = os.path.join(args.output, 'clustering_visualization')
    if visualization and not os.path.isdir(clustering_visualization_dir):
        os.mkdir(clustering_visualization_dir)

    labels_output_dir = os.path.join(args.output, 'labels')
    if not debug and not os.path.isdir(labels_output_dir):
        os.mkdir(labels_output_dir)

    labels_2d_visualization_dir = os.path.join(args.output, 'bounding_box_2d_visualization')
    if visualization and not os.path.isdir(labels_2d_visualization_dir):
        os.mkdir(labels_2d_visualization_dir)

    labels_3d_visualization_dir = os.path.join(args.output, 'bounding_box_visualization')
    if visualization and not os.path.isdir(labels_3d_visualization_dir):
        os.mkdir(labels_3d_visualization_dir)

    labels_corners_visualization_dir = os.path.join(args.output, 'corners_visualization')
    if visualization and not os.path.isdir(labels_corners_visualization_dir):
        os.mkdir(labels_corners_visualization_dir)

    # Get list of point files in directory
    pointcloud_files = sorted(os.listdir(args.points))
    pointcloud_file_paths = [args.points + '/' +
                             filename for filename in pointcloud_files if filename.endswith('.pcd')]

    for file_path in tqdm.tqdm(pointcloud_file_paths):
        scene_id = os.path.basename(file_path).split('.')[0]

        points = get_points(file_path)
        if len(points) == 0:
            # Write empty results
            if not debug:
                save_empty_labels(os.path.join(labels_output_dir, scene_id + '.txt'))

            continue

        point_cluster_labels = cluster_points(points[:, :2], min_clust, min_samples, eps, method=args.method)  # Cluster on x and y coordinates, omitting height

        unique_labels = set(point_cluster_labels)
        unique_labels.discard(-1)
        nr_clusters = len(unique_labels)

        if debug:
            print('Number of clusters in scene {:s}: {:d}'.format(scene_id, nr_clusters))

        cluster_centers = get_cluster_centers(points[:, :2], point_cluster_labels)

        if not debug:
            point_estimate_output_file_path = os.path.join(point_estimate_output_dir, scene_id + '.txt')
            save_point_estimates(point_estimate_output_file_path, cluster_centers)

        if metrics:
            # Calculate the metrics how well the cluster centers match 3D bounding boxes
            gt_bounding_boxes = read_bbox_data(
                os.path.join(data_path, 'label_2', scene_id + '.txt'),
                os.path.join(data_path, 'calib', scene_id + '.txt')
            )
            tp, fp, fn = count_points_bboxes_matching(cluster_centers, gt_bounding_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        if visualization and nr_clusters > 0:
            cluster_visualization_output_path = os.path.join(clustering_visualization_dir, scene_id + '.png')
            visualize_clustering(points, point_cluster_labels, cluster_visualization_output_path)

        # TODO: More specific check
        if nr_clusters == 0:  # No 3D bounding boxes to create when there are no clusters

            # Write empty results
            if not debug:
                save_empty_labels(os.path.join(labels_output_dir, scene_id + '.txt'))

            continue

        calibration_path = os.path.join(data_path, 'calib', scene_id + '.txt')
        bounding_boxes, box_corners = generate_bounding_boxes_for_scene(points, point_cluster_labels, calibration_path)

        if not debug:
            labels_output_file_path = os.path.join(labels_output_dir, scene_id + '.txt')
            save_labels(labels_output_file_path, bounding_boxes)

        if visualization:
            # Save the 2D bounding box onto the image and save it
            raw_image_path = os.path.join(args.data, 'image_2', scene_id + '.png')
            boxes_2d = np.array(bounding_boxes)[:, 4:8]
            bbox_2d_visualization_output_path = os.path.join(labels_2d_visualization_dir, scene_id + '.png')
            visualize_bounding_box_2d(raw_image_path, boxes_2d, bbox_2d_visualization_output_path)

            bbox_visualization_output_path = os.path.join(labels_3d_visualization_dir, scene_id + '.pcd')
            all_points, labels = read_pointcloud(file_path)
            labelled_points = np.c_[all_points, labels]
            # visualize_pointcloud_with_bounding_boxes_KITTI(labelled_points, bounding_boxes[:, 8:].astype(np.float32), bbox_visualization_output_path)
            # Midpoints are already transformed by this point, results unreliable

            corners_visualization_output_path = os.path.join(labels_corners_visualization_dir, scene_id + '.pcd')
            visualize_pointcloud_with_bounding_boxes(labelled_points, box_corners, corners_visualization_output_path)

        if debug:
            nr_calculations += 1

            if nr_calculations >= max_calculations:
                print("Limit reached, breaking")
                break

    if not debug:
        used_scenes_path = os.path.join(args.output, 'split.txt')
        scene_ids = [os.path.basename(file_path).split('.')[0] for file_path in pointcloud_file_paths]
        raw_output = '\n'.join(scene_ids)

        with open(used_scenes_path, 'w') as f:
            f.write(raw_output)

    if metrics:
        recall = 1.0 * total_tp / (total_tp + total_fn)
        precision = 1.0 * total_tp / (total_tp + total_fp)
        f1_score = 2.0 * total_tp / (2 * total_tp + total_fp + total_fn)
        print('Params - minsamples {:d} eps {:.2f}'.format(min_samples, eps))
        print('Recall: {:.3f}, Precision: {:.3f}, F1-score: {:.3f}'.format(recall, precision, f1_score))

    print(max_cluster_size)
