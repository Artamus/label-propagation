import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pypcd
from PIL import Image
from scipy.spatial.transform import Rotation as R


def visualize_clustering(points, cluster_labels, output_file_path):
    """Visualizing the point cloud in the x-y coordinates
    on a scatterplot with the point colour corresponding
    to the cluster the point was assigned to.

    Code is a modified version of the sklearn example for DBSCAN.

    Arguments:
        points {np.ndarray} -- 2D or 3D points, xyz coordinates, x forward, y left, z upward
        cluster_labels {list[int]} -- label per pointcloud point
        output_file_path {string} -- the path to save the image to
    """
    vis_points = points[:, [1, 0]]
    vis_points[:, 0] *= -1

    unique_labels = set(cluster_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black used for noise.

        class_member_mask = (cluster_labels == k)

        xy = vis_points[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.ylim(0, 45)
    plt.xlim(-20, 20)

    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.savefig(output_file_path, pad_inches=0)
    plt.clf()


def visualize_bounding_box_2d(image_path, box_2d, output_file_path):
    image = np.array(Image.open(image_path))

    fig, ax = plt.subplots(1, frameon=False)
    ax.imshow(image)

    for box in box_2d:
        left_bottom = (int(float(box[0])), int(float(box[1])))
        width = int(float(box[2]) - float(box[0]))
        height = int(float(box[3]) - float(box[1]))
        rect = patches.Rectangle(left_bottom, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.set_axis_off()

    fig.savefig(output_file_path, pad_inches=0, dpi=800, bbox_inches='tight')
    fig.clear()
    plt.close(fig)


def rotate_points_around_another_point(points, around, angle):
    rotation = R.from_euler('z', [-1.0 * angle])

    centered_points = points - around
    rotated_points = rotation.apply(centered_points)

    return rotated_points + around


def points_between(a, b):
    x = np.linspace(a[0], b[0], 15)
    y = np.linspace(a[1], b[1], 15)
    z = np.linspace(a[2], b[2], 15)

    return np.c_[x, y, z, np.zeros_like(x)]


def visualize_pointcloud_with_bounding_boxes_KITTI(labelled_points, bboxes, output_file_path):
    """
    This is for sanity checking. This method draws all of the 3D bounding boxes onto the pointcloud 
    and also marks any points inside the bounding boxes as unlabelled (label 0)
    """

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
    pcl.save_pcd(output_file_path, compression='ascii')


def visualize_pointcloud_with_bounding_boxes(labelled_points, bboxes, output_file_path):
    for bbox in bboxes:
        p1, p3, p2, p4, p5, p7, p6, p8 = bbox

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
    pcl.save_pcd(output_file_path, compression='ascii')
