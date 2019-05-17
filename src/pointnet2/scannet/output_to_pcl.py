import pypcd
import os
import numpy as np

all_points = np.load('kitti_val_points.npy')
all_predicted_labels = np.load('kitti_val_predictions.npy')

i = 0

for points, predicted_labels in zip(all_points, all_predicted_labels):
    assert len(points) == len(
        predicted_labels), 'Number of points is different than number of labels :('


    predicted_labels = predicted_labels.astype(np.float32)
    labelled_points = np.c_[points, predicted_labels]

    pcd = pypcd.make_xyz_label_point_cloud(labelled_points)

    pcd.save_pcd(os.path.join('output_validation',
                              '{id:d}.pcd'.format(id=i)), compression='binary_compressed')

    i += 1
