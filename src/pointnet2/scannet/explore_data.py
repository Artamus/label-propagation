import pickle
import pypcd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

with open('/home/markus/thesis/src/pointnet2/data/scannet_data_pointnet2/scannet_train.pickle', 'r') as f:
    scenes_points = pickle.load(f)
    scenes_labels = pickle.load(f)

sizes = []

for index, points in enumerate(scenes_points):

    scene_min = points.min(axis=0)
    scene_max = points.max(axis=0)

    scene_size = scene_max - scene_min
    sizes.append(scene_size)

    # new_points = np.c_[points, scenes_labels[index]]

    # pcd_file = pypcd.make_xyz_label_point_cloud(new_points)
    # pcd_file.save_pcd('/home/markus/thesis/src/pointnet2/scannet/data_explore/{id:03d}.pcd'.format(id=index), compression='binary_compressed')

sizes = np.array(sizes)
print(sizes.mean(axis=0))
print(np.median(sizes, axis=0))
print(sizes.max(axis=0))

fig = pl.hist(sizes[:, 0], normed=0)
pl.title('Mean')
pl.xlabel("Value")
pl.ylabel("Frequency")
pl.savefig("x-kitti.png")
pl.clf()

fig = pl.hist(sizes[:, 1], normed=0)
pl.title('Mean')
pl.xlabel("Value")
pl.ylabel("Frequency")
pl.savefig("y-kitti.png")
pl.clf()

fig = pl.hist(sizes[:, 2], normed=0)
pl.title('Mean')
pl.xlabel("Value")
pl.ylabel("Frequency")
pl.savefig("z-kitti.png")
