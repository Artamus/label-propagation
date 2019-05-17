import os
import pypcd
import pickle
import numpy as np


ALLOWED_SPLITS = ['train', 'val', 'trainval', 'test', 'predict']
DEFAULT_NR_POINTS = 8192
DEFAULT_VOXEL_SIZE = 1.5


def read_input_from_pcd(input_dir, split, has_labels):

    split_scene_ids_file_path = os.path.join(input_dir, split + '.txt')

    with open(split_scene_ids_file_path) as f:
        split_scene_ids = [line.rstrip() for line in f.readlines()]
    split_scene_ids = set(split_scene_ids)

    all_files = os.listdir(input_dir)
    input_files = [
        file_name for file_name in all_files if file_name.endswith('.pcd') and file_name[:-4] in split_scene_ids]

    scene_ids = []
    scene_points_list = []
    semantic_labels_list = []

    for input_file in input_files:
        scene_id = input_file[:-4]
        scene_ids.append(scene_id)

        input_file_path = os.path.join(input_dir, input_file)

        pcd = pypcd.PointCloud.from_path(input_file_path)

        x = pcd.pc_data['x']
        y = pcd.pc_data['y']
        z = pcd.pc_data['z']

        scene_points_list.append(np.c_[x, y, z])

        if has_labels:
            labels = pcd.pc_data['label'].astype(np.int32)
            semantic_labels_list.append(labels)

    return scene_ids, np.array(scene_points_list), np.array(semantic_labels_list)


def read_input_from_pickle(input_dir, pickle_base_name, split):

    data_filename = '{:s}_{:s}.pickle'.format(pickle_base_name, split)
    data_path = os.path.join(input_dir, data_filename)

    with open(data_path, 'rb') as fp:
        scene_points_list = pickle.load(fp)
        semantic_labels_list = pickle.load(fp)

    return None, np.array(scene_points_list), np.array(semantic_labels_list)


def get_label_weights(has_labels, nr_classes, labels):

    if not has_labels:
        return np.ones(nr_classes)

    label_weights = np.zeros(nr_classes)

    # TODO: Verify correctness
    for seg in labels:
        tmp, _ = np.histogram(seg, range(nr_classes + 1))
        label_weights += tmp
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        label_weights = 1 / np.log(1.2 + label_weights)

    return label_weights


class BaseDataset(object):
    def __init__(self, input_dir, npoints, voxel_size, split='train', pickle_base_name=None):
        assert split in ALLOWED_SPLITS, 'Provided split {:s} is not allowed'.format(
            split)

        self.npoints = npoints

        self.voxel_size = voxel_size
        self.voxel_big_delta = voxel_size * 0.2 / 1.5

        self.input_dir = input_dir
        self.split = split

        has_labels = self.split != 'predict'  # Prediction does not have labels

        if pickle_base_name:
            self.scene_ids, self.scene_points_list, self.semantic_labels_list = read_input_from_pickle(
                self.input_dir, pickle_base_name, split)
        else:
            self.scene_ids, self.scene_points_list, self.semantic_labels_list = read_input_from_pcd(
                self.input_dir, self.split, has_labels)

        # TODO: Breaks if there are actually no labels BTW
        # self.nr_classes = NR_CLASSES  # Use this if the below line gives wrong numbers
        self.nr_classes = np.max(
            np.hstack(self.semantic_labels_list)) + 1  # Also count class 0
        print('Number of classes detected: {:d}'.format(self.nr_classes))

        self.label_weights = get_label_weights(
            has_labels, self.nr_classes, self.semantic_labels_list)

    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        raise NotImplementedError('Do not use this class')


class Dataset(BaseDataset):
    def __init__(self, input_dir, npoints=DEFAULT_NR_POINTS, voxel_size=DEFAULT_VOXEL_SIZE, split='train', pickle_base_name=None):
        super(Dataset, self).__init__(input_dir, npoints,
                                      voxel_size, split, pickle_base_name)

        self.voxel_small_delta = voxel_size * 0.01 / 1.5

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)

        coordmax = np.max(point_set, axis=0)
        coordmin = np.min(point_set, axis=0)

        isvalid = False

        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg), 1)[0], :]

            curmin = curcenter - [0.5 * self.voxel_size,
                                  0.5 * self.voxel_size, self.voxel_size]
            curmax = curcenter + [0.5 * self.voxel_size,
                                  0.5 * self.voxel_size, self.voxel_size]

            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]

            curchoice = np.sum((point_set >= (curmin - self.voxel_big_delta))
                               * (point_set <= (curmax + self.voxel_big_delta)), axis=1) == 3
            cur_point_set = point_set[curchoice, :]
            cur_semantic_seg = semantic_seg[curchoice]

            if len(cur_semantic_seg) == 0:
                continue

            mask = np.sum((cur_point_set >= (curmin - self.voxel_small_delta)) *
                          (cur_point_set <= (curmax + self.voxel_small_delta)), axis=1) == 3

            # Get voxel id-s
            vidx = np.ceil(
                (cur_point_set[mask, :] - curmin) / (curmax - curmin) * [31.0, 31.0, 62.0])
            vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 +
                             vidx[:, 1] * 62.0 + vidx[:, 2])

            # Not 0 labels must be at least 70%
            # Must have something in >= 2% subvoxels
            isvalid = np.sum(cur_semantic_seg > 0) / \
                len(cur_semantic_seg) >= 0.7 and len(
                    vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            if isvalid:
                break

        choice = np.random.choice(
            len(cur_semantic_seg), self.npoints, replace=True)

        point_set = cur_point_set[choice, :]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]

        sample_weight = self.label_weights[semantic_seg]
        sample_weight *= mask

        return point_set, semantic_seg, sample_weight


class DatasetWholeScene(BaseDataset):
    def __init__(self, input_dir, npoints=DEFAULT_NR_POINTS, voxel_size=DEFAULT_VOXEL_SIZE, split='train', pickle_base_name=None):
        super(DatasetWholeScene, self).__init__(
            input_dir, npoints, voxel_size, split, pickle_base_name)

        self.voxel_small_delta = voxel_size * 0.001 / 1.5

    def __getitem__(self, index):
        '''
        Fetches full scene at index, then divides points into voxels and for each voxel samples a predetermined amount of points (self.npoints)
        with replacement. Also does some other magic. 

        In the data used by the paper the average density of points was ~6500 per voxel.
        '''

        if self.scene_ids:
            scene_id = self.scene_ids[index]
        else:
            scene_id = None

        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)

        coordmax = np.max(point_set_ini, axis=0)
        coordmin = np.min(point_set_ini, axis=0)

        nsubvolume_x = np.ceil(
            (coordmax[0] - coordmin[0]) / self.voxel_size).astype(np.int32)
        nsubvolume_y = np.ceil(
            (coordmax[1] - coordmin[1]) / self.voxel_size).astype(np.int32)

        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + \
                    [i * self.voxel_size, j * self.voxel_size, 0]
                curmax = coordmin + [(i + 1) * self.voxel_size, (j + 1) * self.voxel_size,
                                     coordmax[2] - coordmin[2]]

                # Accepting those points that satisfy the conditions in all three coordinates
                curchoice = np.sum((point_set_ini >= (curmin - self.voxel_big_delta))
                                   * (point_set_ini <= (curmax + self.voxel_big_delta)), axis=1) == 3

                cur_point_set = point_set_ini[curchoice, :]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg) == 0:
                    continue

                mask = np.sum((cur_point_set >= (curmin - self.voxel_small_delta))
                              * (cur_point_set <= (curmax + self.voxel_small_delta)), axis=1) == 3
                choice = np.random.choice(
                    len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]

                # Check that percentage of points from smaller box is at least 1% of the sampled points
                if sum(mask) / float(len(mask)) < 0.01:
                    continue

                sample_weight = self.label_weights[semantic_seg]
                sample_weight *= mask  # N

                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN

        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)

        return scene_id, point_sets, semantic_segs, sample_weights
