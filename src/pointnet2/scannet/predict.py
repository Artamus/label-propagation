import argparse
import importlib
import os
import sys
import pypcd
import tqdm
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import dataset


def get_last_best_model_path(log_dir):
    files = os.listdir(log_dir)
    files = sorted([filename for filename in files if filename.startswith('best_model_epoch_')])

    last_file = files[-1]
    last_file_base_name = last_file.split('.ckpt')[0]

    return os.path.join(log_dir, last_file_base_name + '.ckpt')


def run_inference(test_data, model, gpu_index, nr_classes, batch_size, num_point, output_dir, log_dir):

    with tf.device('/gpu:' + str(gpu_index)):
        pointclouds_pl, labels_pl, smpws_pl = model.placeholder_inputs(
            batch_size, num_point)

        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred, end_points = model.get_model(
            pointclouds_pl, is_training_pl, nr_classes)

        loss = model.get_loss(pred, labels_pl, smpws_pl)
        tf.summary.scalar('loss', loss)

        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    checkpoint_path = get_last_best_model_path(log_dir)
    # checkpoint_path = os.path.join(log_dir, 'model.ckpt')
    print('##### Restoring from checkpoint %s' % checkpoint_path)
    saver.restore(sess, checkpoint_path)

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'smpws_pl': smpws_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    # Go over batches of the test data
    nr_scenes = len(test_data)

    is_training = False

    total_pred = []
    total_points = []

    for scene_id in tqdm.trange(nr_scenes):

        scene_id, scene_points, scene_labels, scene_sampleweights = test_data[scene_id]

        nr_batches = np.ceil(1.0 * len(scene_points)
                             / batch_size).astype(np.int32)

        predictions = []
        labels = []

        for batch_index in range(nr_batches):

            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size

            batch_points = scene_points[start:end, ...]
            batch_labels = scene_labels[start:end, ...]
            batch_sampleweights = scene_sampleweights[start:end, ...]

            feed_dict = {ops['pointclouds_pl']: batch_points,
                         ops['labels_pl']: batch_labels,
                         ops['smpws_pl']: batch_sampleweights,
                         ops['is_training_pl']: is_training}

            loss_val, pred_val = sess.run(
                [ops['loss'], ops['pred']], feed_dict=feed_dict)

            prediction = np.argmax(pred_val, axis=2)
            predictions.extend(prediction)
            labels.extend(batch_labels)

        predictions = np.array(predictions)
        predictions = predictions.reshape(-1)

        labels = np.array(labels)
        labels = labels.reshape(-1)

        # Unique-fy the points, because I do not want to touch the method by which the data is sampled,
        # I will instead just reconstruct the initial state

        all_scene_points = scene_points.reshape(-1, 3)

        unique_scene_points, unique_indices = np.unique(
            all_scene_points, return_index=True, axis=0)
        unique_predictions = predictions[unique_indices]
        unique_labels = labels[unique_indices]

        # Save result
        save_result(output_dir, scene_id,
                    unique_scene_points, unique_predictions, unique_labels)

        # total_points.append(unique_scene_points)
        # total_pred.append(unique_predictions)

    # total_points = np.array(total_points)
    # total_pred = np.array(total_pred)


def save_result(output_dir, scene_id, points, predictions, ground_truth_labels):
    output_file_path = os.path.join(output_dir, scene_id + '.pcd')

    predictions = predictions.astype(np.float32)
    output_data = np.c_[points, predictions]

    pcd = pypcd.make_xyz_label_point_cloud(output_data)

    ground_truth_metadata = {
        'fields': ['ground_truth'],
        'count': [1],
        'size': [4],
        'type': ['F']
    }

    ground_truth_dt = np.dtype([('ground_truth', np.float32)])
    ground_truth_pc_data = np.rec.fromarrays([ground_truth_labels], dtype=ground_truth_dt)

    pcd = pypcd.add_fields(pcd, ground_truth_metadata, ground_truth_pc_data)

    pcd.save_pcd(output_file_path, compression='ascii')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str,
                        help='Directory from where to read input files')

    parser.add_argument('output', type=str,
                        help='Directory where to put output files')

    parser.add_argument('split', type=str,
                        help='Which split to run inference on')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')

    parser.add_argument('--model', default='pointnet2_sem_seg',
                        help='Model name. [default: pointnet2_sem_seg]')

    parser.add_argument('--num_point', type=int, default=8192,
                        help='Point Number [default: 8192]')

    parser.add_argument('--vox_size', type=float, default=1.5,
                        help='Voxel size [default: 1.5]')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size during training [default: 32]')

    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory with the checkpoints for the trained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    gpu_index = args.gpu
    model_name = args.model
    nr_points = args.num_point
    voxel_size = args.vox_size
    batch_size = args.batch_size
    input_dir = args.input
    output_dir = args.output
    log_dir = args.log_dir
    split = args.split

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Read in model
    model = importlib.import_module(model_name)

    data = dataset.DatasetWholeScene(
        input_dir=input_dir, npoints=nr_points, voxel_size=voxel_size, split=split)

    nr_classes = data.nr_classes

    with tf.Graph().as_default():
        run_inference(data, model, gpu_index,
                      nr_classes, batch_size, nr_points, output_dir, log_dir)
