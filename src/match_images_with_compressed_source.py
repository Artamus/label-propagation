import tempfile
import os
import cv2
import argparse
import numpy as np
from zipfile import ZipFile
from shutil import copyfile, rmtree
from scipy.spatial import distance


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--source', type=str,
        default='/home/markus/KITTI/semantic/training/image_2',
        help='Source directory for images you are looking matches for'
    )

    parser.add_argument(
        '-i', '--input', type=str,
        default='/home/markus/KITTI/raw_data_synced',
        help='Input directory containing KITTI raw data zips'
    )

    parser.add_argument(
        '-b', '--batch', type=int,
        default=500,
        help='Batch size of images to read at once for distance calculations'
    )

    parser.add_argument(
        '-o', '--output', type=str,
        default='/home/markus/KITTI/semantic/training/',
        help='Output directory that should contain the lidar and calibration data directories after running this'
    )

    return parser.parse_args()


def read_images(images_path, nr_images=None, offset=0):
    image_filenames = sorted(os.listdir(images_path))
    images = []

    if nr_images:
        start = offset
        end = offset + nr_images
        image_filenames = image_filenames[start:end]

    for image_filename in image_filenames:
        image_path = os.path.join(images_path, image_filename)
        image = cv2.imread(image_path).flatten()[:1358640]
        images.append(image)

    return image_filenames, np.array(images)


def get_target_zips(input_path):
    zip_files_path = os.path.join(input_path, '*.zip')
    return sorted(os.listdir(zip_files_path))


def unpack_zip(zip_path, target_path):
    with ZipFile(zip_path, 'r') as zip_object:
        zip_object.extractall(target_path)


def copy_velodyne(source_image_name, zip_image_name, zip_velodyne_path, output_dir):
    zip_velodyne_filename = zip_image_name[:-4] + '.bin'
    velodyne_path = os.path.join(
        zip_velodyne_path, zip_velodyne_filename)

    target_velodyne_filename = source_image_name[:-4] + '.bin'
    target_velodyne_path = os.path.join(
        output_dir, 'velodyne', target_velodyne_filename)

    copyfile(velodyne_path, target_velodyne_path)


def calculate_distances_and_find_matches(source_image_names, source_images, zip_image_names, zip_images, zip_velodyne_path, extraction_name, output_dir):

    distances = distance.cdist(
        source_images, zip_images, 'sqeuclidean')
    current_matches = np.where(distances == 0)

    matches = []
    for source_frame_id, zip_frame_id in zip(current_matches[0], current_matches[1]):
        source_image_name = source_image_names[source_frame_id]
        zip_image_name = zip_image_names[zip_frame_id]

        copy_velodyne(source_image_name, zip_image_name,
                      zip_velodyne_path, output_dir)

        match = (source_image_name, extraction_name, zip_image_name)
        matches.append(match)

    return matches


# TODO: Add copying for calibration matrices as well
def match_images(source_images_dir, target_images_dir, batch_size, output_dir):

    source_image_names, source_images = read_images(source_images_dir)
    target_zips = get_target_zips(target_images_dir)

    total_nr_source_images = len(source_image_names)

    matches_file_path = os.path.join(output_dir, 'matches.txt')
    if os.path.isfile(matches_file_path):
        all_matches = []
        with open(matches_file_path, 'r') as f:
            for line in f:
                line = line.rstrip().split(' ')
                all_matches.append((line[0], line[1], line[2]))

    else:
        all_matches = []

    if len(all_matches) > 0:
        processed_zips = ['{name}.zip'.format(name=x[1]) for x in all_matches]
        target_zips = [
            name for name in target_zips if name not in processed_zips]

        processed_source_image_names = [match[0] for match in all_matches]

        # Filter source
        def filter_sources(x): return x[0] not in processed_source_image_names
        filtered_sources = list(
            filter(filter_sources, zip(source_image_names, source_images)))

        source_image_names = [x[0] for x in filtered_sources]
        source_images = np.array([x[1] for x in filtered_sources])

    with tempfile.TemporaryDirectory() as tmp_dir:

        for i, target_zip in enumerate(target_zips):
            if len(all_matches) == total_nr_source_images:
                break

            print('Working on archive {index:d}/{number_archives:d}: {name:s}'.format(
                index=(i+1), number_archives=len(target_zips), name=target_zip))

            target_zip_path = os.path.join(target_images_dir, target_zip)
            unpack_zip(target_zip_path, tmp_dir)

            zip_date = target_zip[:10]
            extraction_name = target_zip[:-4]
            zip_images_path = os.path.join(
                tmp_dir, zip_date, extraction_name, 'image_02', 'data')

            zip_velodyne_path = os.path.join(
                tmp_dir, zip_date, extraction_name, 'velodyne_points', 'data')

            if batch_size > 0:

                batch_offset = 0
                while 1:
                    zip_image_names, zip_images = read_images(
                        zip_images_path, batch_size, batch_offset)

                    batch_matches = calculate_distances_and_find_matches(
                        source_image_names, source_images, zip_image_names, zip_images, zip_velodyne_path, extraction_name, output_dir)
                    all_matches.extend(batch_matches)

                    if zip_images.shape[0] < batch_size:
                        break

                    batch_offset += batch_size
            else:
                zip_image_names, zip_images = read_images(zip_images_path)
                matches = calculate_distances_and_find_matches(
                    source_image_names, source_images, zip_image_names, zip_images, zip_velodyne_path, extraction_name, output_dir)

                all_matches.extend(matches)

            # Remove unpacked folder
            unzipped_folder_path = os.path.join(
                tmp_dir, zip_date, extraction_name)
            rmtree(unzipped_folder_path)

    print('Matches contained {:d} elements'.format(len(all_matches)))

    all_matches = sorted(all_matches, key=lambda x: x[0])
    with open(matches_file_path, 'w') as f:
        for match in all_matches:
            f.write('{semantic_image} {extract} {extract_image}\n'.format(
                semantic_image=match[0], extract=match[1], extract_image=match[2]))

    matched_image_names = [x[0] for x in all_matches]
    sources_not_matched = [
        name for name in source_image_names if name not in matched_image_names]
    print('Source images not matched: ', sources_not_matched)


if __name__ == '__main__':
    args = parse_arguments()
    SOURCE_DIRECTORY = args.source
    INPUT_DIRECTORY = args.input
    BATCH_SIZE = args.batch
    OUTPUT_DIRECTORY = args.output

    if not os.path.isdir(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    velodyne_path = os.path.join(OUTPUT_DIRECTORY, 'velodyne')
    if not os.path.isdir(velodyne_path):
        os.mkdir(velodyne_path)

    match_images(SOURCE_DIRECTORY, INPUT_DIRECTORY,
                 BATCH_SIZE, OUTPUT_DIRECTORY)
