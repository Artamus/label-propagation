import re
import os
import argparse
import urllib.request


BASE_URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input', type=str,
        required=True,
        help='Path to file containing raw text, from which the filenames can be extracted via regex'
    )

    parser.add_argument(
        '-o', '--output', type=str,
        default='/home/markus/KITTI/raw_data_synced',
        help='Output directory where the zip files will be downloaded'
    )

    parser.add_argument(
        '-s', '--size', type=int,
        default=2048,
        help='Maximum size in megabytes'
    )

    return parser.parse_args()


def extract_file_names(input_file_path):
    with open(input_file_path, 'r') as f:
        data = f.read()

    return re.findall('\d{4}_\d{2}_\d{2}_drive_\d{4}', data, re.DOTALL)


def download_sensor_data(file_names, output_dir):
    existing_files = os.listdir(output_dir)

    file_names = [
        file_name for file_name in file_names if '{file_name:s}_sync.zip'.format(file_name=file_name) not in existing_files]

    for i, file_name in enumerate(file_names):
        url = '{base_url:s}{file_name:s}/{file_name:s}_sync.zip'.format(
            base_url=BASE_URL, file_name=file_name)
        print('File {index:d} / {total:d}, URL: {url:s}'.format(index=(i+1),
                                                                total=len(file_names), url=url))

        open_url = urllib.request.urlopen(url).info()
        file_size = int(int(open_url['Content-Length']) / (1024 ** 2))

        if file_size > args.size and args.size > 0:
            print('Skipping....')
            continue

        target_file_path = os.path.join(
            output_dir, file_name + '_sync.zip')
        try:
            urllib.request.urlretrieve(url, target_file_path)
        except Exception as e:
            print(e)
            os.remove(target_file_path)


def download_calibration_data(file_names, output_dir):
    calib_output_dir = os.path.join(output_dir, 'calib')
    if not os.path.isdir(calib_output_dir):
        os.mkdir(calib_output_dir)

    existing_files = os.listdir(calib_output_dir)
    existing_dates = [file_name[:10] for file_name in existing_files]

    raw_data_dates = [file_name[:10] for file_name in file_names]
    data_dates = sorted(list(set(raw_data_dates)))

    data_dates = [
        data_date for data_date in data_dates if data_date not in existing_dates]

    for i, data_date in enumerate(data_dates):
        url = '{base_url:s}{data_date:s}_calib.zip'.format(
            base_url=BASE_URL, data_date=data_date)
        print('Calibration file {index:d} / {total:d}, URL:{url:s}'.format(
            index=(i+1), total=len(data_dates), url=url))

        target_file_path = os.path.join(
            calib_output_dir, data_date + '_calib.zip')
        try:
            urllib.request.urlretrieve(url, target_file_path)
        except Exception as e:
            print(e)
            os.remove(target_file_path)


if __name__ == '__main__':
    args = parse_arguments()

    input_path = args.input
    output_dir = args.output

    file_names = extract_file_names(input_path)
    download_sensor_data(file_names, output_dir)
    download_calibration_data(file_names, output_dir)
