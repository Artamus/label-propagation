import pypcd
import os
import glob
import numpy as np

from cityscapes_classes import id_to_color

base_path = os.path.join('/', 'home', 'markus', 'thesis', 'data', 'val_output')
output_base_path = os.path.join('/', 'home', 'markus', 'thesis', 'data', 'val_output_color')
pcd_files = os.listdir(base_path)

for pcd_file in pcd_files:

    pcd_file_path = os.path.join(base_path, pcd_file)

    pcd = pypcd.PointCloud.from_path(pcd_file_path)

    x = pcd.pc_data['x']
    y = pcd.pc_data['y']
    z = pcd.pc_data['z']
    labels = pcd.pc_data['label'].astype(np.int32)

    colours = [id_to_color(label) for label in labels]
    colours = np.array(colours).astype(np.uint8)

    rgb = pypcd.encode_rgb_for_pcl(colours)

    new_points = np.c_[x, y, z, labels]
    new_pcd = pypcd.make_xyz_label_point_cloud(new_points)
    # new_points = np.c_[x, y, z, rgb]
    
    # new_pcd = pypcd.make_xyz_rgb_point_cloud(new_points)
    
    base_pcd_name = pcd_file[:-4]
    new_pcd_name = base_pcd_name + '_colour.pcd'

    new_pcd_path = os.path.join(output_base_path, new_pcd_name)

    new_pcd.save_pcd(new_pcd_path, compression='ascii')

