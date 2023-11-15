# 只有420l上有空间
# save in /data/yfzhang/dataset/lyft/kitti_format
# /data/yfzhang/data/waymo_mmdet/kitti_format

import os
from tqdm import tqdm

mode='val'
tfrecord_list_file=f'/home/yifan_zhang/git/OpenPCDet/data/waymo/ImageSets/{mode}.txt'
tfrecord_list = [x.strip() for x in open(tfrecord_list_file, 'r').readlines()]

# format: 0 000 000 x + sequence_id + frame_id
mmdet_frame_list_file = f'/data/yfzhang/data/waymo_mmdet/kitti_format/ImageSets/{mode}.txt'
src_dir = '/data/yfzhang/data/waymo_mmdet/kitti_format/training/' # image_{}
tgt_dir = '/data/yfzhang/data/waymo_pcdet_v0.5/kitti_format/' # image_{}

# image 路径
mmdet_frame_list = [x.strip() for x in open(mmdet_frame_list_file, 'r').readlines()]
for frame in tqdm(mmdet_frame_list):
    # import pdb;pdb.set_trace()
    tfrecord_index = int(str(frame)[1:4])
    frame_index = str(frame)[4:7]

    sequence_name = tfrecord_list[tfrecord_index].split(".")[0]

    for image_idx in range(5):
        src_image_path = f'{src_dir}/image_{image_idx}/{frame}.jpg'
        tgt_image_path = f'{tgt_dir}/image_{image_idx}/{sequence_name}/0{frame_index}.jpg'
        save_dir = f'{tgt_dir}/image_{image_idx}/{sequence_name}'
        os.makedirs(save_dir, exist_ok=True)
        os.symlink(src_image_path, tgt_image_path)

    # link calib files
    src_calib_path = f'{src_dir}/calib/{frame}.txt'
    tgt_calib_path = f'{tgt_dir}/calib/{sequence_name}/0{frame_index}.txt'
    save_dir = f'{tgt_dir}/calib/{sequence_name}'
    os.makedirs(save_dir, exist_ok=True)
    os.symlink(src_calib_path, tgt_calib_path)







