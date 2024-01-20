import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
import pdb
import json
import pickle
from tqdm import tqdm
import torch
import copy

from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from pcdet.datasets.kitti import kitti_utils
from pcdet.utils import calibration_kitti
from pcdet.utils import box_utils

split='val'

kitti_root_path = '/home/yifanzhang/git/OpenPCDet/data/kitti'
kitti_infos_path = kitti_root_path + f'/kitti_infos_{split}_ori.pkl'
kitti_dbinfos_path = kitti_root_path + f'/kitti_dbinfos_{split}.pkl'

sem_json_path = '/public/sdi/yfzhang/dataset/kins/instances_train.json'

kitti_infos = pickle.load(open(kitti_infos_path, 'rb'))
kitti_dbinfos = pickle.load(open(kitti_dbinfos_path, 'rb'))

sem_annos = json.load(open(sem_json_path, 'rb'))
sem_annos_id_map_trueid = {}
for img in sem_annos['images']:
    image_id = img['id']
    sem_annos_id_map_trueid[img['file_name'].split('.')[0]] = image_id

sem_annos_id_map_name = {
    1: 'Cyclist',
    2: 'Pedestrian',
    4: 'Car',
    5: 'Tram',
    6: 'Truck',
    7: 'Van',
    8: 'Misc',
}

anns_info = sem_annos["annotations"]
anns_dict = {}
for ann in anns_info:
    image_id = ann["image_id"]
    if not image_id in anns_dict:
        anns_dict[image_id] = []
        anns_dict[image_id].append(ann)
    else:
        anns_dict[image_id].append(ann)

records_dict={}#key={image_idx}_{gt_idx}, value = inmodal_seg

for frame in tqdm(kitti_infos):
    image_idx = frame['image']['image_idx']

    # if image_idx!='000011':
    #     continue
    
    image_shape = frame['image']['image_shape']
    if image_idx not in sem_annos_id_map_trueid:
        sem_anns = []
    else:
        sem_anns = anns_dict.get(sem_annos_id_map_trueid[image_idx], [])

    sem_anns_backup = copy.deepcopy(sem_anns)
    sem_anns_selected_masks = [False] * len(sem_anns_backup)

    lidar_file = kitti_root_path + f'/training/velodyne/{image_idx}.bin'
    calib_file = kitti_root_path + f'/training/calib/{image_idx}.txt'
    image_file = kitti_root_path + f'/training/image_2/{image_idx}.png'
    points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    with open(calib_file, 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
        R0 = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
        V2C = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    # V2R, P2 = kitti_utils.calib_to_matricies(calib)

    boxes3d = frame['annos']['gt_boxes_lidar']
    point_indices = points_in_boxes_cpu(points[:, :3], boxes3d)

    paste_order = frame['annos']['num_points_in_gt'].argsort()
    paste_order = paste_order[::-1]
    sem_idx_list = [-1] * len(paste_order)
    npoint_in_anns_list_list = [-1] * len(paste_order)
    show_mask_list = [False] * len(paste_order)
    for item_idx in paste_order:
        name = frame['annos']['name'][item_idx]
        # if name == 'DontCare':
        #     sem_idx_list.append(-1)
        #     # npoint_in_anns_list_list.append([])
        #     npoint_in_anns_list_list[item_idx] = []
        #     continue
        if name not in ['DontCare', 'Car', 'Pedestrian', 'Cyclist']:
            sem_idx_list.append(-1)
            # npoint_in_anns_list_list.append([])
            npoint_in_anns_list_list[item_idx] = []
            continue
        
        # get point clouds
        try:
            scan = points[point_indices[item_idx]==1]
        except:
            import pdb;pdb.set_trace()
        scan_hom = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1), dtype=np.float32))) # [N, 4]
        scan_C0 = np.dot(scan_hom, np.dot(V2C.T, R0.T)) # [N, 3]
        scan_C0_hom = np.hstack((scan_C0, np.ones((scan.shape[0], 1), dtype=np.float32))) # [N, 4]
        scan_C2 = np.dot(scan_C0_hom, P2.T) # [N, 3]
        scan_C2_depth = scan_C2[:, 2]
        scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T #(u,v)
        ## change to integral coords in image planes
        # remove points outside the image

        scan_C2 = scan_C2.astype(np.int)

        inds = scan_C2[:, 0] > 0
        inds = np.logical_and(inds, scan_C2[:, 0] < image_shape[1])
        inds = np.logical_and(inds, scan_C2[:, 1] > 0)
        inds = np.logical_and(inds, scan_C2[:, 1] < image_shape[0])
        inds = np.logical_and(inds, scan_C2_depth > 0)

        # count of points in sem mask
        # print('### size', [ann['inmodal_seg']['size'] for ann in sem_anns])
        npoint_in_anns_list = []
        for ann in sem_anns:
            if ann is None:
                npoint_in_anns_list.append(0)
                continue

            inmodal_ann_mask = maskUtils.decode(ann['inmodal_seg'])[:,:,np.newaxis] # (H,W,1)
            # import pdb;pdb.set_trace()
            try:
                npoint_in_ann = inmodal_ann_mask[scan_C2[inds, 1], scan_C2[inds, 0]].sum()
            except:
                pdb.set_trace()
            npoint_in_anns_list.append(npoint_in_ann)
        
        # print(npoint_in_anns_list, frame['annos']['num_points_in_gt'][item_idx], "name=", name)
        if len(npoint_in_anns_list)>0:
            sem_idx = npoint_in_anns_list.index(max(npoint_in_anns_list))
            if npoint_in_anns_list[sem_idx]<=0:
                sem_idx = -1
            
            if npoint_in_anns_list[sem_idx]>0:
                # assert sem_annos_id_map_name[sem_anns[sem_idx]['category_id']] == name, "{}!={}".format(sem_annos_id_map_name[sem_anns[sem_idx]['category_id']], name)
                # if sem_annos_id_map_name[sem_anns[sem_idx]['category_id']] != name:
                    # print(npoint_in_anns_list, frame['annos']['num_points_in_gt'][item_idx], "name=", name)
                    # print("###", f"image_idx{image_idx} ", frame['annos']['index'][item_idx], "kins={} kitti={}".format(sem_annos_id_map_name[sem_anns[sem_idx]['category_id']], name))
                kins_name = sem_annos_id_map_name[sem_anns[sem_idx]['category_id']]
                if name == 'Cyclist' and kins_name not in ['Cyclist', 'Pedestrian', 'Misc']:
                    sem_idx = -1
                if name == 'Pedestrian' and kins_name not in ['Cyclist', 'Pedestrian', 'Misc']:
                    sem_idx = -1
                if name == 'Car' and kins_name not in ['Car', 'Truck', 'Van', 'Misc']:
                    sem_idx = -1
        else:
            sem_idx = -1
        # assert category

        # limit the IoU between frame['annos']['bbox'][item_idx] and sem_anns[sem_idx]['inmodal_bbox']
        # unneccessary
        # if sem_idx!=-1:
        #     try:
        #         if 'inmodal_bbox' in sem_anns[sem_idx]:
        #             x,y,w,h = sem_anns[sem_idx]['inmodal_bbox']
        #         else:
        #             x,y,w,h = sem_anns[sem_idx]['bbox'] # self-anno
        #     except:
        #         import pdb;pdb.set_trace()
        #     sem_bbox = np.array([x,y,x+w,y+h]).reshape(1,4)
        #     iou2d = box_utils.boxes_iou_normal(frame['annos']['bbox'][item_idx].reshape(1,-1), sem_bbox)
        #     if iou2d < 0.3:
        #         # sem_idx = -1
        #         show_mask_list[item_idx] = True

        # sem_idx_list.append(sem_idx)
        # npoint_in_anns_list_list.append(npoint_in_anns_list)

        sem_idx_list[item_idx] = sem_idx
        npoint_in_anns_list_list[item_idx] = npoint_in_anns_list

        if sem_idx!=-1:
            sem_anns[sem_idx] = None
            # if sem_anns_selected_masks[sem_idx]:
            #     show_mask_list[item_idx] = True
            # sem_anns_selected_masks[sem_idx] = True



    # import pdb;pdb.set_trace()
    frame['annos']['inmodal_seg'] = [sem_anns_backup[idx]['inmodal_seg'] if idx!=-1 else None for idx in sem_idx_list]
    # check duplicate
    for gt_idx in frame['annos']['index']:
        if gt_idx==-1:
            continue
        k = f'{image_idx}_{gt_idx}'
        try:
            records_dict[k] = frame['annos']['inmodal_seg'][gt_idx]
        except:
            import pdb;pdb.set_trace()


    ## view single object
    # for item_idx in range(len(frame['annos']['name'])):
    #     if not show_mask_list[item_idx]:
    #         continue
    #     bbox = frame['annos']['bbox'][item_idx]
    #     x1,y1,x2,y2 = [int(np.floor(x)) for x in bbox]
    #     # import pdb;pdb.set_trace()
    #     cur_image = copy.copy(image)
    #     image_with_bbox = cv2.rectangle(cur_image, (x1,y1), (x2,y2), (0,255,0), 2)

    #     if sem_idx_list[item_idx]!=-1:
    #         inmodal_ann_mask = maskUtils.decode(sem_anns_backup[sem_idx_list[item_idx]]['inmodal_seg'])[:,:,np.newaxis]
    #         inmodal_ann_mask = inmodal_ann_mask * 255
    #         inmodal_ann_mask = np.concatenate((inmodal_ann_mask, inmodal_ann_mask, inmodal_ann_mask), axis=2)
    #         show_img = np.concatenate((image_with_bbox, inmodal_ann_mask), axis=0)
    #     else:
    #         show_img = image_with_bbox

    #     cv2.imshow("show_img", show_img)
    #     # print("###", npoint_in_anns_list_list[item_idx], 'image_idx=', image_idx, 'name=', 
    #     #          frame['annos']['name'][item_idx], 'difficulty=', frame['annos']['difficulty'][item_idx],)
    #     print(" current image_idx", image_idx)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("show_img")


kitti_infos_out_path = kitti_root_path + f'/kitti_infos_{split}_new.pkl'

with open(kitti_infos_out_path, 'wb') as f:
    pickle.dump(kitti_infos, f)

# kitti_infos = pickle.load(open(kitti_root_path + '/kitti_infos_train_wconf_v5_sem.pkl', 'rb'))
# kitti_infos = pickle.load(open(kitti_root_path + '/kitti_infos_train_wconf_v5_sem_v1.pkl', 'rb'))

# 'inmodal_seg': {'size': [375, 1242], 'counts': 'Umk61e;1O1O1O1O1O1O1N2O1O1O1O1O1O1O1N2O1O1O1O1O1O1O1N2O1O1O1[fP7'}
# list of last line

for cls_name in kitti_dbinfos.keys():
    none_count = 0
    for db_info in kitti_dbinfos[cls_name]:
        # inmodal_seg
        image_idx = db_info['image_idx']
        gt_idx = db_info['gt_idx']
        k = f'{image_idx}_{gt_idx}'
        inmodal_seg = records_dict.get(k, None)
        if inmodal_seg is None:
            none_count += 1
            # print("##", db_info['num_points_in_gt'], 'diff=', db_info['difficulty'])
        db_info['inmodal_seg'] = inmodal_seg
    print(f"none_count={none_count}, len of {cls_name} db_infos = {len(kitti_dbinfos[cls_name])}")
kitti_dbinfos_out_path = kitti_root_path + f'/kitti_dbinfos_{split}_new.pkl'
# import pdb;pdb.set_trace()
# print('### new keys', kitti_dbinfos['Car'][0].keys())
with open(kitti_dbinfos_out_path, 'wb') as f:
    pickle.dump(kitti_dbinfos, f)

