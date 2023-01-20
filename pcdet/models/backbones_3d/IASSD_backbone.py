import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...models.model_utils.pspnet import PSPModel
import os

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (N, 3 + C)
        angle: float, angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = torch.cos(angle)
    sina = torch.sin(angle)

    points_rot_x = points[:, 0:1] * cosa - points[:, 1:2] * sina
    points_rot_y = points[:, 0:1] * sina + points[:, 1:2] * cosa
    points_rot = torch.cat([points_rot_x, points_rot_y, points[:, 2:3]], axis=1)

    return points_rot


class IASSD_Backbone(nn.Module):
    """ Backbone for IA-SSD"""

    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        self.aggregation_mlps = sa_config.get('AGGREGATION_MLPS', None)
        self.confidence_mlps = sa_config.get('CONFIDENCE_MLPS', None)
        self.max_translate_range = sa_config.get('MAX_TRANSLATE_RANGE', None)


        for k in range(sa_config.NSAMPLE_LIST.__len__()):
            if isinstance(self.layer_inputs[k], list): ###
                channel_in = channel_out_list[self.layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[self.layer_inputs[k]]



            if self.layer_types[k] == 'SA_Layer':
                mlps = sa_config.MLPS[k].copy()
                channel_out = 0
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]

                if self.aggregation_mlps and self.aggregation_mlps[k]:
                    aggregation_mlp = self.aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else:
                    aggregation_mlp = None

                if self.confidence_mlps and self.confidence_mlps[k]:
                    confidence_mlp = self.confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                        npoint_list=sa_config.NPOINT_LIST[k],
                        sample_range_list=sa_config.SAMPLE_RANGE_LIST[k],
                        sample_type_list=sa_config.SAMPLE_METHOD_LIST[k],
                        radii=sa_config.RADIUS_LIST[k],
                        nsamples=sa_config.NSAMPLE_LIST[k],
                        mlps=mlps,
                        use_xyz=True,                                                
                        dilated_group=sa_config.DILATED_GROUP[k],
                        aggregation_mlp=aggregation_mlp,
                        confidence_mlp=confidence_mlp,
                        num_class = self.num_class
                    )
                )

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=sa_config.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range
                                                                    )
                                       )
            if k<=3 and k>=0:
                channel_out += [32,32,64,64][k]

            channel_out_list.append(channel_out)

        self.num_point_features = channel_out
        self.image_backbone = PSPModel(n_classes=self.num_class+1, input_channels=3, ratio=0.5)
        # self.segmentation_head = nn.Conv2d(32, self.num_class+1, 1)

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']

        # image backbone
        # img_conv1 = self.image_backbone.relu(self.image_backbone.conv1(batch_dict['images']))
        # img_layer1 = self.image_backbone.layer1(self.image_backbone.maxpool(img_conv1))
        # img_layer2 = self.image_backbone.layer2(self.image_backbone.maxpool(img_layer1))
        # img_features_list = [img_conv1, img_layer1, img_layer2]
        # [torch.Size([2, 16, 188, 621]), torch.Size([2, 16, 94, 311]), torch.Size([2, 32, 24, 78])]
        # print("### img features shape", [x.shape for x in img_features_list])

        segmentation_out, image_feature_dict = self.image_backbone(batch_dict['images'])
        # segmentation_out = self.segmentation_head(image_x)
        # segmentation_out = nn.LogSoftmax()(segmentation_out)
        batch_dict['segmentation_preds'] = segmentation_out
        # batch_dict['images'].shape
        # import pdb;pdb.set_trace()

        img_features_list = [image_feature_dict['conv_s2'], image_feature_dict['conv_s4'], image_feature_dict['conv_s8']]

        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        # points(bs*n,5) -> encoder_coords(bs,n,4)
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        world_scale = batch_dict['noise_scale'] if 'noise_scale' in batch_dict else None
        world_rotation = batch_dict['noise_rot'] if 'noise_rot' in batch_dict else None
        flip_along_x = batch_dict['flip_x'] if 'flip_x' in batch_dict else None
        V2R=batch_dict['trans_lidar_to_cam']
        P2=batch_dict['trans_cam_to_img']
        # image_shape=batch_dict['image_shape']
        image_shape = batch_dict['images'].shape[-2:]

        li_cls_pred = None
        for i in range(len(self.SA_modules)):
            # import pdb;pdb.set_trace()
            xyz_input = encoder_xyz[self.layer_inputs[i]] # LAYER_INPUT: [0, 1, 2, 3, 4, 3]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                # return [bs,npoint,3], [bs,c=\sum_k(mlps[k][-1]),npoint], (B, npoint, num_class)
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            elif self.layer_types[i] == 'Vote_Layer': #i=4
                # xyz_select==xyz_input
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                encoder_coords.append(torch.cat([center_origin_batch_idx[..., None].float(),centers_origin.view(batch_size, -1, 3)],dim =-1))
                    
            if i<=2: # fuse image features
                image_features = img_features_list[i]
                point_image_features_list = []
                for bs_idx in range(batch_size):
                    keypoints_b = li_xyz[bs_idx].clone()
                    if self.training:
                        # restore the raw positions of keypoints, seq: scale->rotate->flip?
                        if world_scale is not None:
                            world_scale_b = world_scale[bs_idx]
                            keypoints_b /= world_scale_b
                        if world_rotation is not None:
                            world_rotation_b = world_rotation[bs_idx]
                            keypoints_b = rotate_points_along_z(keypoints_b, -world_rotation_b)
                        if flip_along_x is not None:
                            flip_along_x_b = flip_along_x[bs_idx] # ad hoc, only process flip_x
                            if flip_along_x_b:
                                keypoints_b[:, 1] = -keypoints_b[:, 1]

                    
                    # project keypoint to image
                    keypoints_b_hom = torch.cat([keypoints_b, keypoints_b.new_ones(len(keypoints_b),1)], dim=-1)
                    scan_C0 = torch.mm(keypoints_b_hom, V2R[bs_idx].T)
                    scan_C2 = torch.mm(scan_C0, P2[bs_idx].T) # [N, 3]
                    scan_C2_depth = scan_C2[:, 2]
                    scan_C2 = (scan_C2[:, :2].T / scan_C2[:, 2]).T
                    # import pdb;pdb.set_trace()
                    scan_C2[:, 0] *= (image_features[bs_idx].shape[2]/image_shape[1]) # w
                    scan_C2[:, 1] *= (image_features[bs_idx].shape[1]/image_shape[0]) # h
                    cur_image_features = image_features[bs_idx].permute(1, 2, 0)  # (C,H,W) -> (H, W, C)
                    cur_point_image_features = bilinear_interpolate_torch(cur_image_features, scan_C2[:, 0], scan_C2[:, 1])
                    point_image_features_list.append(cur_point_image_features)
                point_image_features = torch.stack(point_image_features_list).permute(0,2,1)
                li_features = torch.cat([li_features, point_image_features], dim=1)

            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)            
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
            else:
                sa_ins_preds.append([])
           
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)

        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)

        
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx
        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features
        
        
        ###save per frame 
        if self.model_cfg.SA_CONFIG.get('SAVE_SAMPLE_LIST',False) and not self.training:  
            import numpy as np 
            result_dir = np.load('/home/yifan/tmp.npy', allow_pickle=True)
            for i in range(batch_size)  :
                # i=0      
                # point_saved_path = '/home/yifan/tmp'
                point_saved_path = result_dir / 'sample_list_save'
                os.makedirs(point_saved_path, exist_ok=True)
                idx = batch_dict['frame_id'][i]
                xyz_list = []
                for sa_xyz in encoder_xyz:
                    xyz_list.append(sa_xyz[i].cpu().numpy()) 
                if '/' in idx: # Kitti_tracking
                    sample_xyz = point_saved_path / idx.split('/')[0] / ('sample_list_' + ('%s' % idx.split('/')[1]))

                    os.makedirs(point_saved_path / idx.split('/')[0], exist_ok=True)

                else:
                    sample_xyz = point_saved_path / ('sample_list_' + ('%s' % idx))

                np.save(str(sample_xyz), xyz_list)
                # np.save(str(new_file), point_new.detach().cpu().numpy())
        
        return batch_dict
