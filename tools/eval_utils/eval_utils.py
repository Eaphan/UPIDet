import pickle
import time

import numpy as np
import cv2
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

def mIoU(preds, labels, n_classes=3, return_list=True):
    bs = preds.shape[0]
    batch_mIoUs = []
    for b_idx in range(bs):
        pred = preds[b_idx]
        pred_label = torch.argmax(pred, dim=0)
        
        label = labels[b_idx]
        ious = []
        for c_idx in range(n_classes):
            mul = (pred_label==c_idx) & (label==c_idx)
            intersection = torch.sum(mul)
            union = torch.sum(pred_label==c_idx) + torch.sum(label==c_idx) - intersection
            if union == 0:
                ious.append(np.nan)
            else:
                iou = intersection / union
                iou = iou.cpu().numpy()
                ious.append(iou)
        # miou = np.nanmean(ious)
        # ious.append(miou)
        batch_mIoUs.append(ious)
    if return_list:
        return batch_mIoUs
    else:
        raise NotImplementedError
        # return np.nanmean(batch_mIoUs)

def statistics_info(cfg, ret_dict, metric, disp_dict):
    # for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
    #     metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
    #     metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    for key in metric.keys():
        if key in ret_dict:
            metric[key] += ret_dict[key]
    # metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    
    # initialize customized statistics
    if hasattr(model, 'init_recall_record'):
        model.init_recall_record(metric)

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            _r = model(batch_dict)
            if len(_r)==3:
                pred_dicts, ret_dict, segmentation_preds = _r
            else:
                pred_dicts, ret_dict = _r
            # if 'segmentation_label' in batch_dict:
            #     pred_dicts, ret_dict, segmentation_preds = model(batch_dict)
            # else:
            #     pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        # ad hoc
        if 'part_image_preds' in pred_dicts[0] and save_to_file:
            # part_image_preds = pred_dicts['part_image_preds']
            # bs = part_image_preds.shape[0]
            for i in range(len(pred_dicts)):
                part_image_preds = pred_dicts[i]['part_image_preds']
                frame_id = batch_dict['frame_id'][i]
                part_image_preds = np.transpose(part_image_preds * 255.0, (1, 2, 0)).astype(np.uint8)
            part_image_output_dir = result_dir / 'part_image_preds'
            part_image_output_dir.mkdir(parents=True, exist_ok=True)
            filename = str(part_image_output_dir) + '/part_image_{}.png'.format(frame_id)
            cv2.imwrite(filename, part_image_preds)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall
    
    # print customized statistics
    if dist_test:
        if hasattr(model.module, 'disp_recall_record'):
            model.module.disp_recall_record(metric, logger, sample_num=len(dataloader.dataset))
    else:
        if hasattr(model, 'disp_recall_record'):
            model.disp_recall_record(metric, logger, sample_num=len(dataloader.dataset))


    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
