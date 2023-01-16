# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict,OrderedDict

import numpy as np
from typing import List, Optional, Sequence, Tuple, overload

from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.metrics.utils import YTVIS,YTVISeval

try:
    import torch
except ImportError:
    torch = None


class YouTubeVISMetric(BaseMetric):
    """mAP evaluation metrics for the VIS task.
            Args:
        num_classes (int, optional): The number of classes. If None, it will be
            obtained from the 'num_classes' or 'classes' field in
            `self.dataset_meta`. Defaults to None.
        ignore_index (int, optional): Index that will be ignored in evaluation.
            Defaults to 255.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Defaults to None.
        beta (int, optional): Determines the weight of recall in the F-score.
            Defaults to 1.
        classwise_result (bool, optional): Whether to return the computed
            results of each class. Defaults to False.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

        """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 ignore_index: int = 255,
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 classwise_results: bool = False,
                 metric_items: Optional[Sequence[str]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._num_classes = num_classes
        self.ignore_index = ignore_index
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.classwise_results = classwise_results
        self._vis_meta_info = defaultdict(list)  # record video and image infos
        self.metric_items = metric_items

    @property
    def num_classes(self) -> int:
        """Returns the number of classes.
        The number of classes should be set during initialization, otherwise it
        will be obtained from the 'classes' or 'num_classes' field in
        ``self.dataset_meta``.
        Raises:
            RuntimeError: If the num_classes is not set.
        Returns:
            int: The number of classes.
        """
        if self._num_classes is not None:
            return self._num_classes
        if self.dataset_meta and 'num_classes' in self.dataset_meta:
            self._num_classes = self.dataset_meta['num_classes']
        elif self.dataset_meta and 'classes' in self.dataset_meta:
            self._num_classes = len(self.dataset_meta['classes'])
        else:
            raise RuntimeError(
                'The `num_claases` is required, and not found in '
                f'dataset_meta: {self.dataset_meta}')
        return self._num_classes

    def add(self, predictions: Sequence, groundtruths: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """
        Add the intermediate results to ``self._results``.
            Args:
            predictions (Sequence[dict]): A sequence of dict. Each dict
                representing a detection result for an image, with the
                following keys:
                - bboxes (numpy.ndarray): Shape (N, 4), the predicted
                  bounding bboxes of this image, in 'xyxy' foramrt.
                - scores (numpy.ndarray): Shape (N, 1), the predicted scores
                  of bounding boxes.
                - labels (numpy.ndarray): Shape (N, 1), the predicted labels
                  of bounding boxes.
                -instances_id():
                -masks():
            groundtruths (Sequence[dict]): A sequence of dict. Each dict
                represents a groundtruths for an image, with the following
                keys:
                -width
                -height
                -img_id
                -frame_id
                -video_id
                -video_length
                -anns
        """
        for prediction, groundtruth in zip(predictions, groundtruths):
            assert isinstance(prediction, dict), 'The prediciton should be ' \
                                                 f'a sequence of dict, but got a sequence of {type(prediction)}.'  # noqa: E501
            assert isinstance(groundtruth, dict), 'The label should be ' \
                                            f'a sequence of dict, but got a sequence of {type(groundtruth)}.'
            self._results.append((prediction, groundtruth))

            # 在这里，miou用了一个混淆矩阵的方法然后根据这个结果获取了三类标签；但是vis该怎么获取呢；
            # 看起来似乎miou这边是通过混淆矩阵计算后能获得三种标签，但vis那边像是在构造dict然后datasample读入还是说塞进去标签
            # add方法好像就这样结束了？但我得去补充一下两个dict应该有什么，从vis的那边

    def compute_metric(self, results: list) -> dict:
        """Compute the metrics from processed results.
        Args:
            results (List): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        tmp_pred_results, tmp_gt_results = zip(*results)
        gt_results = self.format_gts(tmp_gt_results)
        pred_results = self.format_preds(tmp_pred_results)
        ytvis = YTVIS(gt_results)
        ytvis_dets = ytvis.loadRes(pred_results)
        vid_ids = ytvis.getVidIds()
        iou_type = metric = 'segm'
        eval_results = OrderedDict()
        ytvisEval = YTVISeval(ytvis, ytvis_dets, iou_type)  # 这行在干嘛？
        ytvisEval.params.vidIds = vid_ids
        ytvisEval.evaluate()
        ytvisEval.accumulate()
        ytvisEval.summarize()

        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@1': 6,
            'AR@10': 7,
            'AR@100': 8,
            'AR_s@100': 9,
            'AR_m@100': 10,
            'AR_l@100': 11
        }

        metric_items = self.metric_items   # 评测哪些指标
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item "{metric_item}" is not supported')

        if metric_items is None:
            metric_items = [
                'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
            ]
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{ytvisEval.stats[coco_metric_names[metric_item]]:.3f}')
            eval_results[key] = val

        return eval_results



    # 下面摘录两个转换，但是它们是对于给出的信息进行转换的
    def _format_one_video_preds(self, pred_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.  这个函数和下方的gt函数本来是对于gts, preds进行套用后转化格式的（pts，preds是在process中解包得到的信息）
        This operation is to make it easier to use the official eval API.
        Args:
            pred_dicts (Tuple[dict]): Prediction of the dataset.
        Returns:
            List: The formatted predictions.
        """
        # Collate preds scatters (tuple of dict to dict of list)
        preds = defaultdict(list)
        for pred in pred_dicts:
            for key in pred.keys():
                preds[key].append(pred[key])

        img_infos = self._vis_meta_info['images']
        vid_infos = self._vis_meta_info['videos']
        inds = [i for i, _ in enumerate(img_infos) if _['frame_id'] == 0]
        inds.append(len(img_infos))
        json_results = []
        video_id = vid_infos[-1]['id']
        # collect data for each instances in a video.
        collect_data = dict()
        for frame_id, (masks, scores, labels, ids) in enumerate(
                zip(preds['masks'], preds['scores'], preds['labels'],
                    preds['instances_id'])):

            assert len(masks) == len(labels)
            for j, id in enumerate(ids):
                if id not in collect_data:
                    collect_data[id] = dict(
                        category_ids=[], scores=[], segmentations=dict())
                collect_data[id]['category_ids'].append(labels[j])
                collect_data[id]['scores'].append(scores[j])
                if isinstance(masks[j]['counts'], bytes):
                    masks[j]['counts'] = masks[j]['counts'].decode()
                collect_data[id]['segmentations'][frame_id] = masks[j]

        # transform the collected data into official format
        for id, id_data in collect_data.items():
            output = dict()
            output['video_id'] = video_id
            output['score'] = np.array(id_data['scores']).mean().item()
            # majority voting for sequence category
            output['category_id'] = np.bincount(
                np.array(id_data['category_ids'])).argmax().item() + 1
            output['segmentations'] = []
            for frame_id in range(inds[-1] - inds[-2]):
                if frame_id in id_data['segmentations']:
                    output['segmentations'].append(
                        id_data['segmentations'][frame_id])
                else:
                    output['segmentations'].append(None)
            json_results.append(output)

        return json_results

    def _format_one_video_gts(self, gt_dicts: Tuple[dict]) -> List:
        """Convert the annotation to the format of YouTube-VIS.  这个是从gts获取，因此获取的东西和preds有点不同导致了代码的不同
        This operation is to make it easier to use the official eval API.
        Args:
            gt_dicts (Tuple[dict]): Ground truth of the dataset.
        Returns:
            list: The formatted gts.
        """
        video_infos = []
        image_infos = []
        instance_infos = defaultdict(list)
        len_videos = dict()  # mapping from instance_id to video_length
        vis_anns = []

        # get video infos
        for gt_dict in gt_dicts:
            frame_id = gt_dict['frame_id']
            video_id = gt_dict['video_id']
            img_id = gt_dict['img_id']
            image_info = dict(
                id=img_id,
                width=gt_dict['width'],
                height=gt_dict['height'],
                frame_id=frame_id,
                file_name='')
            image_infos.append(image_info)
            if frame_id == 0:
                video_info = dict(
                    id=video_id,
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                    file_name='')
                video_infos.append(video_info)

            for ann in gt_dict['anns']:
                label = ann['bbox_label']
                bbox = ann['bbox']
                instance_id = ann['instance_id']
                # update video length
                len_videos[instance_id] = gt_dict['video_length']
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                annotation = dict(
                    video_id=video_id,
                    frame_id=frame_id,
                    bbox=coco_bbox,
                    instance_id=instance_id,
                    iscrowd=ann.get('ignore_flag', 0),
                    category_id=int(label) + 1,
                    area=coco_bbox[2] * coco_bbox[3])
                if ann.get('mask', None):
                    mask = ann['mask']
                    # area = mask_util.area(mask)
                    if isinstance(mask, dict) and isinstance(
                            mask['counts'], bytes):
                        mask['counts'] = mask['counts'].decode()
                    annotation['segmentation'] = mask

                instance_infos[instance_id].append(annotation)

        # update vis meta info
        self._vis_meta_info['images'].extend(image_infos)
        self._vis_meta_info['videos'].extend(video_infos)

        for instance_id, ann_infos in instance_infos.items():
            cur_video_len = len_videos[instance_id]
            segm = [None] * cur_video_len
            bbox = [None] * cur_video_len
            area = [None] * cur_video_len
            # In the official format, no instances are represented by
            # 'None', however, only images with instances are recorded
            # in the current annotations, so we need to use 'None' to
            # initialize these lists.
            for ann_info in ann_infos:
                frame_id = ann_info['frame_id']
                segm[frame_id] = ann_info['segmentation']
                bbox[frame_id] = ann_info['bbox']
                area[frame_id] = ann_info['area']
            instance = dict(
                category_id=ann_infos[0]['category_id'],
                segmentations=segm,
                bboxes=bbox,
                video_id=ann_infos[0]['video_id'],
                areas=area,
                id=instance_id,
                iscrowd=ann_infos[0]['iscrowd'])
            vis_anns.append(instance)
        return vis_anns

    # 下面是两个摘录的，format实现，可以让compute_metric函数中的获取去掉tmp
    def format_gts(self, gts: Tuple[List]) -> dict:
        """Gather all ground-truth from self.results."""
        self.categories = [
            dict(id=id + 1, name=name)
            for id, name in enumerate(self.dataset_meta['CLASSES'])
        ]
        gt_results = dict(
            categories=self.categories,
            videos=self._vis_meta_info['videos'],
            annotations=[])
        for gt_result in gts:
            gt_results['annotations'].extend(gt_result)
        return gt_results

    def format_preds(self, preds: Tuple[List]) -> List:
        """Gather all predictions from self.results."""
        pred_results = []
        for pred_result in preds:
            pred_results.extend(pred_result)
        return pred_results

