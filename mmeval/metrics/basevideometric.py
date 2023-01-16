import numpy as np
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, overload
import os.path as osp
import pickle
import shutil
import tempfile
import warnings
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, overload
from mmeval.core.base_metric import BaseMetric
from mmeval.core.dispatcher import dispatch
from mmeval.utils import try_import
from mmeval.metrics.utils import mkdir_or_exist

if TYPE_CHECKING:
    import paddle
    import tensorflow
    import tensorflow as tf
    import torch
else:
    # 使用 try_import
    paddle = try_import('paddle')
    torch = try_import('torch')
    tf = try_import('tensorflow')


class BaseVideoMetric(BaseMetric):
    """Base class for a metric in video task.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseVideoMetric` should assign a meaningful value
    to the class attribute `default_prefix`. See the argument `prefix` for
    details.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.  #处理所有批次后，评估整个数据集的模型性能。

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        results = collect_tracking_results(self.results, self.collect_device)
        if is_main_process():  # mmengine 的 dist 文件，用以判断是否是主进程，下面是主进程才用的
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)  # 将``Object_list'''中的挑选对象广播到整个组 \
        # 类似于：func：“ broadcast”，但是可以传递python对象。 同样的dist才使用的办法。

        # reset the results list
        self.results.clear()
        return metrics[0]


# 上面先中断，因为用到了下面的一个新的函数，来自源文件#70row
def collect_tracking_results(results: list,
                             device: str = 'cpu',
                             tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments. 收集分布式环境的结果

    Args:
        results (list): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        device (str): Device name. Optional values are 'cpu' and 'gpu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu'. Defaults to None.

    Returns:
        list or None: The collected results.
    """

    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(
            f"device must be 'cpu' or 'gpu', but got {device}")
    if device == 'gpu':
        assert tmpdir is None, 'tmpdir should be None when device is "gpu"'
        raise NotImplementedError('GPU collecting has not been supported yet')
    else:
        return collect_tracking_results_cpu(results, tmpdir)


def collect_tracking_results_cpu(result_part: list,
                                 tmpdir: Optional[str] = None
                                 ) -> Optional[list]:
    """Collect results on cpu mode.  CPU环境的收集结果

    Saves the results on different gpus to 'tmpdir' and collects them by the
    rank 0 worker.

    Args:
        result_part (list): The part of prediction results.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. If is None, use `tempfile.mkdtemp()`
            to make a temporary path. Defaults to None.

    Returns:
        list or None: The collected results.
    """
    rank, world_size = get_dist_info()  # 这里是获取给定进程组的分布式信息，来自于mmengine，考虑可能要重写一遍在mmeval里；但是之前其他的metric里面是怎么处理分布式运算的？
    if world_size == 1:
        return result_part

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8)  # 是存在np.full 和 np.uint8的，似乎可以直接替换
        if rank == 0:
            mkdir_or_exist('.dist_test')  # 来自mmengine 的 utils 中；使用os做的，已迁移过来
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(  # 再次有tensor，是否可以直接写ndarray？
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)  # 本来自于mmengine.dist 作用就是广播代表src进程的data到目标group
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()  # 这里就是doc里面提到的，把tensor转换成ndarry的行为
        # 原来算法库 Metric 有这转换操作，不该放在mmeval的metric中，要么是单独写一个分派机制来存放，要么优先考虑用ndarry来重写（工作量应该在这）

    else:
        mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()  # mmengine 设定好的dist的同步方式，还是那句话得去看mmeval其他metric中怎么实现dist的

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            with open(path, 'rb') as f:
                part_list.extend(pickle.load(f))
        shutil.rmtree(tmpdir)
        return part_list
