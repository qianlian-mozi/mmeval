# 支持矩阵

### 支持的分布式通信后端

|                                                    MPI4Py                                                     |                                                                                                             torch.distributed                                                                                                              |                                                       Horovod                                                       |                                              paddle.distributed                                               |
| :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| [MPI4PyDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.MPI4PyDist) | [TorchCPUDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCPUDist) <br> [TorchCUDADist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TorchCUDADist) | [TFHorovodDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.TFHorovodDist) | [PaddleDist](../api/generated/mmeval.core.dist_backends.MPI4PyDist.html#mmeval.core.dist_backends.PaddleDist) |

### 支持的评测指标及机器学习框架

```{note}
下表列出 MMEval 已实现的评测指标与对应的机器学习框架支持情况，打勾表示能够直接接收对应框架的数据类型（如 Tensor）进行计算。
```

|                                                      评测指标                                                      | NumPy | PyTorch | TensorFlow | Paddle |
| :----------------------------------------------------------------------------------------------------------------: | :---: | :-----: | :--------: | :----: |
|                 [Accuracy](../api/generated/mmeval.metrics.Accuracy.html#mmeval.metrics.Accuracy)                  |   ✔   |    ✔    |            |        |
|    [SingleLabelMetric](../api/generated/mmeval.metrics.SingleLabelMetric.html#mmeval.metrics.SingleLabelMetric)    |   ✔   |    ✔    |            |        |
|                                                  MultiLabelMetric                                                  |   ✔   |    ✔    |            |        |
|                                                  AveragePrecision                                                  |   ✔   |    ✔    |            |        |
|                   [MeanIoU](../api/generated/mmeval.metrics.MeanIoU.html#mmeval.metrics.MeanIoU)                   |   ✔   |    ✔    |            |   ✔    |
|                [VOCMeanAP](../api/generated/mmeval.metrics.VOCMeanAP.html#mmeval.metrics.VOCMeanAP)                |   ✔   |         |            |        |
|                [OIDMeanAP](../api/generated/mmeval.metrics.OIDMeanAP.html#mmeval.metrics.OIDMeanAP)                |   ✔   |         |            |        |
| [CocoDetectionMetric](../api/generated/mmeval.metrics.COCODetectionMetric.html#mmeval.metrics.COCODetectionMetric) |   ✔   |         |            |        |
|                                                   ProposalRecall                                                   |   ✔   |         |            |        |
|                 [F1Metric](../api/generated/mmeval.metrics.F1Metric.html#mmeval.metrics.F1Metric)                  |   ✔   |    ✔    |            |        |
|                 [HmeanIoU](../api/generated/mmeval.metrics.HmeanIoU.html#mmeval.metrics.HmeanIoU)                  |   ✔   |         |            |        |
|                                                    PCKAccuracy                                                     |   ✔   |         |            |        |
|                                                  MpiiPCKAccuracy                                                   |   ✔   |         |            |        |
|                                                  JhmdbPCKAccuracy                                                  |   ✔   |         |            |        |
|          [EndPointError](../api/generated/mmeval.metrics.EndPointError.html#mmeval.metrics.EndPointError)          |   ✔   |    ✔    |            |        |
|                                                     AVAMeanAP                                                      |   ✔   |         |            |        |
|                                                        SSIM                                                        |   ✔   |         |            |        |
|                                                        SNR                                                         |   ✔   |         |            |        |
|                                                        PSNR                                                        |   ✔   |         |            |        |
|                                                        MAE                                                         |   ✔   |         |            |        |
|                                                        MSE                                                         |   ✔   |         |            |        |