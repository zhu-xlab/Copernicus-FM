# Benchmark results for Copernicus-Bench

Frozen encoder evaluation of SOTA EO foundation models on Copernicus-Bench. We report three-run averages with standard deviations.

<div style="overflow-x: auto; white-space: nowrap;">

|                |       Metric      |     Supervised     |     Supervised     |       Random      | SoftCon | CROMA | DOFA |    Copernicus-FM    |
|----------------|:-----------------:|:------------------:|:------------------:|:-----------------:|:----------------------------:|:----------------------------:|:---------------------------:|:-------------------:|
| Backbone       |         --        |      ViT-S/16      |      ViT-B/16      |      ViT-B/16     |           ViT-B/14           |            ViT-B/8           |           ViT-B/16          |       ViT-B/16      |
| Modality       |         --        |         --         |         --         |         --        |             S1/S2            |             S1+S2            |        All (spectral)       |         All         |
| Cloud-S2       |        mIoU       |   64.2 $\pm$ 0.9   |   59.4 $\pm$ 1.0   |   60.4 $\pm$ 0.2  |        66.9 $\pm$ 0.3        |        65.0 $\pm$ 0.2        |        65.0 $\pm$ 0.2       |  **66.7 $\pm$ 0.1** |
| Cloud-S3       |        mIoU       |   61.7 $\pm$ 0.7   | **63.0 $\pm$ 0.8** |   60.9 $\pm$ 0.0  |              --              |              --              |        58.2 $\pm$ 0.1       |    62.0 $\pm$ 0.7   |
| EuroSAT-S1     |         OA        |   81.7 $\pm$ 0.7   |   81.5 $\pm$ 0.9   |   75.4 $\pm$ 0.4  |        83.6 $\pm$ 0.1        |        83.9 $\pm$ 0.1        |        81.7 $\pm$ 0.1       |  **87.2 $\pm$ 0.1** |
| EuroSAT-S2     |         OA        |   97.5 $\pm$ 0.0   |   97.6 $\pm$ 0.1   |   92.5 $\pm$ 0.1  |        96.7 $\pm$ 0.0        |        97.0 $\pm$ 0.1        |        97.2 $\pm$ 0.1       |  **97.9 $\pm$ 0.1** |
| BigEarthNet-S1 |        mAP        |   78.1 $\pm$ 0.6   |   81.2 $\pm$ 0.5   |   66.1 $\pm$ 0.1  |        81.6 $\pm$ 0.0        |        72.8 $\pm$ 0.0        |        74.3 $\pm$ 0.0       |  **83.3 $\pm$ 0.0** |
| BigEarthNet-S2 |        mAP        |   83.6 $\pm$ 0.4   | **86.4 $\pm$ 0.4** |   73.3 $\pm$ 0.1  |        86.1 $\pm$ 0.0        |        78.8 $\pm$ 0.0        |        79.7 $\pm$ 0.0       |    84.6 $\pm$ 0.0   |
| LC100Cls-S3    |        mAP        |   91.3 $\pm$ 0.3   |   91.4 $\pm$ 0.5   |   88.9 $\pm$ 0.1  |              --              |              --              |        89.5 $\pm$ 0.0       |  **93.3 $\pm$ 0.4** |
| DFC2020-S1     |        mIoU       |   49.9 $\pm$ 0.4   |   50.8 $\pm$ 0.5   |   45.4 $\pm$ 0.1  |      **52.8 $\pm$ 0.6**      |      **52.7 $\pm$ 0.1**      |        49.7 $\pm$ 0.1       |    52.4 $\pm$ 0.1   |
| DFC2020-S2     |        mIoU       |   65.3 $\pm$ 0.6   |   66.2 $\pm$ 0.7   |   62.3 $\pm$ 0.0  |        64.1 $\pm$ 0.3        |      **66.5 $\pm$ 0.0**      |        61.8 $\pm$ 0.1       |    64.5 $\pm$ 0.1   |
| LC100Seg-S3    |        mIoU       |   20.1 $\pm$ 0.4   |   19.3 $\pm$ 0.5   |   18.2 $\pm$ 0.1  |              --              |              --              |        16.5 $\pm$ 0.1       |  **24.1 $\pm$ 0.0** |
| Flood-S1       |        mIoU       |   78.0 $\pm$ 0.1   | **78.3 $\pm$ 0.3** |   75.1 $\pm$ 0.1  |        77.2 $\pm$ 0.1        |        77.4 $\pm$ 0.1        |        76.0 $\pm$ 0.1       |    77.7 $\pm$ 0.0   |
| LCZ-S2         |         OA        | **86.6 $\pm$ 0.7** |   85.3 $\pm$ 0.8   |   77.4 $\pm$ 0.1  |        83.6 $\pm$ 0.2        |        84.1 $\pm$ 0.0        |        83.0 $\pm$ 0.3       |    84.4 $\pm$ 0.0   |
| Biomass-S3     | RMSE $\downarrow$ |   68.1 $\pm$ 0.3   |   68.3 $\pm$ 0.4   |   68.7 $\pm$ 0.5  |              --              |              --              |        74.1 $\pm$ 0.1       |  **66.3 $\pm$ 0.1** |
| AQ-NO2-S5P     | RMSE $\downarrow$ |    3.4 $\pm$ 0.0   |    3.4 $\pm$ 0.0   |   3.4 $\pm$ 0.0   |              --              |              --              |        3.3 $\pm$ 0.0        |  **2.8 $\pm$ 0.0**  |
| AQ-O3-S5P      | RMSE $\downarrow$ |  1781.3 $\pm$ 29.8 |  1766.8 $\pm$ 22.1 | 1741.6 $\pm$ 11.5 |              --              |              --              |      1755.6 $\pm$ 19.8      | **789.4 $\pm$ 2.6** |

</div>