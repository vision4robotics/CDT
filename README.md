# [CDT]
# Cascaded Denoising Transformer for UAV Nighttime Tracking

### Kunhan Lu, Changhong Fu, Yucheng Wang, Haobo Zuo, Guangze Zheng, and Jia Pan

This is the official code for the paper "Cascaded Denoising Transformer for UAV Nighttime Tracking".

# Abstract

The automation of unmanned aerial vehicle (UAV) has been greatly promoted by visual object tracking methods with onboard cameras.
However, the random and complicated noise produced by the cameras seriously hinders the performance of state-of-the-art (SOTA) UAV trackers, especially in low-illumination environments.
To address this issue, this work proposes an efficient plug-and-play cascaded denoising Transformer (CDT) to suppress cluttered and complex noise, thereby boosting UAV tracking performance.
Specifically, the novel U-shaped cascaded denoising network is designed with a streamlined structure for efficient computation.
Additionally, shallow feature deepening (SFD) encoder and multi-feature collaboration (MFC) decoder are constructed based on multi-head transposed self-attention (MTSA) and multi-head transposed cross-attention (MTCA), respectively.
A nested residual feed-forward network (NRFN) is developed to focus more on high-frequency information represented by noise.
Extensive evaluation and test experiments demonstrate that the proposed CDT has a remarkable denoising effect and improves UAV nighttime tracking performance.
The source code, pre-trained models, and experimental results are available at here.


## Environment setup


## Demo


## Test

Take the test of SiamAPN_SCT_CDT as an example:

```
python test.py                      \
  --dataset UAVDark135                           \ # dataset_name
  --datasetpath ./test_dataset                    \ # dataset_path
  --config ./experiments/SiamAPN/config.yaml      \ # tracker_config
  --snapshot ./experiments/SiamAPN/model.pth      \ # tracker_model
  --trackername SiamAPN                           \ # tracker_name

  --e_weights ./experiments/SCT/model.pth         \ # enhancer_model
  --enhancername SCT                              \ # enhancer_name

  --d_weights ./experiments/CDT/model.pth         \ # denoiser_model
  --denoisername CDT                              \ # denoiser_name

```

## Evaluation 

If you want to evaluate the trackers mentioned above, please put those results into `results` directory as `results/<dataset_name>/<tracker_name>`.

```
python tools/eval.py                              \
  --dataset UAVDark135                           \ # dataset_name
  --datasetpath path/of/your/dataset              \ # dataset_path
  --tracker_path ./results                        \ # result_path
  --tracker_prefix 'SiamAPN'                        # tracker_name
```

## Contact

If you have any questions, please contact me.

Kunhan Lu

Email: lukunhan@tongji.edu.cn .

## Acknowledgements
- The code is implemented based on [SNOT](https://github.com/vision4robotics/SiameseTracking4UAV). We would like to express our sincere thanks to the contributors.
