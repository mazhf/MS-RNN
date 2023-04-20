
### Source Code for Papers:

`MS-RNN: A Flexible Multi-Scale Framework for Spatiotemporal Predictive Learning`

`PrecipLSTM: A Meteorological Spatiotemporal LSTM for Precipitation Nowcasting`

`MS-LSTM: Exploring Spatiotemporal Multiscale Representations in Video Prediction Domain`

### Reproduced Models:
| ConvRNNs  | MS-RNNs |
| ------------- | ------------- |
| ConvLSTM  | MS-ConvLSTM  |
| TrajGRU  | MS-TrajGRU  |
| PredRNN  | MS-PredRNN  |
| PredRNN++  | MS-PredRNN++  |
| MIM  | MS-MIM  |
| MotionRNN  | MS-MotionRNN  |
| PredRNN-V2  | MS-PredRNN-V2  |
| PrecipLSTM  | MS-PrecipLSTM  |
| CMS-LSTM  | MS-CMS-LSTM  |
| MoDeRNN  | MS-MoDeRNN  |
| MK-LSTM  | MS-LSTM  |

### Installing Libraries:

#### * Installing Libraries:
```shell
pip3 install -r requirements.txt
```

#### * Installing CUDA (Only Needed for Reproducing PrecipLSTM/MS-PrecipLSTM):
Higher versions of CUDA are not supported, and CUDA 11.1 is recommended.
[https://blog.csdn.net/qq_40947610/article/details/114757551](https://blog.csdn.net/qq_40947610/article/details/114757551)

#### * Installing Local Attention (Only Needed for Reproducing PrecipLSTM/MS-PrecipLSTM):
```shell
cd img_local_att
python setup.py install
```
[https://github.com/zzd1992/Image-Local-Attention](https://github.com/zzd1992/Image-Local-Attention)

### Hyperparameters:
See connfig.py

###  Running:
```shell
python -m torch.distributed.launch --nproc_per_node=4 main.py
```
### Citation：
##### If you find this repository useful, please cite the following papers.

```
@article{ma2022ms,
  title={MS-RNN: A flexible multi-scale framework for spatiotemporal predictive learning},
  author={Ma, Zhifeng and Zhang, Hao and Liu, Jie},
  journal={arXiv preprint arXiv:2206.03010},
  year={2022}
}
```
```
@article{ma2022preciplstm,
  title={PrecipLSTM: A Meteorological Spatiotemporal LSTM for Precipitation Nowcasting},
  author={Ma, Zhifeng and Zhang, Hao and Liu, Jie},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--8},
  year={2022},
  publisher={IEEE}
}
```
```
@article{ma2023ms,
  title={MS-LSTM: Exploring Spatiotemporal Multiscale Representations in Video Prediction Domain},
  author={Ma, Zhifeng and Zhang, Hao and Liu, Jie},
  journal={arXiv preprint arXiv:2304.07724},
  year={2023}
}
```
