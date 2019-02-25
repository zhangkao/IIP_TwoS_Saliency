# IIP_TwoS_Video_Saliency

It is a re-implementation code for the following paper: 

Kao Zhang, Zhenzhong Chen. Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network.
IEEE Trans. Circuits Syst. Video Techn. 2018. [Online] Avaliable: https://ieeexplore.ieee.org/document/8543830 <br />


## Installation 
### Environment:
The code was developed using Python 3.6 & Keras 2.2.4 & CUDA 9.0. There may be a problem related to software versions.To fix the problem, you may look at the implementation in "zk_models.py" file and replace the syntax to match the new keras environment. 
* Windows10/Ubuntu16.04
* Anaconda 5.2.0
* Python 3.6
* CUDA 9.0 and cudnn7.1.2

### Pre-trained models
Download the pre-trained models and put the pre-trained model into the "Models" file.

* [TwoS-model]
 [[Baidu Drive](https://pan.baidu.com/s/1MkKxmOPc6itCDpOfyaIKyA)]
 [[Google Drive](https://drive.google.com/open?id=1vXTjW8MjW4308j1HM1Y_MBpUxmcX3I2k)]

* [SF-Net-model]
[[Baidu Drive](https://pan.baidu.com/s/1yT7LUfDzC1aT_L3-4-ivdw)]
[[Google Drive](https://drive.google.com/open?id=1nmzdxsSbePF9aOkl9GDUMO7Ndz5NTVT7)]
    
    
### Python requirements 
Currently, the code supports python 3.6
* conda
* Keras ( >= 2.2.4)
* tensorflow ( >= 1.12.0) 
* python-opencv
* hdf5storage 

### Train and Test

* please change the working directory: "wkdir" to your path in the "zk_config.py" file, like

        dataDir = 'E:/Code/IIP_TwoS_Saliency/DataSet'
        
* More parameters are in the "zk_config.py" file.
* Run the demo "Test_TwoS_Net.py" and "Train_TwoS_Net.py" to test or train the model.

The full training process:

Our model is trained on SALICON and part of the DIEM dataset. We train the SF-Net 
in spatial stream based on the pre-trained VGG-16 model and the training set of SALICON dataset.
Then, we train the whole network on the training set of DIEM dataset, and fix the parameters 
of the trained SF-Net.

* Please download SALICON and DIEM dataset.
* Run the demo "Train_Test_ST_Net.py" to get pre-trained SF-Net model.
* Run the demo "Train_TwoS_Net.py" to train the whole model.



### Output format
And it is easy to change the output format in our code.
* The results of video task is saved by ".mat"(uint8) formats.
* You can get the color visualization results based on the "Visualization Tools".
* You can evaluate the performance based on the "EvalScores Tools".

## Paper & Citation

If you use the TwoS video saliency model, please cite the following paper: 
```
@article{Zhang2018Video,
  author  = {Kao Zhang and Zhenzhong Chen},
  title   = {Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology },
  year    = {2018}
}
```

## Contact
Kao ZHANG  <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zhangkao@whu.edu.cn  <br />

Zhenzhong CHEN (Professor and Director) <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
Wuhan University, Wuhan, China.  <br />
Email: zzchen@whu.edu.cn  <br />
Web: http://iip.whu.edu.cn/~zzchen/  <br />