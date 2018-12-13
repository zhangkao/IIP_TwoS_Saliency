# IIP_TwoS_Video_Saliency

The code is re-implementation code for the following paper: 

Kao Zhang, Zhenzhong Chen. Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network.  <br />
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
* [SF-Net-model] [百度网盘](https://pan.baidu.com/s/1IAdy6XL3FqTKyImx1gJpcw) [google drive](https://drive.google.com/drive/folders/1CdDJ2s9p0A_Qs5QxHZyb3M1TcwU8lAWq) [70MB]
* [TwoS-model]   [百度网盘](https://pan.baidu.com/s/1MkKxmOPc6itCDpOfyaIKyA) [Google drive](https://drive.google.com/drive/folders/1CdDJ2s9p0A_Qs5QxHZyb3M1TcwU8lAWq) [210MB]
    
    
### Python requirements 
Currently, the code supports python 3.6
* conda
* Keras ( >= 2.2.4)
* tensorflow ( >= 1.12.0) 
* python-opencv
* hdf5storage 

### Train and Test
* please change the working directory: "wkdir" to your path in the "zk_config.py" file, like

        wkdir = '/home/zk/zk/TwoS-release'
        
* More parameters are in the "zk_config.py" file.
* Run the demo "Test_TwoS_Net.py" and "Train_TwoS_Net.py" to test or train the model.

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
zhangkao  <br />
Laboratory of Intelligent Information Processing (LabIIP)  <br />
School of Remote Sensing and Information Engineering,  <br />
Wuhan University,  <br />
430079, Wuhan, China.  <br />
Email: zhangkao@whu.edu.cn  <br />
