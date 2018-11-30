# IIP_TwoS_Video_Saliency

The code is re-implementation code for the following paper: 

Kao Zhang, Zhenzhong Chen, Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network.  <br />
IEEE Trans. Circuits Syst. Video Techn. 2018. [Online] Avaliable: https://ieeexplore.ieee.org/document/8543830 <br />



## Environment:

    windows10/Ubuntu16.04, anaconda5.2.0, python3.6, cuda9.0, cudnn7.1.2

## Installation 
### Pre-trained models
Download the pre-trained models and put the pre-trained model into the "Models" file.
* [SF-net-model] ( https://github.com/zhangkao/IIP_TwoS_Saliency)[70MB]
* [TwoS-model]   ( https://github.com/zhangkao/IIP_TwoS_Saliency)[210MB]
    
    
### Python requirements 
Currently, the code supports python 3.6
* conda
* Keras ( >= 2.2.4)
* tensorflow ( >= 1.12.0) 
* python-opencv
* hdf5storage 

### Train and Test
* please change the working directory: "wkdir" to your path in the "zk_config.py" file, like
    "# wkdir = '/home/zk/zk/TwoS-release'" 
* More parameters are in the "zk_config.py" file.
* Run the demo "Test_TwoS_Net.py" and "Train_TwoS_Net.py" to test or train the model.

### Output format
And it is easy to change the output format in our code.
* The results of video task is saved by ".mat"(uint8) formats.
* You can get the color visualization results based on the "Visualization Tools".
* You can evaluate the performance based on the "EvalScores Tools".


## Contact
zhangkao@whu.edu.cn
Insititue of Intelligent Sensing and Computing (IISC),
School of Remote Sensing and Information Engineering,
Wuhan University,
430079, Wuhan, China.
Email: zhangkao@whu.edu.cn
