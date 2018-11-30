# IIP_TwoS_Video_Saliency

The code is re-implementation code for the following paper: 

Kao Zhang, Zhenzhong Chen, Video Saliency Prediction Based on Spatial-Temporal Two-Stream Network.  <br />
IEEE Trans. Circuits Syst. Video Techn. 2018. [Online] Avaliable: https://ieeexplore.ieee.org/document/8543830 <br />



## environment:

    windows10/Ubuntu16.04, anaconda5.2.0, python3.6, cuda9.0, cudnn7.1.2

## Installation 

    # get the source
    git clone https://github.com/zhangkao/IIP_TwoS_Saliency.git
    
    # download the pre-trained models
    * [SF-net-model](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing)[70MB]
    * [TwoS-model](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view?usp=sharing)[210MB]
    and put the pre-trained model into the "Models" file.
    
### Python requirements 
Currently, the code supports python 3.6
* conda
* Keras ( >= 2.2.4)
* tensorflow ( >= 1.12.0) 
* python-opencv
* hdf5storage 
	
zhangkao@whu.edu.cn
Insititue of Intelligent Sensing and Computing (IISC),
School of Remote Sensing and Information Engineering,
Wuhan University,
430079, Wuhan, China.
Email: zhangkao@whu.edu.cn
