#  Light CNN for Deep Face Recognition, in Tensorflow
A Tensorflow implementation of [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683) from the paper by Xiang Wu 

## Updates
- Sep 20, 2017
	- Add model and evaluted code.
	- Add training code.
- Sep 19, 2017
	- The repository was built.


## Datasets
- Training data
	- Download face dataset [MS-Celeb-1M (Aligned)](http://www.msceleb.org/download/aligned).
	- All face images are RGB images and resize to **122x144** 
- Testing data
	- Download aligned LFW (122*144) [images](https://1drv.ms/u/s!AleP5K29t5x7ge88rngfpitnvpkZbw) and [list](https://1drv.ms/t/s!AleP5K29t5x7ge9DV6jfHo392ONwCA)

## Training 
- Add

## Evaluation
- Download [LCNN-29 model](https://1drv.ms/f/s!AleP5K29t5x7ge89GqB3Ue_Pe5rN3A)
- Download [LFW features](https://1drv.ms/u/s!AleP5K29t5x7ge9ElofW_tDzxCq5sw)

## Performance
The Light CNN performance on lfw 6,000 pairs.   

|   Model | ACC | 100% - EER | TPR@FAR=1% | TPR@FAR=0.1% | TPR@FAR=0 | 
| :------- | :----: | :---: | :---: |:---: | 
| LightCNN-29 (Wu Xiang) |   -	| 99.40% |    99.43%    |    98.67%  |    95.70%  | 
| LightCNN-29 (Tensorflow)| 98.36% | 98.2% | 95.13% | 89.53% |


## References
- [Original Light CNN implementation (caffe)](https://github.com/AlfredXiangWu/face_verification_experiment).
