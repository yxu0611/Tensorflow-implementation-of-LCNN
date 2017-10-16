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
- Download [LCNN-29 model](https://1drv.ms/f/s!AleP5K29t5x7ge89GqB3Ue_Pe5rN3A), this model's performacen on LFW:98.2% (100%-EER)
- Download [LFW features](https://1drv.ms/u/s!AleP5K29t5x7ge9ElofW_tDzxCq5sw)

## Performance
The Light CNN performance on lfw 6,000 pairs.   

|   Model | traing data	| method | Acc	|100% - EER | TPR@FAR=1%   | TPR@FAR=0.1%| TPR@FAR=0| 
| :------- | :----: | :----: | :----:| :----: | :---: | :---: |:---: | 
| LightCNN-29 (Wu Xiang)| 70K/-	|Softmax|   -	|99.40% | 99.43% | 98.67% | 95.70% |
| LightCNN-29 (Tensorflow)|10K/- |Softmax|98.36%	|98.2% |    97.73%    |    92.26%  |    60.53%  | 
| LightCNN-29 (Tensorflow)|10K/- |Softmax+L2+PCA|98.76%	|98.66% |    98.36%    |    97%  |    79.33%  |
| LightCNN-29 (Tensorflow)|10K/- |Softmax+L2+PCA+[b]|98.95%	|98.8% |    98.76%    |    97.16%  |    83.36%  |
| LightCNN-29 (Tensorflow)|10K/- |Softmax_enforce+L2+PCA+[b]|99.01%	|98.96% |    98.96%    |    95.83%  |    90.23%  |

|   Model | traing data	| method | Acc	|100% - EER | TPR@FAR=1%   | TPR@FAR=0.1%| TPR@FAR=0| 
| :------- | :----: | :----: | :----:| :----: | :---: | :---: |:---: | 
| LightCNN-29 (Wu Xiang)| 70K/-	|Softmax|   -	|99.40% | 99.43% | 98.67% | 95.70% |
| LightCNN-29 (Tensorflow)|70K/- |Softmax_enforce+L2+PCA|99.18%	|98.9% |    98.86%    |    97.9%  |    94.03%  |
| LightCNN-29 (Tensorflow)|70K/- |Softmax_enforce+L2+PCA+[a]|99.48%	|99.43% |    99.56%    |    98.26%  |    94.53%  |


Some improved solutions:
- [a] It can be further improved by manaully aligned these images which are mis-algined in LFW
- [b] It can be further improved by doing mutiple-crop, e.g. 25 crops for per image
- [c] It can be further improved by ensemble different models
- [d] It can be further improved by adding metric learning method for similarity caculation

## Referencs
- [Original Light CNN implementation (caffe)](https://github.com/AlfredXiangWu/face_verification_experiment).
