import LCNN29 as LCNN29
import numpy as np
import cv2
import scipy.io as sio
import random


def Average(inp):
	a = inp/np.linalg.norm(inp, axis=1, keepdims=True)
	a = np.sum(a, axis=0)
	a = a/np.linalg.norm(a)
	return a

res = []
# labs = []
f = open('lfw_list_part.txt', 'r')
labs = np.empty([13233, 1], dtype=object)
# labs = np.empty([10, 1], dtype=object)
count = 0
for line in f:
	name = []
	line = line.strip()
	name.append(line.split('\\')[-2] + '/' + line.split('\\')[-1])
	labs[count, 0] = name

	imgs = []
	img = cv2.imread(line, 1)
	img = cv2.resize(img,(122,144))
	M2 = np.float32([[1,0,11],[0,1,0]])
	img = cv2.warpAffine(img,M2,(144,144))

	# for i in range(1):
	# 	w = 8
	# 	h = 8
	# 	img2 = img[w:w+128, h:h+128]/255.
	# 	img2 = np.float32(img2)
	# 	imgs.append(img2)

	for i in range(20):
		w = random.randint(0, 16)
		h = random.randint(0, 16)
		img2 = img[w:w+128, h:h+128]/255. 
		img2 = np.float32(img2)
		imgs.append(img2)

	for i in range(5):
		w = random.randint(0, 16)
		h = random.randint(0, 16)
		img2 = img[w:w+128, h:h+128]/255.
		img2 = cv2.flip(img2, 1)
		img2 = np.float32(img2)
		imgs.append(img2)
	
	imgs = np.array(imgs)
	feas = LCNN29.eval(imgs)
	# print (feas.shape)

	feas_avg = Average(feas)
	feas_avg = np.array(feas_avg)
	# print (feas_avg.shape)
	# input()

	res.append(feas_avg)
	count += 1
	# if count == 10:
	# 	break
	if count %10 == 0:
		print (count)
res = np.array(res)
# labs = np.array(labs)
# labs = np.reashape(labs, [-1, labs.shape])
res = np.reshape(res, [13233, 512])
print (res.shape)
print (labs.shape)
sio.savemat('LFW_feas_25.mat',{'data':res, 'label':labs})
f.close()


