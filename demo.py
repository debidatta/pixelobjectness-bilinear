import os
import sys
import fnmatch
import scipy
import scipy.io as sio
import numpy as np
from skimage.io import imread
from bilateral_solver import apply_bilateral_files

#Caffe binary location from deeplab-v1 installation
caffe_binary = '/usr1/debidatd/deeplab-public/build/tools/caffe.bin' 

#Extension of images that need to be processed
ext ='jpg'

base_dir = os.getcwd()
image_dir = os.path.join(base_dir,'images')

image_list = fnmatch.filter(os.listdir(image_dir),'*.'+ext)
image_list.sort()

input_list_file  = base_dir + '/image_list.txt'
output_list_file = base_dir + '/output_list.txt'

input_list  = open(input_list_file,'w')
output_list = open(output_list_file,'w')

for img in image_list:
	input_list.write('/'+img+'\n')
	prefix = img.split('.')[0]
	output_list.write(prefix+'\n')

input_list.close()
output_list.close()

template_file = open(base_dir + '/test_template.prototxt').readlines()

test_file_path = base_dir + '/test.prototxt'
test_file = open(test_file_path,'w')

tokens = {}
tokens['${IMAGE_DIR}'] = 'root_folder: \"' + image_dir + '\"'
tokens['${OUTPUT_DIR}'] = 'prefix: \"' + image_dir + '/\"'

tokens['${IMAGE_LIST}']        = 'source: \"' + input_list_file + '\"'
tokens['${IMAGE_OUTPUT_LIST}'] = 'source: \"' + output_list_file + '\"'

for line in template_file:
	line = line.rstrip()

	for key in tokens:
		if line.find(key)!=-1:
			line = '\t'+tokens[key]
			break

	test_file.write(line+'\n')

test_file.close()

weight_file_path = base_dir + '/pixel_objectness.caffemodel'
cmd = caffe_binary + ' test --model=' + test_file_path + ' --weights=' + weight_file_path + ' --gpu=3 --iterations='+str(len(image_list))
os.system(cmd)

THRESH1 = 0.2
THRESH2 = 0.8

for img in image_list:
    im = imread('images/'+img)
    img_only = img.split('.')[0]
    results_name = img_only+'_blob_0.mat'
    data = sio.loadmat('images/'+results_name)['data']
    channel_swap = (1, 0, 2, 3)
    data = data.transpose(channel_swap)
    h,w = im.shape[:2]
    w_m, h_m = data.shape[1], data.shape[0]
    score = data[:min(h,h_m),:min(w,w_m),:]
    mask = score[:,:,1].squeeze()>THRESH1
    scipy.misc.imsave('images/'+img_only+'_mask.png', mask)
    scipy.misc.imsave('images/'+img_only+'_conf.png', score[:,:,1].squeeze())    
    cleaned_mask = apply_bilateral_files('images/'+img, 'images/'+img_only+'_mask.png', 'images/'+img_only+'_conf.png', thresh=THRESH2, plot=True)
    scipy.misc.imsave('images/'+img_only+'_cleanedmask.png', 255*cleaned_mask)
