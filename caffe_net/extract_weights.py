# # Imports 
import caffe 
import pickle 
import numpy as np


# required only if working in gpu mode 
extract_from_layer = 'fc7'
caffe_root = '/home/dawooood/caffe/'
pretrained_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
input_images_file = '/home/dawooood/Desktop/ccm/caffe_net/img_paths.txt' 
output_pkl_file_name = '/home/dawooood/Desktop/ccm/caffe_net/output_file_fc8.pkl'
batch_size = 2

ext_file = open(input_images_file, 'r')
image_paths_list = [line.strip() for line in ext_file]
ext_file.close()
print(image_paths_list)

images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list]

net = caffe.Net(model_def, pretrained_model, caffe.TEST)
# set up transformer - creates transformer object 
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 
# transpose image from HxWxC to CxHxW 
transformer.set_transpose('data', (2,0,1)) 
# swap image channels from RGB to BGR 
transformer.set_channel_swap('data', (2,1,0)) 
# set raw_scale = 255 to multiply with the values loaded with caffe.io.load_image 
transformer.set_raw_scale('data', 255)


def get_this_batch(image_list, batch_index, batch_size):
     start_index = batch_index * batch_size
     next_batch_size = batch_size
     image_list_size = len(image_list)
     # batches might not be evenly divided
     if(start_index + batch_size > image_list_size):
           reamaining_size_at_last_index = image_list_size - start_index
           next_batch_size = reamaining_size_at_last_index
     batch_index_indices = range(start_index, start_index+next_batch_size,1)
     return image_list[batch_index_indices]

total_batch_nums = int(len(images_loaded_by_caffe)/batch_size)
features_all_images = []
images_loaded_by_caffe = np.array(images_loaded_by_caffe)
# loop through all the batches
for j in range(total_batch_nums+1):
  image_batch_to_process = get_this_batch(images_loaded_by_caffe, j, batch_size)
  num_images_being_processed = len(image_batch_to_process)
  print('num_images_being_processed',num_images_being_processed)
  if num_images_being_processed == 0:
    break
  data_blob_index = range(num_images_being_processed)
  # note that each batch is passed through a transformer
  # before passing to data layer
  print(net.blobs['data'].data[data_blob_index].shape)
  a = np.array([transformer.preprocess('data', img) for img in image_batch_to_process])
  print(a.shape)
  net.blobs['data'].data[data_blob_index] =    [transformer.preprocess('data', img) for img in image_batch_to_process]
  
  # BEWARE: blobs arrays are overwritten
  res = net.forward()
  # actual batch feature extraction
  features_for_this_batch =    net.blobs[extract_from_layer].data[data_blob_index].copy()
  features_all_images.extend(features_for_this_batch)

pkl_object = {"filename": image_paths_list, "features": features_all_images} 
output = open(output_pkl_file_name, 'wb') 
pickle.dump(pkl_object, output, 2) 
output.close()

objects = []
with (open(output_pkl_file_name, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(objects)          
from scipy import spatial

dataSetI = objects[0]['features'][1]
dataSetII = objects[0]['features'][0]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

print(result)