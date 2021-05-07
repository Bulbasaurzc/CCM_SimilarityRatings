 # Imports 
import caffe 
import pickle 
import numpy as np
from scipy import spatial
from itertools import combinations 

extract_from_layer = 'fc7'
caffe_root = '/home/dawooood/caffe/'
pretrained_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
input_images_file = 'img_paths.txt' 
output_pkl_file_name = 'output_file_images.pkl'
ratings_file_name = 'ratings_matrix.csv'
batch_size = 2

ext_file = open(input_images_file, 'r')
image_paths_list = [line.strip() for line in ext_file]
ext_file.close()
print(image_paths_list)

images_loaded_by_caffe = [caffe.io.load_image(im) for im in image_paths_list]

net = caffe.Net(model_def, pretrained_model, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 
transformer.set_transpose('data', (2,0,1)) 
transformer.set_channel_swap('data', (2,1,0)) 
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
output_dict={}

for j in range(total_batch_nums+1):
  image_batch_to_process = get_this_batch(images_loaded_by_caffe, j, batch_size)
  num_images_being_processed = len(image_batch_to_process)
  print('num_images_being_processed',num_images_being_processed)
  if num_images_being_processed == 0:
    break
  data_blob_index = range(num_images_being_processed)
  print(net.blobs['data'].data[data_blob_index].shape)
  a = np.array([transformer.preprocess('data', img) for img in image_batch_to_process])
  print(a.shape)
  net.blobs['data'].data[data_blob_index] =    [transformer.preprocess('data', img) for img in image_batch_to_process]
  res = net.forward()
  features_for_this_batch =    net.blobs[extract_from_layer].data[data_blob_index].copy()
  features_all_images.extend(features_for_this_batch)
features_all_images = np.array(features_all_images)
print(features_all_images.shape)
F = np.dot(features_all_images,np.transpose(features_all_images))
np.savetxt('sim.csv', features_all_images, delimiter=',')
# pkl_object = {"filename": image_paths_list, "features": features_all_images} 
# output = open(output_pkl_file_name, 'wb') 
# pickle.dump(pkl_object, output, 2) 
# output.close()
# n = len(image_paths_list)

# sim_ratings = np.zeros((n,n))
# sim_ratings[:] = np.nan

# objects=[]
# imgs = {}
# features =[]
# with (open(output_pkl_file_name, "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break
# print(objects)          

# for i in range(len(objects[0]['filename'])):
#     imgs[objects[0]['filename'][i].split('/')[-1].split('.')[0]] = objects[0]['features'][i]
# images_order = ["bl1", "bl2", "bl3","hw1","hw2", "hw3", "ib1","ib2", "ib3","mw1", "mw2", "mw3", "pb1", "pb2", "pb3", "sc1", "sc2", "sc3"]


# combos = list(combinations(images_order, 2))


# for i in range(len(combos)):
#     img1 = combos[i][0]
#     img2 = combos[i][1]
#     n_img1 = images_order.index(img1)#np.where(images_order==img1)
#     n_img2 = images_order.index(img2)#np.where(images_order==img2)
#     feat_img_1 = imgs[img1]
#     feat_img_2 = imgs[img2]
#     sim_ratings[n_img1,n_img2]= 1 - spatial.distance.cosine(feat_img_1, feat_img_2)
#     sim_ratings[n_img2,n_img1] = sim_ratings[n_img1,n_img2]
#     print(1 - spatial.distance.cosine(feat_img_1, feat_img_2))
# np.fill_diagonal(sim_ratings,1)
# np.savetxt(ratings_file_name, sim_ratings, delimiter=",")
    