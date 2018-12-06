import numpy as np
from helper_plot import overlay_img_poly
from skimage import measure
from shapely.geometry import Polygon
import pickle
import json
import glob
import zlib
import base64
import cv2
import os

def get_img_paths():
	'''Return a tuple containing path of image file and corresponding path of json meta file'''

	cwd = os.getcwd()
	data_dir = os.path.join(cwd, 'data/supervisely/')
	
	img_meta_tup = []
	for folder_path in glob.glob(data_dir + 'Supervisely*'):
		image_files = glob.glob(folder_path+ '/img/*.png') + \
			glob.glob(folder_path+ '/img/*.jpg')
		for img_path in image_files:
			fn = img_path.split('/')[-1]
			meta_fn = fn[:-3] + 'json'
			meta_path = os.path.join(folder_path, 'ann', meta_fn)

			img_meta_tup.append((img_path, meta_path))

	return img_meta_tup

def coco_annotate(json_data, img_dims=(640,640)):
	'''We require the train/test data to have COCO annotation to input
	into a Masked R-CNN pre-trained network. The annotation format is
	a list of dicts with the following attribute (per image):

	annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    Refer to http://cocodataset.org/#format-data for addition info'''

	mask_data = []
	for obj in json_data['objects']:
		if obj['classTitle'] not in ['person_poly', 'person_bmp']:
			continue
		if obj['bitmap'] != None:
			str_data = obj['bitmap']['data']
			origin = obj['bitmap']['origin']
			enc_data = np.array(base64_2_mask(str_data))
			exterior = bitmap_to_polygon(enc_data, origin, img_dims)
		else:
			exterior = obj['points']['exterior']
			exterior = Polygon(exterior)

		x, y, max_x, max_y = exterior.bounds
		width = max_x - x
		height = max_y - y
		bbox = x, y, width, height
		area = exterior.area
		x, y = exterior.exterior.xy
		segmentation = np.array(exterior.exterior.coords).ravel().tolist()

		mask_data.append((segmentation, bbox, area))

	return mask_data

def base64_2_mask(s):
	'''Turns a base64 encoded string into binary mask'''

	z = zlib.decompress(base64.b64decode(s))
	n = np.fromstring(z, np.uint8)
	mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
	return mask

def bitmap_to_polygon(bitmap, origin, img_dims):
	'''Given a bitmap, top left coordinate, convert to Polygon exterior coords'''

	mask_rows, mask_cols = bitmap.shape
	img = np.zeros(img_dims)
	img[origin[1]:mask_rows+origin[1], origin[0]:mask_cols+origin[0]] = bitmap
	contours = measure.find_contours(img, 0.5, positive_orientation='low')
	contour = contours[0]
	for i in range(len(contour)):
		row, col = contour[i]
		contour[i] = (col - 1, row - 1)
	exterior = Polygon(contour)
	# exterior = exterior.simplify(1.0, preserve_topology=False)
	return exterior

def parse_images(img_paths, save_fp=''):
	'''Given image path containing tuple of (image_path, meta_path),
	return a json list of coco annotations, category, and image_attr
	to form a coco json'''

	is_crowd = 0
	image_id = 1
	category_id = 1
	annotation_id = 1
	
	coco_a = []
	coco_images = []
	coco_cat = [{'id': 1, 'name': 'person'}]

	for file_p, meta_p in img_paths:
		print('Parsing', meta_p)
		data = json.load(open(meta_p))
		mask_data = coco_annotate(data)

		img_info = {
			'id': image_id,
			'file_path': file_p,
			'width': 640,
			'height': 640
		}

		coco_images.append(img_info)

		for mask in mask_data:
			segmentation, bbox, area = mask

			annotations = {
				'segmentation': segmentation,
				'iscrowd': is_crowd,
				'image_id': image_id,
				'category_id': category_id,
				'id': annotation_id,
				'bbox': bbox,
				'area': area
			}

			coco_a.append(annotations)
			annotation_id += 1
		image_id += 1

	annotation_json ={'annotations': coco_a, 'categories': coco_cat, 'images': coco_images}

	if save_fp:
		with open(save_fp, 'w') as f:
			json.dump(annotation_json, f)

		name = 'train_tuple.p' if 'train' in save_fp else 'test_tuple.p'
		name_path = 'data/' + name
		with open(name_path, 'wb') as f:
			pickle.dump(img_paths, f)

	return annotation_json