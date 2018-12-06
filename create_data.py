from coco_annotation import get_img_paths, parse_images
import numpy as np
from random import shuffle
from mrcnn import utils, visualize
import json

def create_train_val(train_test_split = .8):
	img_files = get_img_paths()
	shuffle(img_files)
	split_idx = int(.8*len(img_files))
	train, test = img_files[:split_idx], img_files[split_idx:]
	parse_images(train, save_fp = 'data/coco_train.json')
	parse_images(test, save_fp = 'data/coco_test.json')

# create_train_val()

class PersonDataset(utils.Dataset):
	'''Generate a person dataset based on COCO conventions using
	COCO tools'''

	def load_data(self, annotation_json):
		coco_json = json.load(open(annotation_json))

		# Add class using utils.Dataset method
		for cat in coco_json['categories']:
			c_id = cat['id']
			name = cat['name']
			self.add_class('supervisely', c_id, name)

		annotations = {}
		for annotation in coco_json['annotations']:
			image_id = annotation['image_id']
			if image_id not in annotations:
				annotations[image_id] = []
			annotations[image_id].append(annotation)

		# Add image to dataset. Remember that we have reorganized
		# from individual elements of annotations and gathered them
		# to create a list of annotations(masks) for each image
		parsed_imgs = {}
		for img in coco_json['images']:
			image_id = img['id']
			if image_id in parsed_imgs:
				continue
			else:
				parsed_imgs[image_id] = image_id
				img_fp = img['file_path']
				img_width = img['width']
				img_height = img['height']
				img_annotations = annotations[image_id]

				# Add image using utils.Dataset method
				self.add_image(
					source = 'supervisely',
					image_id = image_id,
					path = img_fp,
					width= img_width,
					height = img_height,
					annotations = img_annotations)

	def load_mask(self, image_id):
		""" Load instance masks for the given image.
		MaskRCNN expects masks in the form of a bitmap [height, width, instances].
		Args:
		    image_id: The id of the image to load masks for
		Returns:
		    masks: A bool array of shape [height, width, instance count] with
		        one mask per instance.
		    class_ids: a 1D array of class IDs of the instance masks.
		"""
		image_info = self.image_info[image_id]
		annotations = image_info['annotations']
		instance_masks = []
		class_ids = []

		for annotation in annotations:
			class_id = annotation['category_id']
			mask = Image.new('1', (image_info['width'], image_info['height']))
			mask_draw = ImageDraw.ImageDraw(mask, '1')
			segmentation = annotation['segmentation']
			mask_draw.polygon(segmentation, fill=1)
			bool_array = np.array(mask) > 0
			instance_masks.append(bool_array)
			class_ids.append(class_id)

		mask = np.dstack(instance_masks)
		class_ids = np.array(class_ids, dtype=np.int32)

		return mask, class_ids