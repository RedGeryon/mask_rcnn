import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def plot_polygon(x,y):
	'''Plot the perimeter polygon'''
	
	fig = plt.figure(figsize=(20,20))
	ax = fig.add_subplot(111)
	ax.plot(x, y, color='#6699cc', alpha=0.7,
	    linewidth=3, solid_capstyle='round', zorder=2)
	ax.set_title('Polygon')
	plt.show()

def overlay_img_poly(img_fp, x, y, bbox):
	'''Overlay an image with segmentation boundary and bounding box'''

	perimeter = (list(zip(x,y)))
	box_coords = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
	print(box_coords)

	img = Image.open(img_fp)
	img2 = img.copy()
	draw = ImageDraw.Draw(img2)
	draw.polygon(perimeter, fill = 'yellow')
	draw.rectangle(box_coords, outline = 'red')
	img3 = Image.blend(img, img2, 0.5)
	plt.imshow(img3, interpolation='nearest')
	plt.show()