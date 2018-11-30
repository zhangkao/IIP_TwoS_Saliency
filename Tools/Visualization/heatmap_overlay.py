import cv2
import numpy as np

def im2uint8(img):
    if img.dtype == np.uint8:
        return img
    else:
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.rint(img).astype(np.uint8)
        return img

def heatmap_overlay(image, heatmap):

	img = np.array(image,   copy=True)
	map = np.array(heatmap, copy=True)

	if img.shape[:2] != map.shape[:2]:
		map = cv2.resize(map, (img.shape[1],img.shape[0]))

	if len(map.shape) == 2:
		map = np.repeat(np.expand_dims(map, axis=2), 3, axis=2)

	if map.dtype == np.uint8:
		map_color = cv2.applyColorMap(map, cv2.COLORMAP_JET)
	else:
		tmap = im2uint8(map/np.max(map)*255)
		map_color = cv2.applyColorMap(tmap, cv2.COLORMAP_JET)

	img = img / np.max(img)
	map = map / np.max(map)
	map_color = map_color / np.max(map_color)

	o_map = 0.8 * (1 - map ** 0.8) * img + map * map_color
	return o_map


if __name__ == "__main__":

	img = cv2.imread('img.jpg')
	map = cv2.imread('map.png',-1)

	color_map = heatmap_overlay(img,map)

	cv2.imshow("",color_map)
	cv2.waitKey(100)

	print("done!!!")