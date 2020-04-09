import requests
import os
from imutils import paths
import cv2


with open('links.txt', 'r') as f:
	urls = f.readlines()

out_folder = 'images/'
for url in urls:
	try:
		r = requests.get(url, timeout=60)
		p = os.path.join(out_folder, f"{str(total).zfill(8)}.jpg")
		with open(p, 'wb') as f:
			f.write(r.content)
		print(f'[INFO] downloaded: {p}')
		total += 1
	except:
		print("[INFO] error downloading {}...skipping".format(p))

for imagePath in paths.list_images(out_folder):
	print(imagePath)
	delete = False
	image = cv2.imread(imagePath)
	if image is None:
			delete = True

	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)
