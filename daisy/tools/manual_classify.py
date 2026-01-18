from pathlib import Path
from shutil import copyfile
import cv2
import numpy as np


def classify_images(image_folder: Path, output_folders: Path, num_class: int):
	# 获取所有图片文件
	image_files = [f for f in image_folder.iterdir() if f.name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]

	for image_file in image_files:
		img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), -1)

		if img is None:
			print(f'无法加载图片: {image_file}')
			continue
		cv2.namedWindow('Image', 0)
		h, w = img.shape[:2]
		print(f'图片尺寸: {w}x{h}')
		delta = 400 / h
		cv2.resizeWindow('Image', int(w * delta), int(h * delta))
		cv2.imshow('Image', img)
		print(f'当前图片: {image_file}')
		exists = False
		for i in range(1, num_class + 1):
			if (output_folders / f'{i}' / image_file.name).exists():
				exists = True
				break
		if exists:
			print('该图片已经分类，下一张。')
			continue
		while True:
			key = cv2.waitKey(0) & 0xFF  # 等待键盘输入

			if key >= ord('1') and key <= ord('9'):  # 检测1-9的按键
				folder_index = key - ord('0')
				if 1 <= folder_index <= num_class:
					dest_path = output_folders / f'{folder_index}' / image_file.name
					(output_folders / f'{folder_index}').mkdir(parents=True, exist_ok=True)
					copyfile(image_file, dest_path)
					print(f'已将图片 {image_file} 复制到 {dest_path}')
					break
				else:
					print('无效的文件夹编号，请重新输入。')
			elif key == 27:  # ESC键退出
				cv2.destroyAllWindows()
				return
			elif key == 255:
				return
			else:
				print(f'无效的输入{key}，请按1-9进行分类，或按ESC退出。')

		cv2.destroyAllWindows()


if __name__ == '__main__':
	# 设置图片文件夹和输出文件夹
	image_folder = Path(r'C:\Resources\Datasets\20250301-Smile-Images')
	output_folders = Path(r'C:\Resources\Datasets\20250301-Smile-Images-Wash\task_1')

	classify_images(image_folder, output_folders, 3)
