import os
import cv2
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from typing import cast


def cv_imread(file_path):
	cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
	return cv_img


def show_gallery(
	images: list[str | Path | NDArray],
	grid_size: tuple[int, int] = (3, 3),
	image_size: tuple[int, int] = (224, 224),
	window_name: str = 'Image Gallery',
) -> None:
	"""
	显示图片画廊，支持翻页浏览

	Args:
	    images: 图片路径或图片对象的列表
	    grid_size: 每页显示的行数和列数，格式为(h, w)
	    image_size: 每张图片的显示大小，格式为(height, width)
	    window_name: 窗口标题
	"""
	if not images:
		print('没有图片可以显示')
		return

	# 将所有输入转换为numpy数组
	processed_images = []
	filenames = []

	for img in images:
		if isinstance(img, (str, Path)):
			path = Path(img)
			try:
				image = cv_imread(str(path))
				if image is None:
					print(f'无法读取图片: {path}')
					continue
				processed_images.append(image)
				filenames.append(path.name)
			except Exception as e:
				print(f'处理图片 {path} 时出错: {e}')
		elif isinstance(img, np.ndarray):
			processed_images.append(img)
			filenames.append(f'Image_{len(filenames)}')
		else:
			print(f'不支持的图片类型: {type(img)}')

	if not processed_images:
		print('没有有效的图片可以显示')
		return

	# 计算页面数量
	rows, cols = grid_size
	images_per_page = rows * cols
	num_pages = (len(processed_images) + images_per_page - 1) // images_per_page

	current_page = 0
	img_h, img_w = image_size

	# 在图片底部添加文字区域的高度
	text_height = 30
	cell_height = img_h + text_height

	# 状态栏高度
	status_bar_height = 30

	while True:
		# 计算当前页面的起始和结束索引
		start_idx = current_page * images_per_page
		end_idx = min(start_idx + images_per_page, len(processed_images))

		# 创建空白画布
		canvas_height = rows * cell_height + status_bar_height
		canvas_width = cols * img_w
		canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

		# 在当前页面上放置图片
		for i in range(start_idx, end_idx):
			idx = i - start_idx
			row = idx // cols
			col = idx % cols

			# 调整图片大小
			img = processed_images[i]
			resized = cv2.resize(img, (img_w, img_h))

			# 计算图片位置
			y_offset = row * cell_height
			x_offset = col * img_w

			# 将图片放置到画布上
			canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w] = resized

			# 添加文件名
			filename = filenames[i]
			if len(filename) > 20:  # 如果文件名过长，进行截断
				filename = filename[:17] + '...'

			# 在图片下方显示文件名
			text_y = y_offset + img_h + 20
			text_x = x_offset + 5
			cv2.putText(canvas, filename, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

		# 添加状态栏（显示页码）
		status_text = f'Page {current_page + 1}/{num_pages} | A/D: Navigate | ESC: Exit'
		cv2.putText(canvas, status_text, (10, canvas_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

		# 显示画布
		cv2.imshow(window_name, canvas)

		# 等待按键
		key = cv2.waitKey(0)

		# 处理按键
		if key == 27:  # ESC键退出
			break
		elif key == ord('a'):  # 左箭头键 (← 键)
			current_page = (current_page - 1) % num_pages
		elif key == ord('d'):  # 右箭头键 (→ 键)
			current_page = (current_page + 1) % num_pages

	cv2.destroyWindow(window_name)


def show_image_comparison(
	images: list[np.ndarray], titles: list[str] | None = None, image_size: tuple[int, int] | None = None, window_name: str = 'Image Comparison'
) -> None:
	"""
	并排显示多张图片以进行比较

	Args:
	    images: 图片对象列表
	    titles: 对应的图片标题列表，如果为None则使用默认标题
	    image_size: 每张图片的显示大小，格式为(height, width)，如果为None则保持原始大小
	    window_name: 窗口标题
	"""
	if not images:
		print('没有图片可以显示')
		return

	# 确保标题列表长度与图片数量相同
	if titles is None:
		titles = [f'Image {i+1}' for i in range(len(images))]
	elif len(titles) < len(images):
		titles.extend([f'Image {i+1}' for i in range(len(titles), len(images))])

	# 如果指定了尺寸，则调整所有图片大小
	if image_size is not None:
		processed_images = [cv2.resize(img, (image_size[1], image_size[0])) for img in images]
	else:
		# 找出所有图片的最大高度
		max_height = max(img.shape[0] for img in images)
		processed_images = []
		for img in images:
			# 按比例调整宽度，保持高度一致
			h, w = img.shape[:2]
			new_w = int(w * max_height / h)
			processed_images.append(cv2.resize(img, (new_w, max_height)))

	# 创建标题区域的高度
	title_height = 30

	# 计算画布宽度（所有图片的宽度总和）
	canvas_width = sum(img.shape[1] for img in processed_images)
	canvas_height = processed_images[0].shape[0] + title_height

	# 创建空白画布
	canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

	# 在画布上放置图片和标题
	x_offset = 0
	for i, img in enumerate(processed_images):
		h, w = img.shape[:2]

		# 添加标题
		cv2.putText(canvas, titles[i], (x_offset + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

		# 添加图片
		canvas[title_height : title_height + h, x_offset : x_offset + w] = img
		x_offset += w

	# 显示画布
	cv2.imshow(window_name, canvas)
	cv2.waitKey(0)
	cv2.destroyWindow(window_name)


if __name__ == '__main__':
	path = open('C:/Temp/path.txt', 'r').readlines()

	path = [i.strip() for i in path]

	for i in range(len(path)):
		path[i] = rf'C:\Resources\Datasets\微笑图片标注汇总25-3-4\微笑图片标注汇总25-3-4\{path[i]}'

	show_gallery(path, (5, 5), (100, 200))  # type:ignore
