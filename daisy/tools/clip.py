import cv2
import os
import numpy as np
from pathlib import Path

def cv_imread(file_path):
	cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
	return cv_img

class ImageClipper:
    def __init__(self, input_folder, output_folder, max_display_size=800):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.max_display_size = max_display_size
        
        # 创建输出文件夹
        self.output_folder.mkdir(exist_ok=True)
        
        # 获取所有图片文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_files.extend(self.input_folder.glob(ext))
        
        self.current_index = 0
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 裁剪相关变量
        self.points = []  # 存储两个端点
        self.crop_rect = None
    
    def is_current_file_cropped(self):
        """检查当前文件是否已被裁剪"""
        if self.image_files:
            input_file = self.image_files[self.current_index]
            output_file = self.output_folder / input_file.name
            return output_file.exists()
        return False

    def load_image(self, index):
        """加载指定索引的图片"""
        if 0 <= index < len(self.image_files):
            self.current_index = index
            self.original_image = cv_imread(str(self.image_files[index]))
            if self.original_image is not None:
                self.prepare_display_image()
                self.reset_crop()
                return True
        return False
    
    def prepare_display_image(self):
        """准备显示图片（等比缩放）"""
        if self.original_image is None:
            return
        
        height, width = self.original_image.shape[:2]
        
        # 计算缩放比例
        if width > self.max_display_size or height > self.max_display_size:
            scale_w = self.max_display_size / width
            scale_h = self.max_display_size / height
            self.scale_factor = min(scale_w, scale_h)
        else:
            self.scale_factor = 1.0
        
        # 计算新尺寸
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        
        # 缩放图片
        self.display_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # 计算居中显示的偏移量
        display_width = max(self.max_display_size, new_width)
        display_height = max(self.max_display_size, new_height)
        self.offset_x = (display_width - new_width) // 2
        self.offset_y = (display_height - new_height) // 2
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 转换为原图坐标
            orig_x = int((x - self.offset_x) / self.scale_factor)
            orig_y = int((y - self.offset_y) / self.scale_factor)
            
            # 确保坐标在图片范围内
            orig_x = max(0, min(orig_x, self.original_image.shape[1] - 1))
            orig_y = max(0, min(orig_y, self.original_image.shape[0] - 1))
            
            if len(self.points) < 2:
                self.points.append((orig_x, orig_y))
                if len(self.points) == 2:
                    self.update_crop_rect()
            else:
                # 重新开始选择
                self.points = [(orig_x, orig_y)]
                self.crop_rect = None
    
    def update_crop_rect(self):
        """更新裁剪矩形"""
        if len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            
            # 确保左上角和右下角
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            self.crop_rect = (left, top, right, bottom)
    
    def reset_crop(self):
        """重置裁剪选择"""
        self.points = []
        self.crop_rect = None
    
    def draw_overlay(self):
        """绘制覆盖层（红点、裁剪框、信息文字）"""
        if self.display_image is None:
            return None
        
        # 创建显示画布
        canvas_height = self.max_display_size + 100  # 额外空间显示信息
        canvas_width = self.max_display_size
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 将图片放到画布上
        img_height, img_width = self.display_image.shape[:2]
        y_start = self.offset_y
        y_end = y_start + img_height
        x_start = self.offset_x
        x_end = x_start + img_width
        canvas[y_start:y_end, x_start:x_end] = self.display_image
        
        # 绘制已选择的点
        for point in self.points:
            orig_x, orig_y = point
            display_x = int(orig_x * self.scale_factor + self.offset_x)
            display_y = int(orig_y * self.scale_factor + self.offset_y)
            cv2.circle(canvas, (display_x, display_y), 5, (0, 0, 255), -1)
        
        # 绘制裁剪框
        if self.crop_rect:
            left, top, right, bottom = self.crop_rect
            display_left = int(left * self.scale_factor + self.offset_x)
            display_top = int(top * self.scale_factor + self.offset_y)
            display_right = int(right * self.scale_factor + self.offset_x)
            display_bottom = int(bottom * self.scale_factor + self.offset_y)
            
            cv2.rectangle(canvas, (display_left, display_top), 
                         (display_right, display_bottom), (0, 255, 0), 2)
            
            # 显示裁剪区域的长宽比
            crop_width = right - left
            crop_height = bottom - top
            if crop_height > 0:
                aspect_ratio = crop_width / crop_height
                ratio_text = f"Crop Ratio: {aspect_ratio:.2f} ({crop_width}x{crop_height})"
                cv2.putText(canvas, ratio_text, (10, canvas_height - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示原图长宽比
        if self.original_image is not None:
            orig_height, orig_width = self.original_image.shape[:2]
            orig_ratio = orig_width / orig_height
            orig_text = f"Original Ratio: {orig_ratio:.2f} ({orig_width}x{orig_height})"
            cv2.putText(canvas, orig_text, (10, canvas_height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 显示当前文件信息和裁剪状态
        if self.image_files:
            filename = self.image_files[self.current_index].name
            cropped_status = "[已裁剪]" if self.is_current_file_cropped() else "[未裁剪]"
            status_color = (0, 255, 0) if self.is_current_file_cropped() else (0, 255, 255)
            
            file_text = f"File: {filename} ({self.current_index + 1}/{len(self.image_files)}) {cropped_status}"
            cv2.putText(canvas, file_text, (10, canvas_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # 显示操作提示
        help_text = "Click: Select corners | S: Save | R: Reset | A/D: Prev/Next | Q: Quit"
        cv2.putText(canvas, help_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return canvas
    
    def crop_and_save(self):
        """裁剪并保存图片"""
        if self.crop_rect is None or self.original_image is None:
            print("请先选择裁剪区域")
            return False
        
        left, top, right, bottom = self.crop_rect
        cropped_image = self.original_image[top:bottom, left:right]
        
        # 保存文件
        input_file = self.image_files[self.current_index]
        output_file = self.output_folder / input_file.name
        
        # 使用cv2.imencode处理中文路径
        ext = input_file.suffix.lower()
        if ext == '.jpg' or ext == '.jpeg':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            ret, encimg = cv2.imencode('.jpg', cropped_image, encode_param)
        elif ext == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            ret, encimg = cv2.imencode('.png', cropped_image, encode_param)
        else:
            # 默认保存为jpg格式
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            ret, encimg = cv2.imencode('.jpg', cropped_image, encode_param)
            output_file = output_file.with_suffix('.jpg')
        
        if ret:
            try:
                with open(str(output_file), 'wb') as f:
                    f.write(encimg.tobytes())
                print(f"已保存裁剪图片: {output_file}")
                return True
            except Exception as e:
                print(f"保存失败: {e}")
                return False
        else:
            print("图片编码失败")
            return False
    
    def run(self):
        """运行裁剪工具"""
        if not self.image_files:
            print(f"在文件夹 {self.input_folder} 中未找到图片文件")
            return
        
        # 加载第一张图片
        if not self.load_image(0):
            print("无法加载图片")
            return
        
        cv2.namedWindow('Image Clipper', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Image Clipper', self.mouse_callback)
        
        while True:
            # 绘制界面
            display = self.draw_overlay()
            if display is not None:
                cv2.imshow('Image Clipper', display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                break
            elif key == ord('s'):  # Enter键 - 确认裁剪
                self.crop_and_save()
            elif key == ord('r'):  # R键 - 重置裁剪
                self.reset_crop()
                print("已重置裁剪选择")
            elif key == ord('a'):  # A键 - 上一张图片
                if self.current_index > 0:
                    self.load_image(self.current_index - 1)
            elif key == ord('d'):  # D键 - 下一张图片
                if self.current_index < len(self.image_files) - 1:
                    self.load_image(self.current_index + 1)
        
        cv2.destroyAllWindows()


def main():
    """主函数"""
    # 设置输入输出文件夹路径
    input_folder = input("请输入图片文件夹路径: ").strip()
    if not input_folder:
        input_folder = "input_images"  # 默认输入文件夹
    
    output_folder = input("请输入输出文件夹路径: ").strip()
    if not output_folder:
        output_folder = "cropped_images"  # 默认输出文件夹
    
    # 创建并运行裁剪工具
    clipper = ImageClipper(input_folder, output_folder)
    clipper.run()


if __name__ == "__main__":
    main()
