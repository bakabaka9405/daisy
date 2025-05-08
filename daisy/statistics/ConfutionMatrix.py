import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np

if __name__ == '__main__':
	cm = np.array([[23, 18, 3, 1, 0], [10, 94, 58, 9, 1], [2, 32, 249, 95, 1], [0, 6, 78, 280, 37], [0, 0, 5, 47, 103]])
	plt.figure(figsize=(8, 6))  # 可选: 设置图像大小
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels='auto', yticklabels='auto')
	plt.title('Confusion Matrix\nresnet34 task 1')
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.tight_layout()  # 调整布局，防止标签重叠
	plt.show()
