from setuptools import setup, find_packages

setup(
	name='daisy',
	version='0.1.0',  # 版本号
	description='none',
	author='bakabaka',
	author_email='none',
	packages=['daisy'],  # 自动查找你的包内的模块
	install_requires=['torch>=2.6.0', 'torchvision>=0.21.0', 'timm>=1.0.0', 'numpy>=2.1.2'],
	classifiers=[
		'Development Status :: 3 - Alpha',  # 项目的开发状态 (Alpha, Beta, Production/Stable 等)
		'Intended Audience :: Developers',
		'License :: OSI Approved :: MIT License',  # 许可证类型
		'Programming Language :: Python :: 3.12',
	],
)
