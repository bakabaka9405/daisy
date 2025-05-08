from pathlib import Path
import pandas as pd
from typing import Literal


class Feeder:
	root: Path
	files: list[Path] = []
	labels: list[int] = []

	def __init__(self, files: list[Path] = [], labels: list[int] = []):
		self.files = files
		self.labels = labels

	def __len__(self) -> int:
		return len(self.files)

	def fetch(self) -> tuple[list[Path], list[int]]:
		return self.files, self.labels


def load_feeder_from_folder(root: Path | str, class_dict: dict[str, int] | None = None) -> tuple[Feeder, dict[str, int]]:
	if isinstance(root, str):
		root = Path(root)

	should_build_dict = class_dict is None
	if should_build_dict:
		class_dict = {}
	files = []
	labels = []
	for i, folder in enumerate(root.iterdir()):
		if not folder.is_dir():
			continue
		if should_build_dict:
			class_dict[folder.name] = i
			label = i
		else:
			if folder.name not in class_dict:
				raise ValueError(f'Class {folder.name} not in class_dict')
			label = class_dict[folder.name]
		for file in folder.iterdir():
			if not file.is_file():
				continue
			files.append(file)
			labels.append(label)

	return Feeder(files, labels), class_dict


def load_feeder_from_sheet(
	dataset_root: Path | str,
	sheet_path: Path | str,
	sheet_type: Literal['csv', 'excel', 'auto'] = 'auto',
	sheet_name: int | str = 0,
	have_header: bool = False,
	column: int | str = 1,
	label_offset: int = 0,
):
	if isinstance(dataset_root, str):
		dataset_root = Path(dataset_root)
	if isinstance(sheet_path, str):
		sheet_path = Path(sheet_path)
	if sheet_type not in ['csv', 'excel', 'auto']:
		raise ValueError(f'Invalid sheet_type {sheet_type}')
	if sheet_type == 'auto':
		if sheet_path.suffix == '.csv':
			sheet_type = 'csv'
		elif sheet_path.suffix == '.xlsx':
			sheet_type = 'excel'
		else:
			raise ValueError(f'Unknown file type {sheet_path.suffix}')
	if (sheet_path.suffix == '.csv' and sheet_type != 'csv') or (sheet_path.suffix == '.xlsx' and sheet_type != 'excel'):
		raise ValueError(f'File type {sheet_path.suffix} does not match sheet_type {sheet_type}')

	if sheet_type == 'csv':
		df = pd.read_csv(
			sheet_path,
			header=0 if have_header else None,
		)
	else:
		df = pd.read_excel(
			sheet_path,
			sheet_name,
			header=0 if have_header else None,
		)

	if have_header and isinstance(column, int):
		column = df.columns[column]

	return Feeder(
		list(map(lambda x: dataset_root / x, df[df.columns[0]].astype(str).tolist())),
		list(map(lambda x: x + label_offset, df[column].astype(int).tolist())),
	)


if __name__ == '__main__':
	feeder = load_feeder_from_sheet(
		Path(r'c:\Resources\Datasets\微笑图片标注汇总25-3-4\微笑图片标注汇总25-3-4'),
		Path(r'C:\Resources\Datasets\微笑图片标注汇总25-3-4\微笑图片标注汇总25-3-4.xlsx'),
		have_header=True,
		column=1,
	)
	print(feeder.fetch())
