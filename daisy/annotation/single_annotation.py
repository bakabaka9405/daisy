import pandas
from pandas import DataFrame
from collections.abc import Callable


class SingleAnnotation:
	name: list[str]
	label: list[int]
	name_map: dict[str, int]

	def __init__(self):
		self.name = []
		self.label = []

	def loadFromDataFrame(self, df: DataFrame):
		self.name = df[0].tolist()
		self.label = df[1].tolist()
		self.name_map = {name: label for name, label in zip(self.name, self.label)}

	def loadFromExcel(self, path: str, sheetName: int | str = 0):
		df = pandas.read_excel(path, sheet_name=sheetName)
		self.loadFromDataFrame(df)

	def loadFromCsv(self, path: str):
		df = pandas.read_csv(path, header=None)
		self.loadFromDataFrame(df)

	def __getitem__(self, name: str) -> int:
		return self.name_map[name]

	def filter(self, fn: Callable[[str, int], bool]) -> list[str]:
		return [i for i, j in zip(self.name, self.label) if fn(i, j)]
