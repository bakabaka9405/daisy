from pathlib import Path
import daisy
from daisy.util import extract_by_label
from daisy.dataset.disk_dataset import DiskDataset


def load_smile(
	root: Path,
	class_sheet: Path,
	split_sheet: Path | None,
	task: int,
	*,
	split_manifest: Path | None = None,
) -> tuple[DiskDataset, DiskDataset, DiskDataset]:
	feeder = daisy.feeder.load_feeder_from_sheet(
		dataset_root=root,
		sheet_path=class_sheet,
		column=task,
		have_header=True,
		label_offset=(0 if task in [2, 3] else -1),
	)

	files, labels = feeder.fetch()

	if split_manifest is not None:
		split_mapping, _ = daisy.protocol.apply_split_manifest(
			files,
			labels,
			split_manifest,
			dataset_root=root,
			split_names=('train', 'val', 'test'),
			strict=True,
		)
		train_dataset = DiskDataset(*split_mapping['train'])
		val_dataset = DiskDataset(*split_mapping['val'])
		test_dataset = DiskDataset(*split_mapping['test'])
		return train_dataset, val_dataset, test_dataset

	if split_sheet is None:
		raise ValueError('split_sheet is required when split_manifest is not provided')

	split_feeder = daisy.feeder.load_feeder_from_sheet(
		dataset_root=root,
		sheet_path=split_sheet,
		column=task,
		have_header=True,
	)
	_, split_labels = split_feeder.fetch()
	train_files, train_labels = extract_by_label(split_labels, 1, files, labels)
	train_dataset = DiskDataset(train_files, train_labels)
	val_files, val_labels = extract_by_label(split_labels, 2, files, labels)
	val_dataset = DiskDataset(val_files, val_labels)
	test_files, test_labels = extract_by_label(split_labels, 3, files, labels)
	test_dataset = DiskDataset(test_files, test_labels)
	return train_dataset, val_dataset, test_dataset
