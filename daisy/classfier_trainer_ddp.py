import daisy
import torch
from torch.optim import AdamW
from timm import create_model
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from timm.data.loader import MultiEpochsDataLoader
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from queue import Queue


def setup(rank: int, world_size: int):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	os.environ['USE_LIBUV'] = '0'
	# 初始化进程组
	torch.distributed.init_process_group('gloo', rank=rank, world_size=world_size)


def train(
	rank: int,
	world_size: int,
	model_t,
	model_args: dict,
	epochs: int,
	warmup_epochs: int,
	lr: float,
	train_dataset: daisy.dataset.IndexDataset,
	val_dataset: daisy.dataset.IndexDataset | None,
	train_batchsize: int,
	val_batchsize: int,
	comm_queue: Queue,
):
	model = model_t(**model_args).to(rank)
	model = DDP(model, device_ids=[rank])

	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
	val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if val_dataset else None

	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=train_batchsize,
		sampler=train_sampler,
		num_workers=4,
		pin_memory=True,
		drop_last=True,
	)

	val_loader = (
		MultiEpochsDataLoader(
			val_dataset,
			batch_size=val_batchsize,
			sampler=val_sampler,
			num_workers=4,
			pin_memory=True,
			drop_last=False,
		)
		if val_dataset
		else None
	)

	criterion = LabelSmoothingCrossEntropy()

	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))

	def lr_func(epoch: int):
		return min((epoch + 1) / (warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / max(1, epochs) * math.pi) + 1))

	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

	for epoch in range(epochs):
		model.train()
		losses = 0.0
		for data, target in train_loader:
			data, target = data.to(rank), target.to(rank)

			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			losses += loss.item()

		lr_scheduler.step()

		if val_dataset is None:
			comm_queue.put({'epoch': epoch, 'loss': losses / len(train_loader)})
			continue

		assert val_loader is not None, 'val_loader should not be None when val_dataset is provided'
		y_pred = []
		y_true = []
		with torch.no_grad():
			model.eval()
			for data, target in val_loader:
				data, target = data.to(rank), target.to(rank)
				output = model(data)
				predicted = torch.argmax(output, dim=1)
				y_pred.extend(predicted.cpu().numpy())
				y_true.extend(target.cpu().numpy())

		comm_queue.put({'epoch': epoch, 'loss': losses / len(train_loader), 'y_pred': y_pred, 'y_true': y_true})


def train_classfier_ddp(
	model_t,
	model_args: dict,
	epochs: int,
	warmup_epochs: int,
	lr: float,
	train_dataset: daisy.dataset.IndexDataset,
	val_dataset: daisy.dataset.IndexDataset | None,
	train_batchsize: int,
	val_batchsize: int,
	comm_queue: Queue,
):
	n_gpus = torch.cuda.device_count()
	mp.start_processes(  # type:ignore
		train,
		args=(
			n_gpus,
			model_t,
			model_args,
			epochs,
			warmup_epochs,
			lr,
			train_dataset,
			val_dataset,
			train_batchsize,
			val_batchsize,
			comm_queue,
		),
		nprocs=n_gpus,
		join=True,
	)
