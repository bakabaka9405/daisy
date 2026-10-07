"""运行溯源：把各相关 git 仓库的工作区记录为隐藏 commit

做法与 DVC experiments 相同：用临时 index 将工作区（含未提交改动与未被忽略的未跟踪文件）写成以 HEAD 为父节点的 commit，
挂在 refs/daisy/runs/ 下防止被垃圾回收；不改动分支、暂存区与工作区。复现时 `git worktree add <目录> <commit>`。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import icontract

RUNS_REF_PREFIX = 'refs/daisy/runs/'


@dataclass(slots=True, frozen=True)
class RepositorySnapshot:
	"""单个仓库的工作区快照"""

	root: Path
	head: str
	commit: str  # 工作区快照 commit，父节点为 head
	tree: str
	dirty: bool  # 工作区与 HEAD 不同（含未跟踪文件）


@dataclass(slots=True, frozen=True)
class SourceSnapshot:
	"""一次运行涉及的全部仓库快照，各仓库使用同一个引用名"""

	ref: str
	repositories: tuple[RepositorySnapshot, ...]

	def to_json(self) -> dict[str, object]:
		return {
			'ref': self.ref,
			'repositories': [
				{'root': str(r.root), 'head': r.head, 'commit': r.commit, 'dirty': r.dirty} for r in self.repositories
			],
		}


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
	return subprocess.run(
		['git', '-C', str(root), *args], capture_output=True, text=True, encoding='utf-8', check=True, env=env
	).stdout.strip()


def repository_root(path: Path) -> Path:
	"""path（文件或目录）所属 git 仓库的根目录；不在仓库中时 git 报错"""
	return Path(_git(path if path.is_dir() else path.parent, 'rev-parse', '--show-toplevel')).resolve()


def worktree_tree(root: Path) -> str:
	"""工作区（含未跟踪且未被忽略的文件）对应的 tree 哈希；在 index 的副本上操作，复用其中的文件状态缓存"""
	index = Path(_git(root, 'rev-parse', '--path-format=absolute', '--git-path', 'index'))
	with tempfile.TemporaryDirectory() as tmp:
		temp_index = Path(tmp) / 'index'
		shutil.copyfile(index, temp_index)
		env = {**os.environ, 'GIT_INDEX_FILE': str(temp_index)}
		_git(root, 'add', '--all', env=env)
		return _git(root, 'write-tree', env=env)


@icontract.require(lambda paths: len(paths) > 0)
@icontract.ensure(lambda result: result.ref.startswith(RUNS_REF_PREFIX))
def snapshot_sources(paths: list[Path], message: str) -> SourceSnapshot:
	"""为 paths 所属的各仓库生成工作区快照 commit，并挂在同名引用 refs/daisy/runs/<时间>-<进程号> 下"""
	ref = f'{RUNS_REF_PREFIX}{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}'
	repositories = []
	for root in dict.fromkeys(repository_root(p) for p in paths):
		head = _git(root, 'rev-parse', 'HEAD')
		tree = worktree_tree(root)
		commit = _git(root, 'commit-tree', tree, '-p', head, '-m', message)
		_git(root, 'update-ref', ref, commit)
		dirty = tree != _git(root, 'rev-parse', f'{head}^{{tree}}')
		repositories.append(RepositorySnapshot(root=root, head=head, commit=commit, tree=tree, dirty=dirty))
	return SourceSnapshot(ref=ref, repositories=tuple(repositories))


def changed_repositories(snapshot: SourceSnapshot) -> list[Path]:
	"""工作区已与快照不同的仓库（运行期间源码被修改）"""
	return [r.root for r in snapshot.repositories if worktree_tree(r.root) != r.tree]


def _generated_code_dirs() -> list[Path]:
	"""torch.compile 在运行时生成并导入的代码缓存目录（inductor，以及已加载 triton 时的 triton 缓存）

	其中的模块由已加载的源码在运行时生成，不是独立的源码。
	inductor 编译期间会把 TRITON_CACHE_DIR 指向自己的缓存目录，而 triton 的驱动工具模块（cuda_utils 等）可能在此之前
	已从默认目录（~/.triton/cache）加载，因此默认目录与当前目录都要计入。
	"""
	from torch._inductor.runtime.cache_dir_utils import cache_dir

	dirs = [Path(cache_dir())]
	if 'triton' in sys.modules:
		import triton.knobs

		dirs += [Path(triton.knobs.cache.get_triton_dir('cache')), Path(triton.knobs.cache.dir)]
	return [d.resolve() for d in dirs]


def modules_outside_snapshot(snapshot: SourceSnapshot, modules: Iterable[tuple[str, object]]) -> list[str]:
	"""源文件既不在任何仓库快照内、也不属于 Python 安装（标准库与 site-packages）或 torch.compile 生成代码的已加载模块

	仓库内被 .gitignore 忽略的文件（如 tmp/）不在快照中，同样计入；嵌套仓库按最内层仓库判断。
	"""
	repositories = sorted(snapshot.repositories, key=lambda r: len(r.root.parts), reverse=True)
	tracked = {
		r.root: {os.path.normcase(r.root / p) for p in _git(r.root, 'ls-tree', '-r', '--name-only', '-z', r.tree).split('\0') if p}
		for r in repositories
	}
	installs = [Path(p).resolve() for p in dict.fromkeys((sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix))]
	installs += _generated_code_dirs()
	outside = []
	for name, module in modules:
		file = getattr(module, '__file__', None)
		if file is None or not Path(file).is_absolute():
			continue
		path = Path(file).resolve()
		# Windows 扩展长度路径（\\?\C:\...，如 triton 加载的扩展模块）去掉前缀后才能与普通路径比较
		if str(path).startswith('\\\\?\\') and not str(path).startswith('\\\\?\\UNC\\'):
			path = Path(str(path)[4:])
		owner = next((r.root for r in repositories if path.is_relative_to(r.root)), None)
		if owner is None:
			if not any(path.is_relative_to(p) for p in installs):
				outside.append(f'{name}: {path}')
		elif os.path.normcase(path) not in tracked[owner]:
			outside.append(f'{name}: {path}')
	return outside


__all__ = [
	'RUNS_REF_PREFIX',
	'RepositorySnapshot',
	'SourceSnapshot',
	'changed_repositories',
	'modules_outside_snapshot',
	'repository_root',
	'snapshot_sources',
	'worktree_tree',
]
