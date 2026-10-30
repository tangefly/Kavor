import bisect
from collections import OrderedDict
from dataclasses import dataclass

import torch

from kavor.utils.logger import logger


@dataclass(frozen=True)
class MMFeatureSpec:
    """一份缓存行的布局:1 份 merged + num_deepstack 份 deepstack 特征。"""

    hidden: int
    num_deepstack: int
    dtype: torch.dtype

    @property
    def row_bytes(self) -> int:
        return self.hidden * self.dtype.itemsize * (1 + self.num_deepstack)


@dataclass
class MMCacheEntry:
    start: int
    rows: int
    ref_count: int = 0


class MMCache:
    """跨请求复用的视觉 embedding 池(按媒体项 LRU)。

    每个 TP rank 各持一份副本。所有决策只由 host 侧状态(项 id、大小、插入/访问
    顺序)决定:各 rank 收到的请求流和媒体清单完全相同,决策必然一致。任何依赖
    GPU 时序或随机性的逻辑都会让 rank 之间悄悄错位——命中/未命中不同,送进文本
    模型的 embedding 就不同,而且逐出后可能读到被覆盖的行。不要加。

    池按「行」管理,一行是一个 merge 之后的视觉 token:merged 特征一份,Qwen3-VL
    的 deepstack 特征再各占一份(见 MMFeatureSpec)。
    """

    def __init__(self, spec: MMFeatureSpec, num_rows: int, device: str = "cuda"):
        assert num_rows > 0
        self.spec = spec
        self.num_rows = num_rows
        self.merged = torch.empty(num_rows, spec.hidden, dtype=spec.dtype, device=device)
        self.deepstack = [
            torch.empty(num_rows, spec.hidden, dtype=spec.dtype, device=device)
            for _ in range(spec.num_deepstack)
        ]
        # OrderedDict 的顺序就是 LRU 顺序:刚访问/插入的挪到队尾,逐出从队头找
        self.entries: OrderedDict[bytes, MMCacheEntry] = OrderedDict()
        self.free: list[list[int]] = [[0, num_rows]]      # [start, rows],按 start 有序
        self.hits = self.misses = self.evictions = 0
        self.hit_rows = self.miss_rows = 0

    @classmethod
    def from_config(cls, config, device: str = "cuda") -> "MMCache | None":
        """按配置建池;未启用、没有视觉塔或大小为 0 时返回 None。"""
        vision_config = config.vision_config
        if vision_config is None or not config.enable_mm_cache or config.mm_cache_size_gb <= 0:
            return None
        spec = MMFeatureSpec(
            hidden=int(getattr(vision_config, "out_hidden_size",
                               config.text_config.hidden_size)),
            num_deepstack=len(getattr(vision_config, "deepstack_visual_indexes", ()) or ()),
            dtype=config.text_config.dtype,
        )
        num_rows = int(config.mm_cache_size_gb * 2**30) // spec.row_bytes
        if num_rows <= 0:
            raise ValueError(
                f"mm_cache_size_gb={config.mm_cache_size_gb} 放不下一行 embedding"
                f"({spec.row_bytes} B):调大 --mm-cache-size-gb")
        return cls(spec, num_rows, device)

    # 生命周期

    def acquire(self, hash_id: bytes) -> MMCacheEntry | None:
        """命中则 pin 住并返回条目,未命中返回 None(由 put 负责写入)。"""
        entry = self.entries.get(hash_id)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.hit_rows += entry.rows
        entry.ref_count += 1
        self.entries.move_to_end(hash_id)
        return entry

    def release(self, hash_id: bytes):
        entry = self.entries.get(hash_id)
        if entry is not None:
            entry.ref_count = max(0, entry.ref_count - 1)

    def peek(self, hash_id: bytes) -> bool:
        """只查是否存在,不 pin、不计命中统计(给日志/观测用)。"""
        return hash_id in self.entries

    def put(self, hash_id: bytes, merged: torch.Tensor,
            deepstack: list[torch.Tensor]) -> MMCacheEntry | None:
        """写入一个新算出的媒体,成功时 ref_count 从 1 开始(调用方持有 pin)。

        放得下就逐出最老的未 pin 条目腾位置;单个媒体比整池还大时返回 None,
        调用方直接用这份张量、不进缓存。
        """
        rows = int(merged.shape[0])
        self.miss_rows += rows
        assert len(deepstack) == len(self.deepstack), (
            f"写入 {len(deepstack)} 份 deepstack 特征,池按 {len(self.deepstack)} 份规划")
        if rows <= 0 or rows > self.num_rows:
            return None
        existing = self.entries.get(hash_id)
        if existing is not None:
            # 同一步里同一个媒体出现两次:复用第一次写入的结果
            existing.ref_count += 1
            self.entries.move_to_end(hash_id)
            return existing
        start = self._reserve(rows)
        if start is None:
            return None
        self.merged[start:start + rows].copy_(merged.to(self.merged.dtype))
        for buf, src in zip(self.deepstack, deepstack):
            buf[start:start + rows].copy_(src.to(buf.dtype))
        entry = MMCacheEntry(start, rows, ref_count=1)
        self.entries[hash_id] = entry
        return entry

    def read(self, hash_id: bytes, begin: int = 0,
             count: int | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """返回池内 [begin, begin+count) 行的视图(调用方必须持有 pin)。"""
        entry = self.entries[hash_id]
        count = entry.rows if count is None else count
        assert 0 <= begin and begin + count <= entry.rows, (
            f"mm cache 读取越界: begin={begin}, count={count}, rows={entry.rows}")
        start = entry.start + begin
        return (self.merged[start:start + count],
                [buf[start:start + count] for buf in self.deepstack])

    # 分配

    def _reserve(self, rows: int) -> int | None:
        start = self._take_free(rows)
        if start is not None:
            return start
        while True:
            victim = next(
                (hash_id for hash_id, entry in self.entries.items() if entry.ref_count == 0),
                None)
            if victim is None:
                return None
            entry = self.entries.pop(victim)
            self._give_back(entry.start, entry.rows)
            self.evictions += 1
            start = self._take_free(rows)
            if start is not None:
                return start

    def _take_free(self, rows: int) -> int | None:
        for index, (start, size) in enumerate(self.free):
            if size < rows:
                continue
            if size == rows:
                self.free.pop(index)
            else:
                self.free[index] = [start + rows, size - rows]
            return start
        return None

    def _give_back(self, start: int, rows: int):
        index = bisect.bisect_left(self.free, [start, rows])
        self.free.insert(index, [start, rows])
        if index + 1 < len(self.free) and start + rows == self.free[index + 1][0]:
            self.free[index][1] += self.free[index + 1][1]
            self.free.pop(index + 1)
        if index > 0 and self.free[index - 1][0] + self.free[index - 1][1] == start:
            self.free[index - 1][1] += self.free[index][1]
            self.free.pop(index)

    # 统计

    @property
    def used_rows(self) -> int:
        return self.num_rows - sum(size for _, size in self.free)

    def log_summary(self):
        logger.info(
            "mm cache: %.2f GiB = %d 行 × %d hidden × %d 份特征(%s),每行 %d B",
            self.num_rows * self.spec.row_bytes / 2**30, self.num_rows, self.spec.hidden,
            1 + self.spec.num_deepstack, self.spec.dtype, self.spec.row_bytes)

    def log_step(self):
        logger.info(
            "mm cache: 累计命中 %d/%d 项 (%d/%d token),逐出 %d,占用 %d/%d 行",
            self.hits, self.hits + self.misses, self.hit_rows,
            self.hit_rows + self.miss_rows, self.evictions,
            self.used_rows, self.num_rows)
