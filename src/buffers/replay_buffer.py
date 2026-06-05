import torch
from torch.utils.data import IterableDataset, DataLoader

import numpy as np


"""

Sampling with replacement (RL-style):

Utilities for creating replay-style data loaders and an optional CUDA prefetcher.

Provides:
- ReplayBufferLoader: DataLoader wrapper to produce a fixed number of sampled batches.
- ReplayBufferDataset: IterableDataset that samples from a dataset with replacement.
- CUDAPrefetcher: Asynchronous GPU prefetcher that moves batches to CUDA in a background stream.

Notes on conventions:
- Input dataset is expected to provide dict items with keys "images" and "labels".
- Produced batches are stacked tensors where the sample dimension is first.

"""

class ReplayBufferLoader(DataLoader):

    """

    Convenience DataLoader that wraps a ReplayBufferDataset.

    Parameters
    - dataset: torch.utils.data.Dataset
        Source dataset whose elements are dicts with "images" and "labels".
    - batch_size: int
        Number of samples per batch to be produced by the underlying ReplayBufferDataset.
    - num_batches: int
        Total number of batches the loader will produce per epoch/iteration.
    - **kwargs:
        Additional DataLoader kwargs passed to the parent DataLoader constructor.

    Behavior
    - Instantiates a ReplayBufferDataset and then builds a DataLoader with batch_size=None
      because the dataset already returns full batches.
    - The DataLoader is configured with pin_memory=True and num_workers=1 by default here,
      but kwargs can override additional DataLoader settings.

    """

    def __init__(self, dataset, batch_size, num_batches, **kwargs):

        """

        Instantiate the loader, wrapping a ReplayBufferDataset.

        Args:
        - dataset: The source dataset.
        - batch_size: Number of samples per batch.
        - num_batches: Total number of batches to produce.
        - **kwargs: Additional arguments for DataLoader.

        """

        replay_ds = ReplayBufferDataset(dataset, batch_size, num_batches)
        super().__init__(replay_ds, batch_size=None, **kwargs, pin_memory=True, num_workers=12, prefetch_factor=16)

    def feedback(self, *args, **kwargs):
        pass


class ReplayBufferDataset(IterableDataset):

    """

    Wraps any torch Dataset and produces random samples with replacement.

    Purpose
    - Produce exactly `num_batches` batches of size `batch_size` per iteration, sampling
      indices uniformly with replacement from the provided dataset. This is useful for
      replay buffers or randomized mini-batch generation that doesn't exhaust the dataset.

    Constructor Parameters
    - dataset: Sequence-like dataset supporting len() and __getitem__ that returns dicts with
      keys "images" and "labels".
    - batch_size: int
        Number of samples per produced batch.
    - num_batches: int
        Number of batches to produce when iterating over this IterableDataset.
    - device: str (default "cpu")
        Device used for the internal torch.Generator to produce indices if using multiple workers.

    Key methods
    - __iter__:
        Produces `num_batches` tensors of shape [batch_size, C+1, H, W] where the dataset
        items are concatenated as implemented below (images + label channel).
        Uses a per-worker torch.Generator seeded by worker_info.seed if available to ensure
        independent RNG streams across DataLoader workers.
    - __len__:
        Returns the configured number of batches (num_batches).

    Internal representation details
    - The constructor pre-stacks dataset items into a single tensor `self.data` for fast,
      vectorized indexing:
        self.data shape: [N, C+1, H, W] (N == len(dataset))
      This accelerates batch gathering via advanced indexing.

    """

    """
    Replay-style iterable dataset using NumPy indices.

    - Samples with replacement
    - Vectorized NumPy indexing
    - Worker-safe RNG
    """

    def __init__(self, dataset, batch_size, num_batches, device="cpu"):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n = len(dataset)

        # Load backing storage ONCE (NPZ, Zarr, etc.)
        self.data = self.dataset.source_adapter.load(dataset.split)

        self.images = self.data["image"]        # shape [N, H, W, C]
        self.labels = self.data["path_label"]

        self.num_batches = num_batches

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
            seed = None
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            seed = worker_info.seed

        rng = np.random.default_rng(seed)

        for _ in range(worker_id, self.num_batches, num_workers):
            idx = rng.integers(
                low=0,
                high=self.n,
                size=self.batch_size,
                dtype=np.int64,
            )

            imgs_np = np.asarray(self.images[idx])
            labels_np = np.asarray(self.labels[idx])

            imgs = (
                torch.from_numpy(imgs_np)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )

            yield {
                "images": imgs,
                "labels": torch.from_numpy(labels_np),
                "idx": torch.from_numpy(idx), 
                "weights": torch.ones(imgs.shape[0], dtype=torch.float32)
            }

    def __len__(self):
        return self.num_batches
    

class CUDAPrefetcher:

    """

    Wrap a PyTorch DataLoader to asynchronously prefetch batches to GPU.

    Purpose
    - Move batches returned by a CPU DataLoader onto a CUDA device asynchronously using
      a dedicated CUDA stream. This can hide data transfer latency by overlapping
      host-to-device copies with CPU-side batch preparation.

    Constructor Parameters
    - loader: Iterable or DataLoader
        Source loader producing batches (can be the ReplayBufferLoader or any DataLoader).
    - device: str or torch.device (default "cuda")
        CUDA device to which batches will be moved.

    Methods
    - _move_to_device(batch): Recursively moves a batch (tensor/list/tuple/dict) to self.device.
    - __iter__(): Yields batches already moved to CUDA. Internally uses a background
      torch.cuda.Stream to prefetch the next batch while the current batch is being consumed.
    - __len__(): Returns len(loader) if available.

    Detailed behavior in __iter__
    1. Construct an iterator over the underlying loader.
    2. Preload the first batch into the background stream (next_batch).
    3. For each subsequent batch:
       - wait for the prefetch stream to finish moving the previous next_batch,
       - yield the already-prefetched batch (current_batch),
       - launch an asynchronous transfer of the new batch into the background stream.
    4. After loop, yield the final prefetched batch.

    """

    def __init__(self, loader, device="cuda"):

        """

        Initialize the CUDA prefetcher.

        Args:
        - loader: The source loader.
        - device: The CUDA device.

        """

        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def _move_to_device(self, batch):

        """

        Recursively move tensors in `batch` to self.device using non_blocking=True where possible.

        Accepts tensors, lists/tuples of tensors, and dictionaries of tensors. Non-tensor objects
        are returned unchanged.

        """

        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return [self._move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        else:
            return batch  # non-tensor data (OK to leave on CPU)

    def __iter__(self):

        """

        Iterate over the loader, yielding batches that have been moved to CUDA.

        Notes on synchronization:
        - Uses torch.cuda.Stream for asynchronous transfers.
        - Before consuming a batch moved on the prefetch stream, current_stream waits on that stream,
          ensuring memory is ready for computation on the default stream.

        """

        loader_iter = iter(self.loader)

        # Preload the first batch
        with torch.cuda.stream(self.stream):
            try:
                next_batch = self._move_to_device(next(loader_iter))
            except StopIteration:
                return

        for batch in loader_iter:
            # Wait for previous load to finish
            torch.cuda.current_stream().wait_stream(self.stream)

            # Consume the preloaded batch
            current_batch = next_batch

            # Asynchronously prefetch the next batch
            with torch.cuda.stream(self.stream):
                next_batch = self._move_to_device(batch)

            yield current_batch

        # Yield the final batch
        torch.cuda.current_stream().wait_stream(self.stream)
        yield next_batch
        
    def __len__(self):

        """

        Proxy to the underlying loader length if defined.

        """

        return len(self.loader)
