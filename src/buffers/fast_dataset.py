import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np


"""

Vectorized dataset. Use this one especially for very large datasets.

"""

class FastDataloader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=16, **kwargs):
        replay_ds = FastDataset(dataset, batch_size)
        super().__init__(
            replay_ds,
            batch_size=None,   # dataset yields full batches
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=16,
            **kwargs,
        )


class FastDataset(IterableDataset):
    """
    Replay-style iterable dataset using NumPy indices.

    - Samples with replacement
    - Vectorized NumPy indexing
    - Worker-safe RNG
    """

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n = len(dataset)

        # Load backing storage ONCE (NPZ, Zarr, etc.)
        self.data = self.dataset.source_adapter.load(dataset.split)

        self.images = self.data["image"]        # shape [N, H, W, C]
        self.labels = self.data["path_label"]   # shape [N, H, W]

        self.num_batches = self.n // self.batch_size

        if hasattr(self.data, "keys") and "ppm" in self.data.keys():
            self.ppms = self.data["ppm"]        # shape [N, H, W]
            self.deliver_ppms = True
        else: 
            self.deliver_ppms = False

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

            imgs_np = np.array(self.images[idx])
            labels_np = np.array(self.labels[idx])

            imgs = (
                torch.from_numpy(imgs_np)
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )

            if self.deliver_ppms:
                ppms_np = np.array(self.ppms[idx])
                yield {
                    "images": imgs,
                    "labels": torch.from_numpy(labels_np),
                    "unnormalized_image": torch.from_numpy(imgs_np)
                        .to(dtype=torch.uint8),
                    "ppms": torch.from_numpy(ppms_np)
                }
            else:
                yield {
                    "images": imgs,
                    "labels": torch.from_numpy(labels_np),
                    "unnormalized_image": torch.from_numpy(imgs_np)
                        .to(dtype=torch.uint8)
                }

    def __len__(self):
        return self.n // self.batch_size
