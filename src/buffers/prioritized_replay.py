import torch
from torch.utils.data import IterableDataset, DataLoader


"""

Utilities for creating replay-style data loaders with prioritiezd replay (sampling with replacement).

"""

class PrioritizedReplayBufferLoader(DataLoader):

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

    def __init__(
        self, 
        dataset, 
        batch_size, 
        num_batches, 
        max_reward: float = 1.0,
        alpha: float = 0.7,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        eps: float = 1e-3,
        **kwargs
    ):

        """

        Instantiate the loader, wrapping a ReplayBufferDataset.

        Args:
        - dataset: The source dataset.
        - batch_size: Number of samples per batch.
        - num_batches: Total number of batches to produce.
        - **kwargs: Additional arguments for DataLoader.

        """

        self.replay_ds = PrioritizedReplayBufferDataset(
            dataset=dataset, 
            batch_size=batch_size,
            num_batches=num_batches,
            max_reward=max_reward,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment,
            eps=eps
        )

        super().__init__(self.replay_ds, batch_size=None, **kwargs, pin_memory=True, num_workers=1)

    def feedback(self, idx: torch.Tensor, rewards: torch.Tensor):
        self.replay_ds.feedback(idx, rewards)


class PrioritizedReplayBufferDataset(IterableDataset):

    def __init__(
        self,
        dataset,
        batch_size,
        num_batches,
        max_reward: float = 1.0,
        alpha: float = 0.7,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        eps: float = 1e-3,
        device="cpu",
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.n = len(dataset)
        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.max_reward = max_reward

        self.l2 = torch.compile(lambda x: (x - max_reward) ** 2)

        # Load backing storage ONCE (NPZ, Zarr, etc.)
        self.data = self.dataset.source_adapter.load(dataset.split)

        self.images = self.data["image"]        # shape [B, H, W, C]

        # Initialize priorities uniformly
        self.priorities = torch.ones(self.n, dtype=torch.float32)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # torch.Generator ensures independent worker streams
            g = torch.Generator(device=self.device)
            g.manual_seed(worker_info.seed)
        else:
            g = torch.Generator(device=self.device)

        for _ in range(self.num_batches):
            probs = self.priorities / self.priorities.sum()

            idx = torch.multinomial(
                probs,
                num_samples=self.batch_size,
                replacement=True,
                generator=g,
            )

            # Importance sampling weights
            with torch.no_grad():
                p = probs[idx]
                weights = (self.n * p).pow(-self.beta)
                weights /= weights.max()  # normalize

            imgs = (
                torch.from_numpy(self.images[idx])
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )

            yield {
                "images": imgs,
                "idx": idx, 
                "weights": weights
            }

    def feedback(self, idx: torch.Tensor, rewards: torch.Tensor):
        """
        Update priorities for sampled transitions.

        Args:
        - idx: indices of sampled items, shape [B]
        - rewards: observed rewards, shape [B]
        """
        
        with torch.no_grad():
            # Compute loss
            loss = self.l2(rewards)

            # Convert loss to priority
            new_priorities = (loss + self.eps).pow(self.alpha)

            # Hard overwrite priorities
            self.priorities[idx] = new_priorities.to(device=self.device)

            # Anneal beta toward 1
            self.beta = min(1.0, self.beta + self.beta_increment)

    def __len__(self):
        return self.num_batches
