from typing import List, Tuple

import numpy as np
import torch


def collate_fn(batch: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
    """To be used in a torch DataLoader.

    Args:
        batch (List[np.array]): List of numpy arrays.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors as batch.
    """
    return (torch.tensor(np.array([x[0] for x in batch])), 
            torch.tensor(np.array([x[1] for x in batch])))
