from typing import Dict, List

import numpy as np
import torch


def collate_fn(batch: List[np.array]) -> Dict[str, torch.Tensor]:
    """To be used in a torch DataLoader.

    Args:
        batch (List[np.array]): List of numpy arrays.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary of tensors as batch.
    """
    return {
        "sensors": torch.tensor(np.array([x[0] for x in batch])),
        "weather": torch.tensor(np.array([x[1] for x in batch])),
        "targets": torch.tensor(np.array([x[2] for x in batch])),
    }
