"""RC2 learning maps adapted to this project's ignore_index=-100 convention."""

import numpy as np

from .semantickitti_utils import LEARNING_MAP_7, LEARNING_MAP_19
from .semanticposs_utils import LEARNING_MAP_6, LEARNING_MAP_13


def get_learning_map(dataset_name, num_classes):
    """
    Return RC2 learning map for the given outdoor dataset / class count.
    Map values use RC2 convention: 0 = ignore, 1..C = valid classes.
    """
    name = dataset_name.lower()
    if name in ('semantickitti', 'kitti'):
        if num_classes == 19:
            return LEARNING_MAP_19
        if num_classes == 7:
            return LEARNING_MAP_7
        raise ValueError(f'Unsupported SemanticKITTI num_classes={num_classes}')
    if name in ('semanticposs', 'poss'):
        if num_classes == 13:
            return LEARNING_MAP_13
        if num_classes == 6:
            return LEARNING_MAP_6
        raise ValueError(f'Unsupported SemanticPOSS num_classes={num_classes}')
    raise ValueError(f'Unknown dataset_name={dataset_name}')


def remap_labels(raw_labels, learning_map):
    """
    Apply RC2 learning_map then convert to project labels:
    ignore(0) -> -100, class k -> k-1  (so valid labels are 0..C-1).
    """
    mapped = np.vectorize(learning_map.__getitem__)(raw_labels).astype(np.int32)
    labels = mapped - 1
    labels = np.where(mapped == 0, -100, labels)
    return labels
