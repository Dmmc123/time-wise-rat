import numpy as np


def construct_patches(
        array: np.ndarray,
        num_patches: int,
        patch_length: int
) -> np.ndarray:
    array = np.lib.stride_tricks.sliding_window_view(array, num_patches)  # (N, N_p)
    array = np.lib.stride_tricks.sliding_window_view(array, patch_length, axis=0)  # (N, N_p, L_p)
    return array


def construct_windows(
        array: np.ndarray,
        window_length: int
) -> np.ndarray:
    array = np.lib.stride_tricks.sliding_window_view(array, window_length)  # (N, L_w)
    return array
