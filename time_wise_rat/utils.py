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


def get_log_return(array: np.ndarray) -> np.ndarray:
    num_zeros = np.count_nonzero(array == 0.0)
    if num_zeros > 0:
        raise ValueError("elements of array have zeros, can't compute log return")
    return np.log(array[1:]/array[:-1])
