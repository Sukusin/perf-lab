import platform

import torch


def snapshot_env() -> dict:
    out = {
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    if torch.cuda.is_available():
        out.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
    return out
