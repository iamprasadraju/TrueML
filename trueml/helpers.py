from __future__ import annotations

import time
from functools import wraps

from tqdm import tqdm


def bench_pbar(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        M, K = self.X.shape
        N = self.y.shape[1]

        flops_per_step = (2 * M * K * N) + (2 * K * M * N)
        epochs = kwargs.get("epochs", args[1] if len(args) > 1 else 1000)

        yellow = "\033[33m"
        green = "\033[32m"
        cyan = "\033[36m"
        reset = "\033[0m"

        custom_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            f"[{yellow}{{elapsed}}<{{remaining}}{reset}{{postfix}}]"
        )

        self.pbar = tqdm(
            total=epochs,
            mininterval=0.1,
            bar_format=custom_format,
        )

        training_step = self.training_step

        @wraps(training_step)
        def monitored_step(*step_args, **step_kwargs):
            t0 = time.perf_counter()
            loss = training_step(*step_args, **step_kwargs)
            dt = time.perf_counter() - t0

            gflops = (flops_per_step / dt) / 1e9 if dt > 0 else 0

            self.pbar.set_postfix(
                loss=f"{green}{loss:.4f}{reset}",
                GFLOPs=f"{cyan}{gflops:.2f}{reset}",
                refresh=False,
            )

            self.pbar.update(1)
            return loss

        self.training_step = monitored_step

        try:
            return func(self, *args, **kwargs)
        finally:
            self.training_step = training_step
            self.pbar.close()

    return wrapper
