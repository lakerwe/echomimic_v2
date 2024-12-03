# TODO: Adapted from cli
from typing import Callable, List, Optional

import numpy as np


def ordered_halving(val):           # 可以得到一个0-1之间的小数
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]        #逆序排列
    as_int = int(bin_flip, 2)       #将2进制数转换为十进制数

    return as_int / (1 << 64)


def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:                      # 如果总帧数小于上下文要求的长度,直接返回总帧数列表
        yield list(range(num_frames))
        return

    context_stride = min(                               # 计算一个上下文步长，之所以给log2(x)加1，是因为，当x=1时，log2(1)=0,则log2(1)+1 = 1,即上下文步长最小要是大于等于1
        context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1            #之所以以2为底，因为后面步长的计算使用的左移操作
    )

    for context_step in 1 << np.arange(context_stride):         # 注意这里是将数字 1 分别左移 np.arange(context_stride) 位
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [
                e % num_frames
                for e in range(j, j + context_size * context_step, context_step)
            ]


def get_context_scheduler(name: str) -> Callable:
    if name == "uniform":
        return uniform
    else:
        raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: List[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )
