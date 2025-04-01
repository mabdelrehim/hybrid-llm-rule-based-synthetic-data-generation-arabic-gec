from dataclasses import dataclass, field
from typing import Optional, Union, Literal

@dataclass
class LoRAArguments:
    """
    Arguments for using peft training
    """
    lora_r: Optional[int] = field(default=32,
                                  metadata={"help": "Rank parameter for LoRA."})
    lora_alpha: Optional[int] = field(default=8,
                                      metadata={"help": "Alpha parameter for LoRA scaling."})
    lora_dropout: Optional[float] = field(default=0.05,
                                          metadata={"help": "Dropout rate for LoRA."})
    init_lora_weights: bool | Literal["gaussian", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"] = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the LoRA layers. Passing `'True'` (default) results in the default "
                "initialization from the reference implementation from Microsoft. Passing `'gaussian'` results "
                "in Gaussian initialization scaled by the LoRA rank for linear and layers. Setting the initialization "
                "to `'False'` leads to completely random initialization and *is discouraged.*"
                "Passing `'olora'` results in OLoRA initialization."
                "Passing `'pissa'` results in PiSSA initialization."
                "Passing `'pissa_niter_[number of iters]'` initiates Fast-SVD-based PiSSA initialization, "
                "where [number of iters] indicates the number of subspace iterations to perform fsvd, and must be a nonnegative integer."
                "Pass `'loftq'` to use LoftQ initialization"
            ),
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "When set to True, uses <a href='https://doi.org/10.48550/arXiv.2312.03732'>Rank-Stabilized LoRA</a>"
                " which sets the adapter scaling factor to `lora_alpha/math.sqrt(r)`, since it"
                " was proven to work better. Otherwise, it will use the original default"
                " value of `lora_alpha/r`."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )