# llm_finetune_peft
Experiments with fine-tuning large language models

> A quick fix was needed in the process.
> If the following error occurs in `deepspeed/runtime/zero/stage3.py`:<br> `ValueError: max() arg is an empty sequence`
> <br>Then we need to replace this code fragment:
> ```python
> largest_partitioned_param_numel = max([
>            max([max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group])
>            for fp16_partitioned_group in self.fp16_partitioned_groups
>        ])
> ```
> on this one:
> ```python
> largest_partitioned_param_numel = max([
>            max([max(tensor.numel(), tensor.ds_numel) for tensor in fp16_partitioned_group])
>            for fp16_partitioned_group in self.fp16_partitioned_groups if len (fp16_partitioned_group) > 0
>        ])
> ```



