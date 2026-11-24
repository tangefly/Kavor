# Vendored Marlin (FP8 W8A16)

Marlin is a fast, layout-optimized weight-only GEMM kernel originally by Elias
Frantar (IST-DASLab), later modified by Neural Magic / vLLM. Kavor vendors the
FP8 weight-only path so that blockwise-FP8 checkpoints (128×128 scales) can run
as W8A16 without a CUTLASS dependency.

## Source

- Repository: https://github.com/vllm-project/vllm
- Commit: `a2f713002df9fd08c0fe13272c76547421721f2d`
  (`v0.23.1rc0-723-ga2f713002d`, 2026-07-02)
- Local path: `csrc/libtorch_stable/quantization/marlin/`
- License: Apache-2.0 (see `LICENSE.vllm` / `NOTICE`)

## Files

| File | Origin / change |
| --- | --- |
| `marlin.cu` | vLLM upstream; binding rewritten (`torch::stable::Tensor` → `at::Tensor`, `STABLE_TORCH_LIBRARY_IMPL` → `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` under namespace `kavor_marlin`), FP8-only type dispatch |
| `gptq_marlin_repack.cu` | vLLM upstream; same binding rewrite |
| `marlin_template.h`, `marlin_mma.h`, `marlin.cuh`, `marlin_dtypes.cuh`, `kernel.h`, `dequant.h` | vLLM upstream, unmodified except the local scalar-type include |
| `scalar_type_local.h` | vendored `csrc/core/scalar_type.hpp` (namespace `vllm`) |
| `generate_kernels.py` | vLLM upstream, trimmed to the FP8 weight config |
| `generated/` | pre-generated FP8 kernels + `kernel_selector.h` (committed) |

## Kernel instantiation

Only `kFE4M3fn` weights with 16-bit activations are generated, i.e. the two
files

```
generated/sm80_kernel_float16_fe4m3fn_float16.cu
generated/sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu
```

They are produced with:

```bash
cd kavor/csrc/marlin
python generate_kernels.py 8.0
```

The generated files are committed; the build does **not** run the generator.
The `sm80` template is used for SM80/86/89/90 (gencode `8.0+PTX` plus the
current device's cubin).

## Runtime constraints

- 128×128 block scales (`group_size = 128`) or channel-wise (`-1`).
- `size_n % 64 == 0`, `size_k % 128 == 0` after the necessary zero padding.
- Weight is repacked to int32 by `gptq_marlin_repack(num_bits=8)`.
- Scales must be converted to the compute dtype and multiplied by
  `2**120` (bf16) / `2**8` (fp16) before the permute, so the kernel can do the
  exponent re-bias for free.
