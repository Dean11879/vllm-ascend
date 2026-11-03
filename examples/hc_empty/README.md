# HC AscendC empty-output adaptation

## Version and scope

Branch base: official vllm-project/vllm-ascend main at
`426a26bc1c1c3ebe008a47b162f8e3e6aee395c6` (fetched 2026-11-02).
Original model integration came from PR https://github.com/vllm-project/vllm-ascend/pull/16981.
This branch preserves current main model-state and cache-spec hooks.

- `vllm_ascend/models/qwen4_exp/model.py`: replace upstream GatedResidual and prepare weights after loading.
- `vllm_ascend/models/qwen4_exp/hc_fused.py`: AscendC norm/down/SiLU/up/mix/inject; Triton residual combine with BLOCK=1024.
- `zeros_to_empty.patch`: external vendor wrapper patch. This is required; installing the unmodified wheel still zeros outputs.

Vendor delivery: https://gitcode.com/kdy18482276080/test11/tree/main/text_gated_residual_a2a3a5_20261028
Use its matching run and wheel. Device kernel computation is unchanged.
Original wheel SHA256: `56ed90fddf4780ff154c34f983bff2d49afcf1b6741037490258362fb87d29a1`.
Original run SHA256: `ab17699053abeb2eb13a2371bcbe2b8314ff779d34cde795622f8beb0d7a0195`.

## Apply empty optimization

Install the matching operator run and wheel in an isolated environment.
Locate the installed package's `cann_ops_transformer_custom/csrc/attention` directory.
Check `text_gated_residual.cpp` SHA256 before applying:
`d670e50050b29a83d788991773815cc61b7b00b0671f9e1fe9bc987ae03649f0`.
From that directory, run (use the absolute branch checkout path):

```bash
git -c core.autocrlf=false apply --check /path/to/vllm-ascend/examples/hc_empty/zeros_to_empty.patch
git -c core.autocrlf=false apply /path/to/vllm-ascend/examples/hc_empty/zeros_to_empty.patch
```

Expected patched SHA256:
`d3c2d82277cf818986d512a6300ed726c5e6df8275aaf1ea94ef3c29ecfe5896`.
Use a fresh extension build cache and process to rebuild the wrapper; do not reuse a loaded zeros extension.
The patch changes only mixed/inject output allocations. Internal accumulator initialization and backward caches remain unchanged.
Both outputs are overwritten by the kernel before reading; prior poison/output coverage tests cover the inference cases below.

## Validation boundary

Historical isolated HC validation used vLLM `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`
and Ascend baseline `adc05909ea0125d058c2c9eb3131e761929f2551`.
M16/32/64/128/256/512 vendor precision cases passed with FP32 gamma.
24 gate/gamma coverage cases matched zeros outputs exactly, including poisoned outputs.
M16/32/64/128/256 graph profiling confirmed removal of the two output zero kernels.
These are historical component results, not full-model validation of this new main-based branch.
`gamma=None` remains a known vendor issue; model adapter supplies raw FP32 gamma.
The 2026-11-03 module smoke validation below covers this main-based adapter; full-model validation is still required.
The adapter inherits the upstream checkpoint contract. Remove global class substitution when an upstream supported HC factory becomes available.

## Git remotes and future sync

Remotes are local Git configuration, not branch contents. For another clone:

```bash
git remote set-url origin https://github.com/Dean11879/vllm-ascend.git
git remote add upstream https://github.com/vllm-project/vllm-ascend.git
git fetch upstream main
git switch qwen38-hc-adapt
git merge upstream/main
```

Resolve any HC interface conflicts, validate, then `git push origin qwen38-hc-adapt`.
GitHub Sync fork updates the fork's selected/default branch; it does not automatically merge upstream into this HC feature branch.

## 2026-11-03 debug and module smoke validation

Use the paired vLLM source revision in `.github/vllm-main-verified.commit`:
`544f182d9f0c9bfe8bd7645525951d5e3692a542` for this branch baseline.
The old image vLLM revision `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
lacks the PLE module path required by current Ascend main.
The validated Python dependency overrides were Transformers 5.19.0 and
Tokenizers 0.23.2 (image Transformers 5.14.1 cannot import Qwen4ExpTextConfig).
Use the matching vLLM requirements when preparing a full deployment.

On 109, the isolated test used CANN 9.2.0, Torch 2.10.0+cpu,
torch-npu 2.10.0.post7 and Python 3.12.13. Only physical device 4 was mapped;
inside this container its visible logical index is 0, so
`ASCEND_RT_VISIBLE_DEVICES=0` was required, not 4.
Set `TMPDIR` to a writable executable directory with enough space for installation
and compilation. Keep extension caches isolated from older vendor builds.

The host Docker 20.10.8 default seccomp configuration blocked even a basic Python
thread and ninja process creation. A controlled diagnostic container with
`--security-opt seccomp=unconfined` passed both checks and compiled the wrapper.
This is an environment diagnosis, not a requirement to disable security globally;
use an approved compatible container policy for deployment.

The patch is stored with LF via a directory-local `.gitattributes`; validation
checks the Git-staged patch against the original vendor source.
The adapter reads `config.hc_lowrank` when slicing the merged projection weight,
because the paired vLLM no longer defines the old `lora_rank` attribute.
Projection sizes and HC mathematical operations are unchanged.

Passed on one NPU: fresh C++ wrapper compilation, model module import/class wiring,
weight preparation, and complete HC combine_and_mix eager/graph replay for
M=16,32,64,128,256 with gate enabled and disabled. Synthetic BF16 weights/inputs
were used with FP32 gamma; outputs were finite and graph replay matched eager
exactly. This is smoke coverage, not independent golden precision acceptance,
full-model execution, or a performance result. The complete Ascend repository
native build was not repeated; only the vendor wrapper and Triton combine were
compiled for this check.
