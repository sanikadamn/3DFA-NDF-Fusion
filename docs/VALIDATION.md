# Validation record

Executed in the development environment:

- Python compileall: passed for all delivered Python source.
- bash -n: passed for the two added shell scripts.
- unittest: 14 tests discovered; 7 passed, 7 skipped because PyTorch is unavailable.
- The launcher test executes original/new shell scripts with a recording torchrun
  and parses their actual arguments with the actual main.py argument function.
  Only dataset/output paths and the NDF config differ at default settings.
- Vendor-source comparison: only the documented relative import and four device
  allocations differ from the supplied NDF sources.
- The generated unified patch applies to a pristine copy of the supplied baseline;
  all resulting files are byte-compared with the delivered tree (excluding the
  patch/manifest themselves).

Not executed here:

- Miniconda creation or complete dependency installation/resolution.
- PyTorch tensor tests (installation was attempted but the CPU wheel was unavailable
  through the execution environment's network).
- Actual pretrained NDF checkpoint download/loading.
- Real-batch smoke test, GPU training, memory/throughput benchmarks.
- RLBench/CoppeliaSim installation or closed-loop task evaluation.

Required next checks on AWS are documented in the guide:

1. pip check and CUDA/import checks.
2. tools.check_ndf using real pretrained weights.
3. unittest (ensure tensor tests run, not skip).
4. tools.check_data for the chosen task data.
5. SMOKE_ONLY=1 with the actual batch-16 training configuration.

No performance, convergence, success-rate or policy-equivariance claims are made.
