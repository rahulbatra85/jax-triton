# Copyright 2023 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for calling Triton kernels from JAX."""
# b/301982023
from __future__ import annotations

import copy
import functools
import os
import subprocess
import shutil
import types
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple, Union
import zlib
import re
from pathlib import Path

from absl import logging
import jax
jax.devices()
import jaxlib
from jax import tree_util
from jax._src import core
from jax._src import state
from jax._src import util
from jax._src.lib.mlir import ir
import jax.dlpack
from jax.interpreters import mlir
from jax.interpreters import xla
import jax.numpy as jnp
import numpy as np

CAN_USE_TRITON = False
try:
  import triton
  from triton.compiler import code_generator as code_gen
  from triton.compiler import compiler as tc
  import triton.language as tl
  from triton.runtime import autotuner
  import triton._C.libtriton.triton as _triton
  CAN_USE_TRITON = True
  arch_info = _triton.get_arch_info()
  warp_size = _triton.get_warp_size()
except ModuleNotFoundError:
  pass
try:
  from jax._src.lib import gpu_triton as triton_kernel_call_lib
except ImportError:
  raise ValueError(
    "Cannot import jaxlib triton library. You may need a newer version of jaxlib. Try installing a nightly wheel from: https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda_releases.html or https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html"
    )

os.environ["TRITON_CACHE_DIR"] = ""
map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


_JAX_TO_TRITON_TYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    # Triton has 'fp8' as well which Jax doesn't support yet.
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    # Triton defines a 'B' type, which is an alias for both i1 and bool.
    jnp.dtype("bool"): "B",
}


Grid = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Dict[str, Any]], Grid]]


def normalize_grid(grid: GridOrLambda, metaparams) -> Tuple[int, int, int]:
  if callable(grid):
    grid = grid(metaparams)
  if isinstance(grid, int):
    grid = (grid,)
  elif len(grid) > 3:
    raise ValueError("`grid` should have three or fewer dimensions.")
  return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
  return [list(reversed(range(aval.ndim))) for aval in avals]

def get_triton_type(obj: Any) -> str:
  if isinstance(obj, (jax.core.ShapedArray, state.AbstractRef)):
    return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
  if isinstance(obj, tl.constexpr):
    obj = obj.value
  if isinstance(obj, int):
    if -2**31 <= obj < 2**31:
      return "i32"
    elif 2**31 <= obj < 2**32:
      return "u32"
    elif -2**63 <= obj < 2**63:
      return "i64"
    elif 2**63 <= obj < 2**64:
      return "u64"
    else:
      raise ValueError(f"integer overflow representing {obj}")
  if isinstance(obj, float):
    return "fp64"
  if isinstance(obj, np.float32):
    return "fp32"
  if isinstance(obj, bool):
    return "B"
  if isinstance(obj, str):
    return "str"
  raise NotImplementedError(
      f"could not compute type name for {obj}: {type(obj)}"
  )


Grid = Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]
GridOrLambda = Union[Grid, Callable[[Dict[str, Any]], Grid]]

triton_kernel_call_p = jax.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True
triton_kernel_call_p.def_impl(
    functools.partial(xla.apply_primitive, triton_kernel_call_p))


@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
  return [
      core.ShapedArray(out_shape.shape, out_shape.dtype)
      for out_shape in out_shapes
  ]


def aval_size_bytes(aval):
  return np.dtype(aval.dtype).itemsize * aval.size


#def ptx_get_kernel_name(module) -> str:
def get_kernel_name(module) -> str:
  return tc.get_kernel_name(module[0], pattern='.globl')

def gpu_matrix_core_version() -> int:
    """ Determine matrix core type available on current GPU.

        0 means no tensor cores are available
        1 corresponds to MFMA in CDNA 1 architecture
        2 corresponds to MFMA in CDNA 2 architecture
        3 corresponds to MFMA in CDNA 3 architecture
    """

    arch_info = _triton.get_arch_info()
    gfx_arch_details = re.search('amd.*', arch_info)
    if gfx_arch_details is None:
        return 0
    gfx_arch_details = gfx_arch_details.group(0).strip().split('--')
    gpu_name = gfx_arch_details[1].split(':')[0]
    if gpu_name in ['gfx908']:
        return 1
    if gpu_name in ['gfx90a']:
        return 2
    if gpu_name in ['gfx940', 'gfx941', 'gfx942']:
        return 3
    return 0

def get_amdgpu_arch_fulldetails():
    """
    get the amdgpu fulll ISA details for compiling:
    i.e., arch_triple: amdgcn-amd-amdhsa; arch_name: gfx906; arch_features: sramecc+:xnack-
    """
    try:
        # TODO: package rocm.cc with Triton
        rocm_path_dir = os.getenv("ROCM_PATH", default="/opt/rocm")
        rocminfo = subprocess.check_output(rocm_path_dir + '/bin/rocminfo').decode()
        gfx_arch_details = re.search('amd.*', rocminfo).group(0).strip().split('--')
        arch_triple = gfx_arch_details[0]
        arch_name_features = gfx_arch_details[1].split(':')
        arch_name = arch_name_features[0]
        arch_features = ""

        # overwrite if provided by user
        gfx_arch = os.environ.get('MI_GPU_ARCH', arch_name)
        if gfx_arch is None:
            raise RuntimeError('gfx_arch is None (not specified)')

        mat_core_ver = gpu_matrix_core_version()
        capability = gpu_matrix_core_version() * 100

        return {"gfx_triple": arch_triple, "gfx_arch": gfx_arch, "gfx_features": arch_features,\
                 "capability": capability, "matrix_core_version": mat_core_ver}
    except BaseException:
        return None

def load_hsaco_file(filename) -> str:
    f = open(filename, "r")
    module = f.read()
    return module

def write_to_file(content: any, file_name: str, dirpath = "log"):
  # clean and setup log file
  if not os.path.exists(dirpath):
    os.mkdir(dirpath)
  file_path = os.path.join(dirpath, file_name)

  # write file
  content = str(content)
  f = open(file_path, "a")
  f.write(content)
  f.close()

def compile_ttir_to_ptx_inplace(
    ttir,
    device: int = 0,
    device_type: str = "cuda",
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    num_ctas: int = 1,
    enable_fp_fusion: bool = True,
    enable_warp_specialization: bool = False,
    enable_persistent: bool = False,
    dump: bool = False,
) -> Tuple[str, str, int, int]:
  compute_capability = triton_kernel_call_lib.get_compute_capability(device)
  arch_full_details = get_amdgpu_arch_fulldetails()
  gfx_arch = os.environ.get('MI_GPU_ARCH', arch_full_details["gfx_arch"])
  if num_warps is None:
    num_warps = tc.get_arch_default_num_warps(device_type)
    #num_warps = 1
  if num_stages is None:
    num_stages = tc.get_arch_default_num_stages(
        device_type, capability=compute_capability
    )
    #num_stages = 1
  if dump:
    write_to_file(ttir, "dump.ttir")
    print(ttir)
  try:
    ttir = tc.optimize_ttir(ttir, arch_full_details)
    ttgir = tc.ttir_to_ttgir(ttir, num_warps, warpsize=64, num_ctas=num_ctas, target=arch_full_details)
    ttgir = tc.optimize_ttgir(
        ttgir,
        num_stages,
        num_warps,
        num_ctas=num_ctas,
        target=arch_full_details,
        cluster_info=_triton.ClusterInfo(),
        enable_warp_specialization=enable_warp_specialization,
        enable_persistent=enable_persistent,
        optimize_epilogue=False,
        matrix_inst_type=0
    )
  except RuntimeError as e:
    ttir.dump()
    raise ValueError("TTIR->TTGIR pass failed!") from e
  if dump:
    write_to_file(ttgir, "dump.ttgir")
    print(ttgir)

  extern_libs = {}
  extern_libs.update(tc.get_amdgcn_bitcode_paths(gfx_arch))
  #print(extern_libs)

  try:
    llir = tc.ttgir_to_llir(ttgir, extern_libs, target=arch_full_details, tma_infos=_triton.TMAInfos())
  except RuntimeError as e:
    ttgir.dump()
    raise ValueError("TTGIR->LLIR pass failed!") from e
  shared_mem_bytes = _triton.get_shared_memory_size(ttgir)
  if dump:
    print(llir)
  
  hsa = _triton.translate_llvmir_to_hsaco(llir, gfx_arch, arch_full_details['gfx_triple'], arch_full_details['gfx_features'])

  #print(f"HSA: {type(hsa)} hsa={hsa[1]}")
  if dump:
    write_to_file(hsa[0], "dump.gcn")
    write_to_file(hsa[1], "dump.hsaco_path")
  name = get_kernel_name(hsa)
  return hsa[1], name, shared_mem_bytes, compute_capability


_COMPILED_KERNEL_CACHE = {}  # TODO(cjfj): Convert to LRU cache?


def get_or_create_triton_kernel(
    fn,
    arg_dtypes,
    scalar_args,
    *,
    num_warps,
    num_stages,
    num_ctas,
    enable_fp_fusion,
    enable_warp_specialization,
    enable_persistent,
    metaparams,
    dump: bool,
) -> Tuple[triton_kernel_call_lib.TritonKernel, Any]:
  device_type = "cuda"
  if num_warps is None:
    num_warps = tc.get_arch_default_num_warps(device_type)
    #num_warps = 1

  signature = dict(enumerate(arg_dtypes))
  # TODO(sharadmv,zhangqiaorjc): handle differently aligned pointers
  # We assume that all arrays are aligned to 16 bytes, and Triton may use this
  # assumption, unless array args are include in the `do_not_specialize` list.
  # We replace array arguments with mock Torch tensors, to allow us to use
  # `JITFunction._get_config` to get the specialization.
  mock_torch_tensor = types.SimpleNamespace(data_ptr=lambda: 16)
  args_for_specialization = [mock_torch_tensor] * len(arg_dtypes)
  for i, _, v in scalar_args:
    args_for_specialization[i] = v
  specialization = fn._get_config(*args_for_specialization)  # pylint: disable=protected-access

  constants = {fn.arg_names.index(k): v for k, v in metaparams.items()}
  constants.update({i: None for i, _, v in scalar_args if v is None})
  constants.update({i: 1 for i in specialization.equal_to_1})

  # Cache key should contain any parameter that can affect the compiler output.
  cache_key = (
      fn,
      tuple(signature.items()),
      specialization,
      tuple(constants.items()),
      num_warps,
      num_stages,
      num_ctas,
      enable_fp_fusion,
      enable_warp_specialization,
      enable_persistent,
  )
  kernel = _COMPILED_KERNEL_CACHE.get(cache_key)

  if kernel is None:
    # TODO(sharadmv): handle multiple devices, right now we assume device 0
    # which is fine when we have multiple of the same GPU but this won't work in
    # general.
    device = 0
    arch = triton_kernel_call_lib.get_compute_capability(device)
    module = code_gen.ast_to_ttir(
        fn, signature, specialization, constants, debug=dump, target=arch
    )
    ttir = str(module)  # `module`` is compiled in-place, so copy TTIR here.
    cubin, kernel_name, shared_mem_bytes, compute_capability = (
        compile_ttir_to_ptx_inplace(
            module,
            device=device,
            device_type=device_type,
            num_warps=num_warps,
            num_stages=num_stages,
            num_ctas=num_ctas,
            enable_fp_fusion=enable_fp_fusion,
            enable_warp_specialization=enable_warp_specialization,
            enable_persistent=enable_persistent,
            dump=dump,
        )
    )

    kernel = triton_kernel_call_lib.TritonKernel(
        kernel_name, num_warps, shared_mem_bytes, 
        cubin[1], ttir, compute_capability
    )

    _COMPILED_KERNEL_CACHE[cache_key] = kernel

  return kernel, specialization


def triton_kernel_call_lowering(
    ctx,
    *array_args,
    fn,
    scalar_args,
    name,
    custom_call_target_name,
    out_shapes,
    grid,
    num_warps,
    num_stages,
    num_ctas,
    enable_fp_fusion,
    enable_warp_specialization,
    enable_persistent,
    input_output_aliases,
    zeroed_outputs,
    debug,
    serialized_metadata,
    **metaparams,
):
  if jaxlib.version.__version_info__ < (0, 3, 22) and input_output_aliases:
    raise NotImplementedError(
        "`input_output_aliases` only supported on `jaxlib>=0.3.22")

  kernel_call_name = name
  args = list(ctx.avals_in)
  arg_dtypes = list(map(get_triton_type, ctx.avals_in))
  for idx, dtype, v in scalar_args:
    args.insert(idx, v)
    arg_dtypes.insert(idx, dtype)
  args.extend(ctx.avals_out)
  arg_dtypes.extend(map(get_triton_type, ctx.avals_out))
  named_args = dict(unsafe_zip(fn.arg_names, args))

  if isinstance(fn, autotuner.Autotuner):
    if any(idx not in fn.key_idx for idx, _, _ in scalar_args):
      logging.warning(
          "Auto-tuning key does not include all scalar arguments. "
          "We may perform redundant auto-tuning."
      )

    # If any metaparams have been specified explicitly, we prune any configs
    # that conflict. Note that this is more permissive than Triton's autotuner
    # implementation, which will throw an error if any keys match.
    # TODO(cjfj): Prune explicit `num_warps` / `num_stages`.
    prev_early_config_prune_fn = fn.early_config_prune

    def prune_configs(configs, named_args):
      pruned_configs = []
      for config in configs:
        if config.pre_hook is not None:
          raise NotImplementedError("`pre_hook` is not supported")

        if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
          pruned_configs.append(config)
      if prev_early_config_prune_fn is not None:
        pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
      return pruned_configs

    fn.early_config_prune = prune_configs
    fn.nargs = named_args
    configs = fn.prune_configs(metaparams)
    fn = fn.fn
  else:
    config = triton.Config(
        {},
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        enable_warp_specialization=enable_warp_specialization,
    )
    config.enable_persistent = enable_persistent
    configs = [config]

  if isinstance(fn, autotuner.Heuristics):
    updated_configs = []
    for config in configs:
      kwargs = config.kwargs.copy()
      for name, heuristic in fn.values.items():
        kwargs[name] = heuristic({**named_args, **metaparams, **kwargs})
      updated_config = copy.copy(config)
      updated_config.kwargs = kwargs
      updated_configs.append(updated_config)
    configs = updated_configs
    fn = fn.fn

  if not isinstance(fn, triton.JITFunction):
    raise ValueError(
        "`kernel` must be a Triton `JITFunction`, `Heuristics` or `Autotuner`."
    )

  outputs_offset = len(ctx.avals_in) + len(scalar_args)
  config_params = []
  for config in configs:
    config_metaparams = {**metaparams, **config.kwargs}
    config_grid = normalize_grid(grid, config_metaparams)

    config_zeroed_outputs = zeroed_outputs
    if callable(zeroed_outputs):
      config_zeroed_outputs = config_zeroed_outputs(config_metaparams)

    zeroed_params_with_sizes = {
        i + outputs_offset: aval_size_bytes(ctx.avals_out[i])
        for i in sorted(config_zeroed_outputs)
    }

    config_params.append(
        dict(
            metaparams=tuple(sorted(config_metaparams.items())),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            num_ctas=config.num_ctas,
            enable_warp_specialization=config.enable_warp_specialization,
            enable_persistent=config.enable_persistent,
            grid=config_grid,
            zeroed_params_with_sizes=tuple(zeroed_params_with_sizes.items()),
        )
    )

  kernel_calls = []
  for params in config_params:
    kernel, specialization = get_or_create_triton_kernel(
        fn,
        arg_dtypes,
        scalar_args,
        num_warps=params["num_warps"],
        num_stages=params["num_stages"],
        num_ctas=params["num_ctas"],
        enable_fp_fusion=enable_fp_fusion,
        enable_warp_specialization=params["enable_warp_specialization"],
        enable_persistent=params["enable_persistent"],
        metaparams=dict(params["metaparams"]),
        dump=debug,
    )

    kernel_params = []
    zeroed_params_with_sizes = dict(params["zeroed_params_with_sizes"])
    for i, (arg, dtype) in enumerate(zip(args, arg_dtypes)):
      if isinstance(arg, core.ShapedArray):
        kernel_params.append(
            triton_kernel_call_lib.create_array_parameter(
                zeroed_params_with_sizes.get(i, 0),
                16 if (i in specialization.divisible_by_16) else 0,
            )
        )
      elif i not in specialization.equal_to_1:
        kernel_params.append(
            triton_kernel_call_lib.create_scalar_parameter(arg, dtype)
        )

    # TODO(cjfj): Add support for `num_ctas` in kernel launch code.
    if num_ctas != 1:
      raise ValueError("`num_ctas != 1` is not yet supported.")

    kernel_calls.append(
        triton_kernel_call_lib.TritonKernelCall(
            kernel,
            params["grid"][0],
            params["grid"][1],
            params["grid"][2],
            kernel_params,
        )
    )

  if len(kernel_calls) > 1:
    named_scalar_args = {fn.arg_names[i]: v for i, _, v in scalar_args}
    input_output_aliases_with_sizes = tuple(
        (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
        for input_idx, output_idx in input_output_aliases
    )
    kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
        f"{kernel_call_name} ({fn.fn.__name__}) {named_scalar_args}",
        [(call, str(config)) for call, config in zip(kernel_calls, configs)],
        input_output_aliases_with_sizes,
    )
  else:
    kernel_call = kernel_calls[0]

  out_types = [
      ir.RankedTensorType.get(shape.shape, mlir.dtype_to_ir_type(shape.dtype))
      for shape in out_shapes
  ]
  if jaxlib.version.__version_info__ >= (0, 4, 15):
    call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)
  else:
    call_proto = kernel_call.to_proto(serialized_metadata)
  return jaxlib.hlo_helpers.custom_call(
      call_target_name=custom_call_target_name,
      result_types=out_types,
      operands=array_args,
      backend_config=zlib.compress(call_proto),
      operand_layouts=avals_to_layouts(ctx.avals_in),
      result_layouts=avals_to_layouts(ctx.avals_out),
      operand_output_aliases=dict(input_output_aliases),
  ).results


mlir.register_lowering(triton_kernel_call_p, triton_kernel_call_lowering)


class ShapeDtype(Protocol):

  @property
  def shape(self) -> Tuple[int, ...]:
    ...

  @property
  def dtype(self) -> np.dtype:
    ...


def triton_call(
    *args: Union[jax.Array, bool, int, float, np.float32],
    kernel: triton.JITFunction,
    out_shape: Union[ShapeDtype, Sequence[ShapeDtype]],
    grid: GridOrLambda,
    name: str = "",
    custom_call_target_name: str = "triton_kernel_call",
    num_warps: Optional[int] = 1,
    num_stages: Optional[int] = 1,
    num_ctas: int = 1,
    enable_fp_fusion: bool = True,
    enable_warp_specialization: bool = False,
    enable_persistent: bool = False,
    input_output_aliases: Optional[Dict[int, int]] = None,
    zeroed_outputs: Union[
        Sequence[int], Callable[[Dict[str, Any]], Sequence[int]]
    ] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
) -> Any:
  """Calls a Triton kernel with `jax.Array` arguments.

  Example usage:

  First we define a simple kernel that adds two vectors.

  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def add_kernel(
      x_ptr,
      y_ptr,
      output_ptr,
      block_size: tl.constexpr,
  ):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < 8
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
  ```

  Then we use `triton_call` to call it from JAX.

  ```python
  import jax
  import jax.numpy as jnp
  import jax_triton as jt

  def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 8
    return jt.triton_call(
        x,
        y,
        kernel=add_kernel,
        out_shape=out_shape,
        grid=(x.size // block_size,),
        block_size=block_size)

  x_val = jnp.arange(8)
  y_val = jnp.arange(8, 16)
  print(add(x_val, y_val))
  print(jax.jit(add)(x_val, y_val))
  ```

  Args:
    *args: Inputs for the Triton kernel.
    kernel: A Triton kernel (e.g. a function decorated with `triton.jit`). All
      static values should be annotated with `triton.language.constexpr`.
    out_shape: A `jax.ShapeDtypeStruct` (or something that has `.shape` and
      `.dtype` attributes) or a sequence thereof that specify the output(s) of
      the kernel. Pointers for each of the `jax.ShapeDtypeStruct`s in
      `out_shape` will be passed into `kernel` following the input parameters.
    grid: An integer, tuple of up to 3 integers, or a function that returns a
      tuple of up to 3 integers. When `grid` is an integer, `kernel` is
      invocated in `grid`-many parallel executions. When `grid` is a sequence of
      integers, `kernel` is launched in a `prod(grid)`-many parallel execution.
      When `grid` is a function, it is passed `**metaparams` and should return a
      tuple of up to 3 integers.
    input_output_aliases: A dictionary mapping input argument indices to output
      indices. Providing a mapping will alias the corresponding buffers.
    zeroed_outputs: A sequence of indices, or a function returning a sequence of
      indices, for outputs that should be zeroed before the kernel is launched.
    num_warps: The number of warps used to execute the Triton kernel.
    num_stages: The number of stages emitted by the Triton compiler.
    debug: Prints out intermediate IRs if True for debugging purposes.
    serialized_metadata: Arbitrary metadata that will be added into the
      serialized kernel call.
    **metaparams: Additional keyword arguments that will be provided to a `grid`
      (if it is a function) and to the Triton kernel as `constexpr` arguments.

  Returns:
    Outputs from the Triton kernel.
  """
  if not CAN_USE_TRITON:
    raise ValueError(
        "`triton_call` is only available when `triton` is installed."
    )
  out_shape = tree_util.tree_map(
      lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
  flat_args, _ = tree_util.tree_flatten(args)
  # TODO(sharadmv): check in_tree is flat (no Pytrees allowed in triton_call)
  flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

  array_args = []
  scalar_args = []
  for i, arg in enumerate(flat_args):
    if isinstance(arg, (bool, int, float)):
      scalar_args.append((i, get_triton_type(arg), arg))
    elif isinstance(arg, np.float32):
      scalar_args.append((i, get_triton_type(arg), float(arg)))
    else:
      array_args.append(arg)

  if input_output_aliases is None:
    input_output_aliases = {}

  out_flat = triton_kernel_call_p.bind(
      *array_args,
      fn=kernel,
      scalar_args=tuple(scalar_args),
      name=name,
      custom_call_target_name=custom_call_target_name,
      out_shapes=tuple(flat_out_shapes),
      grid=grid,
      num_warps=num_warps,
      num_stages=num_stages,
      num_ctas=num_ctas,
      enable_fp_fusion=enable_fp_fusion,
      enable_warp_specialization=enable_warp_specialization,
      enable_persistent=enable_persistent,
      input_output_aliases=tuple(input_output_aliases.items()),
      zeroed_outputs=zeroed_outputs,
      debug=debug,
      serialized_metadata=serialized_metadata,
      **metaparams,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
