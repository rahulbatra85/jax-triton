import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    length,
    output_ptr,
    block_size: tl.constexpr,
):
    """Adds two vectors."""
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < length
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = 8
    return jt.triton_call(
        x,
        y,
        x.size,
        kernel=add_kernel,
        out_shape=out_shape,
        grid=(x.size // block_size,),
        block_size=block_size)


@triton.jit
def empty_kernel(x_ptr, block_size: tl.constexpr):
    """empty kernel"""
    pid = tl.program_id(axis=0)


def empty() -> jnp.ndarray:
    x = jnp.arange(2)
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    return jt.triton_call(x, kernel=empty_kernel,
                              out_shape=out_shape,
                              grid=(1,),
                              block_size=8)

x_val = jnp.arange(8)
y_val = jnp.arange(8, 16)
print(f"x_val: {x_val}")
print(f"y_val: {y_val}")
print(add(x_val, y_val))
print(jax.jit(add)(x_val, y_val))
