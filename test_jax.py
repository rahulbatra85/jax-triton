import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

k1, k2 = random.split(random.PRNGKey(0), 2)
print(k1, k2)