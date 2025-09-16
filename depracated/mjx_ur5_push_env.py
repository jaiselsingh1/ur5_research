import jax 
import jax.numpy as jnp 
from jax import jit, vmap 
import mujoco 
from mujoco import mjx 
import numpy as np 
from typing import NamedTuple, Tuple
from functools import partial


class Command(NamedTuple):
    """Desired EE Pose"""
    trans_x: jnp.ndarray
    trans_y: jnp.ndarray
    trans_z: jnp.ndarray
    rot_x: jnp.ndarray
    rot_y: jnp.ndarray
    rot_z: jnp.ndarray
    rot_w: jnp.ndarray

class EnvState(NamedTuple):
    """Environment state for MJX"""
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    ee_target_pos: jnp.ndarray
    ee_target_quat: jnp.ndarray
    time: jnp.ndarray

