"""File-backed parameters for backends that keep parameters in host memory.

TT's PJRT plugin keeps every parameter on the host, in its JAX dtype and
layout, and caches only the copies it converts for the device. A model whose
parameters exceed host RAM therefore cannot load, even when its converted
copies fit the device: Qwen3.8-Flash-Next is ~250 GB of BF16.

``device_put`` of a contiguous ``np.memmap`` borrows the file's pages instead
of copying them, so a parameter written once, shard by shard, to a cache
directory lives in clean page cache the kernel can evict and read back. Later
launches map the files directly and skip the checkpoint and its recipes.

The key covers the checkpoint files a parameter reads (size, mtime, inode),
the mapping spec, the target and the mesh, but not the loader's code: clear
the directory after changing a recipe. Entries for replaced files are not
removed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

_ALIGN = 4096  # page-aligned shards keep every view eligible for borrowing


def _bounds(index, shape):
    """A shard's index (a tuple of slices) as [[start, stop], ...]."""
    return [list(s.indices(n)[:2]) for s, n in zip(index, shape)]


def _spec_to_json(spec: P):
    return [list(a) if isinstance(a, tuple) else a for a in spec]


def _spec_from_json(axes):
    return P(*(tuple(a) if isinstance(a, list) else a for a in axes))


class HostWeightCache:
    def __init__(self, directory: str | os.PathLike, mesh):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mesh = mesh
        self.hits = self.writes = 0

    def key(self, spec_identity: str, path: str, expected, files) -> str:
        """``files`` are the checkpoint files the parameter is read from. Only
        their stat enters the key, so files added or replaced elsewhere in the
        checkpoint (a download still running) keep the other entries valid."""
        signature = sorted(
            (str(name), (stat := os.stat(name)).st_size, stat.st_mtime_ns, stat.st_ino)
            for name in files
        )
        description = [
            signature,
            spec_identity,
            path,
            list(expected.shape),
            str(expected.dtype),
            sorted(self.mesh.shape.items()),
        ]
        return hashlib.sha256(json.dumps(description).encode()).hexdigest()[:32]

    def load(self, key: str, expected) -> jax.Array | None:
        """The cached parameter as file-backed shards, or None on a miss."""
        meta_path = self.directory / f"{key}.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        shape, dtype = tuple(meta["shape"]), jnp.dtype(meta["dtype"])
        if (shape, dtype) != (tuple(expected.shape), jnp.dtype(expected.dtype)):
            return None
        sharding = NamedSharding(self.mesh, _spec_from_json(meta["spec"]))
        data = np.memmap(self.directory / f"{key}.bin", dtype=np.uint8, mode="r")
        views = {}
        for shard in meta["shards"]:
            nbytes = int(np.prod(shard["shape"])) * dtype.itemsize
            view = data[shard["offset"] : shard["offset"] + nbytes]
            views[json.dumps(shard["bounds"])] = view.view(dtype).reshape(shard["shape"])
        arrays = []
        for device, index in sharding.addressable_devices_indices_map(shape).items():
            arrays.append(jax.device_put(views[json.dumps(_bounds(index, shape))], device))
        self.hits += 1
        return jax.make_array_from_single_device_arrays(shape, sharding, arrays)

    def store(self, key: str, value: jax.Array) -> jax.Array:
        """Write ``value``'s distinct shards, then return the file-backed copy."""
        spec = value.sharding.spec if isinstance(value.sharding, NamedSharding) else None
        if spec is None:
            return value
        dtype = jnp.dtype(value.dtype)
        shards, offset = {}, 0
        bin_path = self.directory / f"{key}.bin"
        with open(bin_path, "wb") as f:
            for shard in value.addressable_shards:
                bounds = _bounds(shard.index, value.shape)
                name = json.dumps(bounds)
                if name in shards:
                    continue  # a replica of a shard already written
                host = np.ascontiguousarray(np.asarray(shard.data))
                offset = -(-offset // _ALIGN) * _ALIGN
                f.seek(offset)
                f.write(host.tobytes())
                shards[name] = {"bounds": bounds, "shape": list(host.shape), "offset": offset}
                offset += host.nbytes
        meta = {
            "shape": list(value.shape),
            "dtype": dtype.name,
            "spec": _spec_to_json(spec),
            "shards": list(shards.values()),
        }
        # The sidecar marks the entry complete, so write it last and atomically.
        tmp = self.directory / f"{key}.json.tmp"
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, self.directory / f"{key}.json")
        self.writes += 1
        cached = self.load(key, value)
        self.hits -= 1
        return cached
