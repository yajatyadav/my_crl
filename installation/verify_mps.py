"""Sanity check that JAX is dispatching to the Metal (Apple Silicon) backend.

Run: python verify_metal.py
"""

import time

import jax
import jax.numpy as jnp


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"Backend platform: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    devices = jax.devices()
    if not any(d.platform == "metal" or d.platform == "gpu" for d in devices):
        print(
            "\n⚠️  WARNING: No Metal/GPU device found — JAX is likely falling back to CPU.\n"
            "   Check that `jax-metal` is installed and versions match jax/jaxlib pins."
        )
    else:
        print("\n✅ Metal device detected.")

    # Quick compute + timing check: matmul should be fast on Metal, slow-ish on CPU
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4096, 4096))
    y = jax.random.normal(key, (4096, 4096))

    @jax.jit
    def matmul(a, b):
        return a @ b

    # warmup (compilation)
    matmul(x, y).block_until_ready()

    start = time.perf_counter()
    for _ in range(10):
        result = matmul(x, y)
    result.block_until_ready()
    elapsed = time.perf_counter() - start

    print(f"\n10x 4096x4096 matmul: {elapsed:.4f}s ({elapsed / 10 * 1000:.2f}ms/iter)")
    print(f"Result device: {result.devices()}")
    print(f"Result sample dtype/shape: {result.dtype}, {result.shape}")

    # As a rough heuristic: on M-series Metal GPU this should be well under
    # 200ms/iter for this size; multi-second times usually mean CPU fallback.
    if elapsed / 10 > 0.5:
        print(
            "\n⚠️  This is slower than expected for Metal GPU execution — "
            "double check the backend above isn't silently CPU."
        )


if __name__ == "__main__":
    main()