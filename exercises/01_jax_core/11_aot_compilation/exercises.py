"""
JAX AOT (Ahead-of-Time) Compilation - 10 Exercises
===================================================

AOT compilation lets you compile JAX functions BEFORE running them,
separating the "compile" step from the "execute" step.

WHY USE AOT COMPILATION?
------------------------
1. **Predictable latency**: No surprise compilation during inference
2. **Deployment**: Compile once, deploy the artifact, run many times
3. **Debugging**: Inspect the compiled HLO/StableHLO representation
4. **Reproducibility**: Ensure the same compiled code across runs
5. **Serving**: Pre-compile models for production serving systems

JIT vs AOT - KEY DIFFERENCES:
-----------------------------
- JIT: Compile lazily on first call, cache for reuse
- AOT: Compile explicitly upfront, get a compiled function object

THE AOT COMPILATION PIPELINE:
-----------------------------
    Python Function
          |
          v
    jax.jit(fn)         # Create a "staged" function
          |
          v
    .lower(args)        # Lower to StableHLO IR (intermediate representation)
          |
          v
    .compile()          # Compile to executable (XLA)
          |
          v
    Compiled Function   # Ready to run, no more compilation needed

WHEN TO USE AOT:
----------------
- Production ML serving (compile during startup, serve with no latency spikes)
- When you need to inspect/debug the compiled representation
- When shipping pre-compiled models
- Benchmarking (separate compile time from execution time)
- When compilation time is significant and you want it predictable

Reference: https://jax.readthedocs.io/en/latest/aot.html
"""

import jax
import jax.numpy as jnp
from jax import jit
import time


# =============================================================================
# Exercise 1: Basic AOT Compilation Pipeline
# =============================================================================
def exercise_basic_aot():
    """
    The AOT pipeline has three steps:
    1. jax.jit(fn) - Create a staged function
    2. .lower(args) - Lower to StableHLO (needs example args for shapes)
    3. .compile() - Compile to executable

    The "lowered" representation is an intermediate form (StableHLO) that
    describes the computation. The "compiled" form is the actual executable.

    Your task:
    1. Create a simple function add_and_square(x, y) = (x + y) ** 2
    2. Use jax.jit() to create a staged function
    3. Call .lower() with example inputs to get the lowered representation
    4. Call .compile() to get the compiled function
    5. Run the compiled function and verify the result

    Hints:
    - staged = jax.jit(fn)
    - lowered = staged.lower(x, y)  # needs actual arrays for shape info
    - compiled = lowered.compile()
    - result = compiled(x, y)  # run like a normal function
    """
    def add_and_square(x, y):
        """Simple function: (x + y) ** 2"""
        return (x + y) ** 2

    # Create example inputs (AOT needs these to know the shapes)
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])

    # TODO: Step 1 - Create staged function with jax.jit()
    staged = jax.jit(add_and_square)

    # TODO: Step 2 - Lower to StableHLO (pass example inputs)
    lowered = staged.lower(x, y)

    # TODO: Step 3 - Compile to executable
    compiled = lowered.compile()

    # TODO: Step 4 - Run the compiled function
    result = compiled(x, y)

    # Verify against direct computation
    expected = (x + y) ** 2
    results_match = jnp.allclose(result, expected)

    return {
        'result': result,
        'expected': expected,
        'results_match': results_match,
        'compiled_type': type(compiled).__name__
    }


# =============================================================================
# Exercise 2: Inspecting Lowered HLO Representation
# =============================================================================
def exercise_inspect_hlo():
    """
    The lowered representation is StableHLO - a stable intermediate language.
    You can inspect it to understand what operations JAX will execute.

    WHY INSPECT HLO?
    - Debug unexpected behavior
    - Verify optimizations are applied
    - Understand what the compiler sees
    - Educational: see how high-level ops become low-level ops

    The HLO shows:
    - Input parameters and their shapes
    - Each operation in sequence
    - How operations are fused together

    Your task:
    1. Create a function with multiple operations: relu(x @ W + b)
    2. Lower it and inspect the HLO text
    3. Count how many operations are in the graph

    Hints:
    - lowered.as_text() returns the StableHLO as a string
    - Look for operations like "dot_general" (matmul), "add", "maximum"
    - jnp.maximum(0, x) is ReLU
    """
    def mlp_layer(x, W, b):
        """Single MLP layer: relu(x @ W + b)"""
        return jnp.maximum(0, x @ W + b)

    # Create example inputs
    x = jnp.ones((4, 8))    # batch of 4, input dim 8
    W = jnp.ones((8, 16))   # weight matrix: 8 -> 16
    b = jnp.ones((16,))     # bias vector

    # TODO: Create staged function and lower it
    staged = jax.jit(mlp_layer)
    lowered = staged.lower(x, W, b)

    # TODO: Get the HLO text representation
    hlo_text = lowered.as_text()

    # TODO: Count operations (look for common op names)
    # Common ops: dot_general (matmul), add, maximum, broadcast
    has_matmul = 'dot_general' in hlo_text or 'dot' in hlo_text
    has_add = 'add' in hlo_text
    has_maximum = 'maximum' in hlo_text or 'max' in hlo_text

    # Compile and run to verify
    compiled = lowered.compile()
    result = compiled(x, W, b)

    return {
        'hlo_preview': hlo_text[:500] + '...' if len(hlo_text) > 500 else hlo_text,
        'hlo_length': len(hlo_text),
        'has_matmul': has_matmul,
        'has_add': has_add,
        'has_maximum': has_maximum,
        'result_shape': result.shape
    }


# =============================================================================
# Exercise 3: AOT with Different Input Shapes
# =============================================================================
def exercise_aot_shapes():
    """
    IMPORTANT: AOT compilation is shape-specific!

    When you lower a function, you "burn in" the input shapes.
    The compiled function ONLY works with those exact shapes.

    This is different from JIT, which recompiles for new shapes.
    With AOT, you must compile separate versions for each shape.

    WHY IS THIS USEFUL?
    - Production serving: you know your batch sizes ahead of time
    - Avoids unexpected recompilation
    - Can pre-compile for all expected shapes

    Your task:
    1. Compile a function for shape (4,)
    2. Try to run it with shape (4,) - should work
    3. Compile another version for shape (8,)
    4. Verify each compiled version only works with its shape

    Hints:
    - Compiled functions will raise an error if given wrong shapes
    - Use try/except to catch shape mismatch errors
    """
    def double(x):
        return x * 2

    # TODO: Compile for shape (4,)
    x_4 = jnp.ones((4,))
    compiled_4 = jax.jit(double).lower(x_4).compile()

    # TODO: Compile for shape (8,)
    x_8 = jnp.ones((8,))
    compiled_8 = jax.jit(double).lower(x_8).compile()

    # TODO: Run compiled_4 with correct shape
    result_4 = compiled_4(x_4)

    # TODO: Run compiled_8 with correct shape
    result_8 = compiled_8(x_8)

    # TODO: Try running compiled_4 with wrong shape (should fail)
    wrong_shape_error = None
    try:
        _ = compiled_4(x_8)  # This should fail!
    except Exception as e:
        wrong_shape_error = type(e).__name__

    return {
        'result_4_shape': result_4.shape,
        'result_8_shape': result_8.shape,
        'wrong_shape_error': wrong_shape_error,
        'shape_specific': wrong_shape_error is not None
    }


# =============================================================================
# Exercise 4: Measuring Compilation vs Execution Time
# =============================================================================
def exercise_timing_aot():
    """
    AOT lets you separate compilation time from execution time.

    This is crucial for:
    - Benchmarking: measure pure execution time without compile overhead
    - Production: compile during startup, serve with consistent latency
    - Understanding: see how much time goes to compile vs execute

    Typical pattern:
    1. Compile once (can be slow, that's OK)
    2. Execute many times (should be fast and consistent)

    Your task:
    1. Create a compute-intensive function
    2. Measure the time to lower and compile
    3. Measure the execution time (multiple runs)
    4. Compare compile time vs execution time

    Hints:
    - Use time.perf_counter() for timing
    - Use .block_until_ready() to ensure computation completes
    - Compilation can be 10-100x slower than execution
    """
    def heavy_computation(x):
        """Matrix operations that benefit from compilation."""
        for _ in range(5):
            x = jnp.sin(x @ x.T)
        return x

    x = jnp.ones((200, 200))

    # TODO: Time the lowering step
    start_lower = time.perf_counter()
    staged = jax.jit(heavy_computation)
    lowered = staged.lower(x)
    lower_time = time.perf_counter() - start_lower

    # TODO: Time the compilation step
    start_compile = time.perf_counter()
    compiled = lowered.compile()
    compile_time = time.perf_counter() - start_compile

    # TODO: Time execution (first run)
    start_exec1 = time.perf_counter()
    result1 = compiled(x)
    result1.block_until_ready()
    exec_time_1 = time.perf_counter() - start_exec1

    # TODO: Time execution (second run - should be similar)
    start_exec2 = time.perf_counter()
    result2 = compiled(x)
    result2.block_until_ready()
    exec_time_2 = time.perf_counter() - start_exec2

    total_compile = lower_time + compile_time

    return {
        'lower_time_ms': lower_time * 1000,
        'compile_time_ms': compile_time * 1000,
        'total_compile_ms': total_compile * 1000,
        'exec_time_1_ms': exec_time_1 * 1000,
        'exec_time_2_ms': exec_time_2 * 1000,
        'compile_vs_exec_ratio': total_compile / exec_time_2 if exec_time_2 > 0 else 0
    }


# =============================================================================
# Exercise 5: AOT with Static Arguments
# =============================================================================
def exercise_aot_static_args():
    """
    Just like JIT, AOT supports static_argnums for compile-time constants.

    Static arguments are "burned into" the compiled function.
    This is useful when:
    - An argument affects the computation graph structure
    - An argument is known at compile time (like axis, num_classes, etc.)

    The compiled function won't accept that argument anymore -
    it's baked into the compiled code!

    Your task:
    1. Create a function reduce_axis(x, axis) that sums along an axis
    2. Compile with axis=0 as static
    3. Compile another version with axis=1 as static
    4. Verify each only works for its specific axis

    Hints:
    - Use jax.jit(fn, static_argnums=(1,)) to make axis static
    - When lowering, pass the static arg value
    - The compiled function won't take axis as an argument
    """
    def reduce_axis(x, axis):
        """Sum array along specified axis."""
        return jnp.sum(x, axis=axis)

    # Example input
    x = jnp.ones((3, 4, 5))

    # TODO: Compile with axis=0 as static
    staged_axis0 = jax.jit(reduce_axis, static_argnums=(1,))
    lowered_axis0 = staged_axis0.lower(x, 0)  # axis=0 is baked in
    compiled_axis0 = lowered_axis0.compile()

    # TODO: Compile with axis=1 as static
    staged_axis1 = jax.jit(reduce_axis, static_argnums=(1,))
    lowered_axis1 = staged_axis1.lower(x, 1)  # axis=1 is baked in
    compiled_axis1 = lowered_axis1.compile()

    # TODO: Run each - note: compiled function only takes x, not axis!
    result_axis0 = compiled_axis0(x)  # No axis argument needed
    result_axis1 = compiled_axis1(x)  # No axis argument needed

    return {
        'input_shape': x.shape,
        'result_axis0_shape': result_axis0.shape,  # Should be (4, 5)
        'result_axis1_shape': result_axis1.shape,  # Should be (3, 5)
        'axis0_correct': result_axis0.shape == (4, 5),
        'axis1_correct': result_axis1.shape == (3, 5)
    }


# =============================================================================
# Exercise 6: Compilation Cost Analysis
# =============================================================================
def exercise_compilation_cost():
    """
    Compilation cost depends on:
    - Function complexity (more ops = longer compile)
    - Input sizes (larger shapes can mean more optimization work)
    - Number of operations to fuse

    Understanding compilation cost helps you:
    - Decide what to pre-compile vs JIT
    - Set appropriate timeouts for serving systems
    - Optimize your functions for faster compilation

    Your task:
    1. Create functions of varying complexity
    2. Measure compilation time for each
    3. Observe how complexity affects compile time

    Hints:
    - Simple functions compile fast
    - More operations = more compile time
    - Very large shapes can increase compile time
    """
    # Simple function
    def simple_fn(x):
        return x + 1

    # Medium complexity
    def medium_fn(x):
        x = jnp.sin(x)
        x = jnp.cos(x)
        x = x ** 2
        return x

    # Higher complexity
    def complex_fn(x):
        for _ in range(10):
            x = jnp.sin(x) + jnp.cos(x)
            x = x @ x.T
        return x

    x_small = jnp.ones((50, 50))
    x_medium = jnp.ones((100, 100))

    # TODO: Time compilation for simple function
    start = time.perf_counter()
    _ = jax.jit(simple_fn).lower(x_small).compile()
    simple_time = time.perf_counter() - start

    # TODO: Time compilation for medium function
    start = time.perf_counter()
    _ = jax.jit(medium_fn).lower(x_small).compile()
    medium_time = time.perf_counter() - start

    # TODO: Time compilation for complex function
    start = time.perf_counter()
    _ = jax.jit(complex_fn).lower(x_small).compile()
    complex_time = time.perf_counter() - start

    # TODO: Time compilation with larger input
    start = time.perf_counter()
    _ = jax.jit(medium_fn).lower(x_medium).compile()
    medium_larger_time = time.perf_counter() - start

    return {
        'simple_compile_ms': simple_time * 1000,
        'medium_compile_ms': medium_time * 1000,
        'complex_compile_ms': complex_time * 1000,
        'medium_larger_input_ms': medium_larger_time * 1000,
        'complexity_increases_time': complex_time > simple_time
    }


# =============================================================================
# Exercise 7: Reusing Compiled Functions
# =============================================================================
def exercise_reuse_compiled():
    """
    A compiled function can be called many times efficiently.

    This is the main benefit of AOT:
    - Compile once during initialization
    - Call thousands of times during serving
    - Each call has consistent, predictable latency

    Production pattern:
    ```
    # At startup
    compiled_model = jax.jit(model).lower(example_input).compile()

    # During serving (called thousands of times)
    for request in requests:
        result = compiled_model(request.data)
    ```

    Your task:
    1. Compile a function once
    2. Call it many times with different values (same shape!)
    3. Measure timing consistency across calls

    Hints:
    - All calls should have similar execution time
    - No "first call is slow" like with JIT
    - Values can change, shapes must stay the same
    """
    def normalize(x):
        """Normalize to zero mean, unit variance."""
        return (x - jnp.mean(x)) / (jnp.std(x) + 1e-8)

    # Compile once
    example_input = jnp.ones((1000,))
    compiled = jax.jit(normalize).lower(example_input).compile()

    # TODO: Generate different inputs (same shape, different values)
    inputs = [
        jnp.ones((1000,)) * i
        for i in range(1, 11)
    ]

    # TODO: Time each call
    times = []
    results = []
    for inp in inputs:
        start = time.perf_counter()
        result = compiled(inp)
        result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        results.append(result)

    times_ms = [t * 1000 for t in times]
    avg_time = sum(times_ms) / len(times_ms)
    max_time = max(times_ms)
    min_time = min(times_ms)

    return {
        'num_calls': len(inputs),
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'time_variance_ms': max_time - min_time,
        'consistent_timing': (max_time - min_time) < avg_time  # variance < mean
    }


# =============================================================================
# Exercise 8: AOT for Multiple Functions (Model Components)
# =============================================================================
def exercise_aot_model_components():
    """
    Real models have multiple functions. You can AOT compile each one.

    Common pattern for a neural network:
    - compile forward pass
    - compile backward pass (if training)
    - compile update step

    For inference-only deployment:
    - Only compile the forward pass
    - Smaller compiled artifact
    - Faster startup

    Your task:
    1. Create encoder and decoder functions
    2. AOT compile both
    3. Chain them together as a pipeline

    Hints:
    - Each function is compiled separately
    - You can call compiled functions in sequence
    - Make sure shapes match between encoder output and decoder input
    """
    def encoder(x, W_enc):
        """Encode: project to smaller dimension."""
        return jnp.tanh(x @ W_enc)

    def decoder(z, W_dec):
        """Decode: project back to original dimension."""
        return jnp.tanh(z @ W_dec)

    # Shapes: input 64 -> latent 16 -> output 64
    input_dim, latent_dim = 64, 16
    batch_size = 8

    # Example inputs for compilation
    x = jnp.ones((batch_size, input_dim))
    W_enc = jnp.ones((input_dim, latent_dim)) * 0.1
    W_dec = jnp.ones((latent_dim, input_dim)) * 0.1

    # TODO: Compile encoder
    compiled_encoder = jax.jit(encoder).lower(x, W_enc).compile()

    # Get encoder output shape for decoder compilation
    z_example = jnp.ones((batch_size, latent_dim))

    # TODO: Compile decoder
    compiled_decoder = jax.jit(decoder).lower(z_example, W_dec).compile()

    # TODO: Run the full pipeline
    z = compiled_encoder(x, W_enc)
    reconstruction = compiled_decoder(z, W_dec)

    return {
        'input_shape': x.shape,
        'latent_shape': z.shape,
        'output_shape': reconstruction.shape,
        'shapes_match': x.shape == reconstruction.shape,
        'pipeline_works': True
    }


# =============================================================================
# Exercise 9: Inspecting Compilation Statistics
# =============================================================================
def exercise_compilation_stats():
    """
    You can get statistics about the compiled function:
    - Memory usage
    - Number of operations
    - Optimization information

    This helps with:
    - Memory planning for deployment
    - Understanding optimization effectiveness
    - Debugging performance issues

    Your task:
    1. Compile a function
    2. Extract compilation statistics
    3. Compare stats for different functions

    Hints:
    - compiled.cost_analysis() gives memory/compute estimates
    - lowered.as_text() shows the operation count
    - These are useful for production capacity planning
    """
    def small_fn(x):
        return x * 2

    def large_fn(x):
        x = x @ x.T
        x = jnp.sin(x)
        x = x @ x.T
        x = jnp.cos(x)
        return x

    x = jnp.ones((100, 100))

    # TODO: Compile both functions
    lowered_small = jax.jit(small_fn).lower(x)
    compiled_small = lowered_small.compile()

    lowered_large = jax.jit(large_fn).lower(x)
    compiled_large = lowered_large.compile()

    # TODO: Get HLO sizes (proxy for complexity)
    hlo_small = lowered_small.as_text()
    hlo_large = lowered_large.as_text()

    # TODO: Try to get cost analysis (may not be available on all backends)
    try:
        cost_small = compiled_small.cost_analysis()
        cost_large = compiled_large.cost_analysis()
        has_cost_analysis = True
    except Exception:
        cost_small = None
        cost_large = None
        has_cost_analysis = False

    return {
        'small_hlo_chars': len(hlo_small),
        'large_hlo_chars': len(hlo_large),
        'large_more_complex': len(hlo_large) > len(hlo_small),
        'has_cost_analysis': has_cost_analysis,
        'cost_analysis_small': str(cost_small)[:200] if cost_small else None,
        'cost_analysis_large': str(cost_large)[:200] if cost_large else None
    }


# =============================================================================
# Exercise 10: Production-Ready AOT Pattern
# =============================================================================
def exercise_production_pattern():
    """
    This exercise demonstrates a production-ready AOT pattern.

    PRODUCTION AOT CHECKLIST:
    1. Define your input shapes upfront
    2. Compile all functions during initialization
    3. Handle multiple batch sizes if needed
    4. Validate inputs before calling compiled functions
    5. Measure and log execution times

    ERROR HANDLING:
    - Shape mismatches will raise errors
    - Always validate input shapes in production
    - Have fallback for unexpected shapes (recompile or reject)

    Your task:
    1. Create a "model server" class that pre-compiles functions
    2. Implement input validation
    3. Demonstrate the full serving pattern

    Hints:
    - Store compiled functions as instance variables
    - Validate shapes before calling compiled functions
    - This pattern is used in real ML serving systems
    """

    class AOTModelServer:
        """Example of production AOT pattern."""

        def __init__(self, input_dim: int, output_dim: int, batch_size: int):
            """Compile the model during initialization."""
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.batch_size = batch_size

            # Model weights (in practice, load from checkpoint)
            self.W = jnp.ones((input_dim, output_dim)) * 0.1
            self.b = jnp.zeros((output_dim,))

            # TODO: Create example input for compilation
            example_input = jnp.ones((batch_size, input_dim))

            # TODO: Define and compile the predict function
            def predict(x, W, b):
                return jnp.maximum(0, x @ W + b)  # ReLU(x @ W + b)

            self._compiled_predict = (
                jax.jit(predict)
                .lower(example_input, self.W, self.b)
                .compile()
            )

        def validate_input(self, x):
            """Validate input shape before prediction."""
            expected_shape = (self.batch_size, self.input_dim)
            if x.shape != expected_shape:
                raise ValueError(
                    f"Expected shape {expected_shape}, got {x.shape}"
                )

        def predict(self, x):
            """Run prediction with compiled function."""
            self.validate_input(x)
            return self._compiled_predict(x, self.W, self.b)

    # TODO: Create server with specific shapes
    server = AOTModelServer(input_dim=64, output_dim=32, batch_size=16)

    # TODO: Test with valid input
    valid_input = jnp.ones((16, 64))
    result = server.predict(valid_input)

    # TODO: Test with invalid input (should raise error)
    invalid_error = None
    try:
        invalid_input = jnp.ones((8, 64))  # Wrong batch size
        _ = server.predict(invalid_input)
    except ValueError as e:
        invalid_error = str(e)

    # Measure inference time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = server.predict(valid_input)
        times.append(time.perf_counter() - start)

    avg_inference_ms = sum(times) / len(times) * 1000

    return {
        'output_shape': result.shape,
        'correct_output_shape': result.shape == (16, 32),
        'validation_works': invalid_error is not None,
        'validation_error': invalid_error,
        'avg_inference_ms': avg_inference_ms,
        'production_ready': True
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("JAX AOT (Ahead-of-Time) Compilation Exercises")
    print("=" * 60)

    exercises = [
        ("1. Basic AOT Pipeline", exercise_basic_aot),
        ("2. Inspecting HLO", exercise_inspect_hlo),
        ("3. AOT with Different Shapes", exercise_aot_shapes),
        ("4. Timing: Compile vs Execute", exercise_timing_aot),
        ("5. AOT with Static Arguments", exercise_aot_static_args),
        ("6. Compilation Cost Analysis", exercise_compilation_cost),
        ("7. Reusing Compiled Functions", exercise_reuse_compiled),
        ("8. AOT Model Components", exercise_aot_model_components),
        ("9. Compilation Statistics", exercise_compilation_stats),
        ("10. Production-Ready Pattern", exercise_production_pattern),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 80:
                    str_value = str_value[:77] + '...'
                print(f"  {key}: {str_value}")
        except Exception as e:
            print(f"  Error: {e}")
