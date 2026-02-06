"""
Optax Learning Rate Schedules - 10 Examples
============================================

Learning rate schedules adjust the learning rate during training.
Critical for achieving good convergence.

Key pattern:
    schedule = optax.warmup_cosine_decay_schedule(...)
    optimizer = optax.adam(schedule)
    # or
    optimizer = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule))

Reference: https://optax.readthedocs.io/en/latest/api/schedules.html
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# =============================================================================
# Example 1: Constant Schedule
# =============================================================================
def example_constant_schedule():
    """
    Constant learning rate - the simplest schedule.
    Useful as a baseline.
    """
    schedule = optax.constant_schedule(0.001)

    # Schedule is a function: step -> learning_rate
    lr_at_0 = schedule(0)
    lr_at_100 = schedule(100)
    lr_at_1000 = schedule(1000)

    # Use with optimizer
    optimizer = optax.adam(schedule)

    return {
        'lr_at_0': float(lr_at_0),
        'lr_at_100': float(lr_at_100),
        'lr_at_1000': float(lr_at_1000),
        'is_constant': lr_at_0 == lr_at_100 == lr_at_1000
    }


# =============================================================================
# Example 2: Exponential Decay
# =============================================================================
def example_exponential_decay():
    """
    Exponential decay: lr = init_lr * decay_rate^(step / decay_steps)
    Smooth decay that never reaches zero.
    """
    schedule = optax.exponential_decay(
        init_value=0.1,
        transition_steps=1000,
        decay_rate=0.96,
        staircase=False  # Smooth decay vs step-wise
    )

    # Sample learning rates
    steps = [0, 500, 1000, 2000, 5000]
    lrs = [float(schedule(s)) for s in steps]

    # Staircase version (step-wise decay)
    schedule_staircase = optax.exponential_decay(
        init_value=0.1,
        transition_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    lrs_staircase = [float(schedule_staircase(s)) for s in steps]

    return {
        'steps': steps,
        'smooth_lrs': lrs,
        'staircase_lrs': lrs_staircase,
        'decays_over_time': lrs[0] > lrs[-1]
    }


# =============================================================================
# Example 3: Cosine Decay Schedule
# =============================================================================
def example_cosine_decay():
    """
    Cosine decay: smooth decay following cosine curve.
    Popular for training transformers and CNNs.

    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / decay_steps))
    """
    schedule = optax.cosine_decay_schedule(
        init_value=0.1,
        decay_steps=10000,
        alpha=0.0  # Final lr as fraction of init (0.0 = decay to 0)
    )

    # Sample learning rates
    steps = [0, 2500, 5000, 7500, 10000]
    lrs = [float(schedule(s)) for s in steps]

    # Cosine with minimum lr
    schedule_with_min = optax.cosine_decay_schedule(
        init_value=0.1,
        decay_steps=10000,
        alpha=0.01  # Final lr = 0.1 * 0.01 = 0.001
    )
    lrs_with_min = [float(schedule_with_min(s)) for s in steps]

    return {
        'steps': steps,
        'cosine_lrs': lrs,
        'cosine_with_min': lrs_with_min,
        'starts_at_init': abs(lrs[0] - 0.1) < 1e-6,
        'ends_near_zero': lrs[-1] < 0.01
    }


# =============================================================================
# Example 4: Warmup + Cosine Decay (Essential for Transformers)
# =============================================================================
def example_warmup_cosine():
    """
    Warmup followed by cosine decay.
    ESSENTIAL for training transformers!

    1. Linear warmup from 0 to peak_value
    2. Cosine decay from peak to end_value
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,           # Start from 0
        peak_value=0.001,         # Warmup to this
        warmup_steps=1000,        # Warmup duration
        decay_steps=9000,         # Total steps (including warmup)
        end_value=0.0001          # Final learning rate
    )

    # Sample at key points
    steps = [0, 500, 1000, 5000, 10000]
    lrs = [float(schedule(s)) for s in steps]

    # Alternative: warmup_exponential_decay
    schedule_exp = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=0.001,
        warmup_steps=1000,
        transition_steps=2000,
        decay_rate=0.96
    )

    return {
        'steps': steps,
        'warmup_cosine_lrs': lrs,
        'warmup_linear': lrs[1] < lrs[2],  # Increasing during warmup
        'decay_after_peak': lrs[2] > lrs[3] > lrs[4]  # Decreasing after peak
    }


# =============================================================================
# Example 5: Linear Schedule
# =============================================================================
def example_linear_schedule():
    """
    Linear interpolation between two values.
    Useful for warmup or linear decay.
    """
    # Linear warmup
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=0.001,
        transition_steps=1000
    )

    warmup_lrs = [float(warmup_schedule(s)) for s in [0, 250, 500, 750, 1000]]

    # Linear decay
    decay_schedule = optax.linear_schedule(
        init_value=0.001,
        end_value=0.0,
        transition_steps=10000
    )

    decay_lrs = [float(decay_schedule(s)) for s in [0, 2500, 5000, 7500, 10000]]

    return {
        'warmup_lrs': warmup_lrs,
        'decay_lrs': decay_lrs,
        'warmup_increases': warmup_lrs[-1] > warmup_lrs[0],
        'decay_decreases': decay_lrs[-1] < decay_lrs[0]
    }


# =============================================================================
# Example 6: Piecewise Constant Schedule (Step Decay)
# =============================================================================
def example_piecewise_constant():
    """
    Step decay: fixed lr that drops at specified steps.
    Common in computer vision (e.g., drop by 10x every 30 epochs).
    """
    # Drop lr at steps 1000 and 2000
    schedule = optax.piecewise_constant_schedule(
        init_value=0.1,
        boundaries_and_scales={
            1000: 0.1,   # Multiply by 0.1 at step 1000
            2000: 0.1    # Multiply by 0.1 again at step 2000
        }
    )

    steps = [0, 500, 1000, 1500, 2000, 2500]
    lrs = [float(schedule(s)) for s in steps]

    # Expected: 0.1, 0.1, 0.01, 0.01, 0.001, 0.001

    return {
        'steps': steps,
        'piecewise_lrs': lrs,
        'initial_lr': lrs[0],
        'after_first_drop': lrs[2],
        'after_second_drop': lrs[4]
    }


# =============================================================================
# Example 7: Join Schedules
# =============================================================================
def example_join_schedules():
    """
    Combine multiple schedules sequentially.
    """
    # Warmup -> constant -> decay
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, 0.001, 1000),      # Warmup
            optax.constant_schedule(0.001),               # Constant
            optax.cosine_decay_schedule(0.001, 5000)      # Decay
        ],
        boundaries=[1000, 3000]  # Switch points
    )

    steps = [0, 500, 1000, 2000, 3000, 5000, 8000]
    lrs = [float(schedule(s)) for s in steps]

    return {
        'steps': steps,
        'joined_lrs': lrs,
        'warmup_phase': lrs[0] < lrs[2],
        'constant_phase': abs(lrs[2] - lrs[3]) < 1e-6,
        'decay_phase': lrs[4] > lrs[6]
    }


# =============================================================================
# Example 8: inject_hyperparams - Making LR Trackable
# =============================================================================
def example_inject_hyperparams():
    """
    inject_hyperparams exposes hyperparameters in optimizer state.
    Useful for logging the current learning rate.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.001,
        warmup_steps=100,
        decay_steps=1000
    )

    # Wrap optimizer to track hyperparameters
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=schedule)

    params = {'w': jnp.ones(10)}
    opt_state = optimizer.init(params)

    # Access current learning rate from state
    current_lr_0 = opt_state.hyperparams['learning_rate']

    # Advance a few steps
    grads = {'w': jnp.ones(10) * 0.1}
    for _ in range(100):
        _, opt_state = optimizer.update(grads, opt_state, params)

    current_lr_100 = opt_state.hyperparams['learning_rate']

    return {
        'lr_at_step_0': float(current_lr_0),
        'lr_at_step_100': float(current_lr_100),
        'lr_trackable': True
    }


# =============================================================================
# Example 9: Custom Schedule Function
# =============================================================================
def example_custom_schedule():
    """
    Create custom schedules as Python functions.
    """
    def triangle_schedule(peak_value, total_steps):
        """Triangle: increase then decrease."""
        def schedule(step):
            half = total_steps // 2
            if step < half:
                return peak_value * step / half
            else:
                return peak_value * (total_steps - step) / half
        return schedule

    schedule = triangle_schedule(0.001, 10000)

    steps = [0, 2500, 5000, 7500, 10000]
    lrs = [float(schedule(s)) for s in steps]

    # One-cycle schedule (popular for training)
    def one_cycle_schedule(max_lr, total_steps, pct_start=0.3):
        """One-cycle: warmup + annealing."""
        def schedule(step):
            if step < total_steps * pct_start:
                # Warmup phase
                return max_lr * step / (total_steps * pct_start)
            else:
                # Annealing phase (cosine)
                progress = (step - total_steps * pct_start) / (total_steps * (1 - pct_start))
                return max_lr * (1 + jnp.cos(jnp.pi * progress)) / 2
        return schedule

    one_cycle = one_cycle_schedule(0.001, 10000)
    one_cycle_lrs = [float(one_cycle(s)) for s in steps]

    return {
        'triangle_lrs': lrs,
        'one_cycle_lrs': one_cycle_lrs,
        'triangle_peaks_at_middle': lrs[2] == max(lrs)
    }


# =============================================================================
# Example 10: Visualizing Schedules
# =============================================================================
def example_visualize_schedules():
    """
    Visualize different schedule curves.
    """
    total_steps = 10000

    schedules = {
        'constant': optax.constant_schedule(0.001),
        'exponential': optax.exponential_decay(0.001, 2000, 0.9),
        'cosine': optax.cosine_decay_schedule(0.001, total_steps),
        'warmup_cosine': optax.warmup_cosine_decay_schedule(
            0.0, 0.001, 1000, total_steps
        ),
        'piecewise': optax.piecewise_constant_schedule(
            0.001, {3000: 0.1, 6000: 0.1}
        ),
    }

    steps = jnp.arange(0, total_steps, 100)
    curves = {}

    for name, schedule in schedules.items():
        curves[name] = [float(schedule(int(s))) for s in steps]

    # Create visualization (save to file instead of showing)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, lrs in curves.items():
        ax.plot(steps, lrs, label=name)

    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Save to file
    plt.savefig('/tmp/lr_schedules.png', dpi=100, bbox_inches='tight')
    plt.close()

    return {
        'schedules_compared': list(schedules.keys()),
        'num_points': len(steps),
        'plot_saved': '/tmp/lr_schedules.png'
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Learning Rate Schedules Examples")
    print("=" * 60)

    examples = [
        ("1. Constant Schedule", example_constant_schedule),
        ("2. Exponential Decay", example_exponential_decay),
        ("3. Cosine Decay", example_cosine_decay),
        ("4. Warmup + Cosine", example_warmup_cosine),
        ("5. Linear Schedule", example_linear_schedule),
        ("6. Piecewise Constant", example_piecewise_constant),
        ("7. Join Schedules", example_join_schedules),
        ("8. inject_hyperparams", example_inject_hyperparams),
        ("9. Custom Schedule", example_custom_schedule),
        ("10. Visualize Schedules", example_visualize_schedules),
    ]

    for name, func in examples:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
