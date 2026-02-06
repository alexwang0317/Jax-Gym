"""
Optax Learning Rate Schedules - 10 Exercises
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
# Exercise 1: Constant Schedule
# =============================================================================
def exercise_constant_schedule():
    """
    Constant learning rate - the simplest schedule.
    Useful as a baseline.
    """
    # TODO: Implement this function
    # 1. Create a constant schedule with value 0.001
    # 2. Evaluate the schedule at steps 0, 100, and 1000
    # 3. Create an adam optimizer with the schedule
    schedule = None

    lr_at_0 = None
    lr_at_100 = None
    lr_at_1000 = None

    return {
        'lr_at_0': float(lr_at_0) if lr_at_0 is not None else None,
        'lr_at_100': float(lr_at_100) if lr_at_100 is not None else None,
        'lr_at_1000': float(lr_at_1000) if lr_at_1000 is not None else None,
        'is_constant': lr_at_0 == lr_at_100 == lr_at_1000 if lr_at_0 is not None else None
    }


# =============================================================================
# Exercise 2: Exponential Decay
# =============================================================================
def exercise_exponential_decay():
    """
    Exponential decay: lr = init_lr * decay_rate^(step / decay_steps)
    Smooth decay that never reaches zero.
    """
    # TODO: Implement this function
    # 1. Create an exponential decay schedule with:
    #    - init_value=0.1
    #    - transition_steps=1000
    #    - decay_rate=0.96
    #    - staircase=False (smooth decay)
    # 2. Sample learning rates at steps [0, 500, 1000, 2000, 5000]
    # 3. Create a staircase version (staircase=True) and sample same steps
    schedule = None
    schedule_staircase = None

    steps = [0, 500, 1000, 2000, 5000]
    lrs = None
    lrs_staircase = None

    return {
        'steps': steps,
        'smooth_lrs': lrs,
        'staircase_lrs': lrs_staircase,
        'decays_over_time': lrs[0] > lrs[-1] if lrs is not None else None
    }


# =============================================================================
# Exercise 3: Cosine Decay Schedule
# =============================================================================
def exercise_cosine_decay():
    """
    Cosine decay: smooth decay following cosine curve.
    Popular for training transformers and CNNs.

    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * step / decay_steps))
    """
    # TODO: Implement this function
    # 1. Create a cosine decay schedule with:
    #    - init_value=0.1
    #    - decay_steps=10000
    #    - alpha=0.0 (decay to 0)
    # 2. Sample learning rates at steps [0, 2500, 5000, 7500, 10000]
    # 3. Create another schedule with alpha=0.01 (minimum lr) and sample same steps
    schedule = None
    schedule_with_min = None

    steps = [0, 2500, 5000, 7500, 10000]
    lrs = None
    lrs_with_min = None

    return {
        'steps': steps,
        'cosine_lrs': lrs,
        'cosine_with_min': lrs_with_min,
        'starts_at_init': abs(lrs[0] - 0.1) < 1e-6 if lrs is not None else None,
        'ends_near_zero': lrs[-1] < 0.01 if lrs is not None else None
    }


# =============================================================================
# Exercise 4: Warmup + Cosine Decay (Essential for Transformers)
# =============================================================================
def exercise_warmup_cosine():
    """
    Warmup followed by cosine decay.
    ESSENTIAL for training transformers!

    1. Linear warmup from 0 to peak_value
    2. Cosine decay from peak to end_value
    """
    # TODO: Implement this function
    # 1. Create a warmup_cosine_decay_schedule with:
    #    - init_value=0.0
    #    - peak_value=0.001
    #    - warmup_steps=1000
    #    - decay_steps=9000 (total steps including warmup)
    #    - end_value=0.0001
    # 2. Sample at steps [0, 500, 1000, 5000, 10000]
    schedule = None

    steps = [0, 500, 1000, 5000, 10000]
    lrs = None

    return {
        'steps': steps,
        'warmup_cosine_lrs': lrs,
        'warmup_linear': lrs[1] < lrs[2] if lrs is not None else None,
        'decay_after_peak': lrs[2] > lrs[3] > lrs[4] if lrs is not None else None
    }


# =============================================================================
# Exercise 5: Linear Schedule
# =============================================================================
def exercise_linear_schedule():
    """
    Linear interpolation between two values.
    Useful for warmup or linear decay.
    """
    # TODO: Implement this function
    # 1. Create a linear warmup schedule:
    #    - init_value=0.0, end_value=0.001, transition_steps=1000
    # 2. Sample at steps [0, 250, 500, 750, 1000]
    # 3. Create a linear decay schedule:
    #    - init_value=0.001, end_value=0.0, transition_steps=10000
    # 4. Sample at steps [0, 2500, 5000, 7500, 10000]
    warmup_schedule = None
    decay_schedule = None

    warmup_lrs = None
    decay_lrs = None

    return {
        'warmup_lrs': warmup_lrs,
        'decay_lrs': decay_lrs,
        'warmup_increases': warmup_lrs[-1] > warmup_lrs[0] if warmup_lrs is not None else None,
        'decay_decreases': decay_lrs[-1] < decay_lrs[0] if decay_lrs is not None else None
    }


# =============================================================================
# Exercise 6: Piecewise Constant Schedule (Step Decay)
# =============================================================================
def exercise_piecewise_constant():
    """
    Step decay: fixed lr that drops at specified steps.
    Common in computer vision (e.g., drop by 10x every 30 epochs).
    """
    # TODO: Implement this function
    # 1. Create a piecewise constant schedule with:
    #    - init_value=0.1
    #    - boundaries_and_scales={1000: 0.1, 2000: 0.1}
    #      (multiply by 0.1 at step 1000, multiply by 0.1 again at step 2000)
    # 2. Sample at steps [0, 500, 1000, 1500, 2000, 2500]
    schedule = None

    steps = [0, 500, 1000, 1500, 2000, 2500]
    lrs = None

    return {
        'steps': steps,
        'piecewise_lrs': lrs,
        'initial_lr': lrs[0] if lrs is not None else None,
        'after_first_drop': lrs[2] if lrs is not None else None,
        'after_second_drop': lrs[4] if lrs is not None else None
    }


# =============================================================================
# Exercise 7: Join Schedules
# =============================================================================
def exercise_join_schedules():
    """
    Combine multiple schedules sequentially.
    """
    # TODO: Implement this function
    # 1. Create a joined schedule with:
    #    - Linear warmup from 0.0 to 0.001 over 1000 steps
    #    - Constant at 0.001
    #    - Cosine decay from 0.001 over 5000 steps
    #    - boundaries at [1000, 3000]
    # 2. Sample at steps [0, 500, 1000, 2000, 3000, 5000, 8000]
    schedule = None

    steps = [0, 500, 1000, 2000, 3000, 5000, 8000]
    lrs = None

    return {
        'steps': steps,
        'joined_lrs': lrs,
        'warmup_phase': lrs[0] < lrs[2] if lrs is not None else None,
        'constant_phase': abs(lrs[2] - lrs[3]) < 1e-6 if lrs is not None else None,
        'decay_phase': lrs[4] > lrs[6] if lrs is not None else None
    }


# =============================================================================
# Exercise 8: inject_hyperparams - Making LR Trackable
# =============================================================================
def exercise_inject_hyperparams():
    """
    inject_hyperparams exposes hyperparameters in optimizer state.
    Useful for logging the current learning rate.
    """
    # TODO: Implement this function
    # 1. Create a warmup_cosine_decay_schedule with:
    #    - init_value=0.0, peak_value=0.001
    #    - warmup_steps=100, decay_steps=1000
    # 2. Wrap adam optimizer with inject_hyperparams
    # 3. Initialize with params = {'w': jnp.ones(10)}
    # 4. Get current lr from opt_state.hyperparams['learning_rate']
    # 5. Run 100 update steps and get lr again
    schedule = None
    optimizer = None

    params = {'w': jnp.ones(10)}
    opt_state = None

    current_lr_0 = None
    current_lr_100 = None

    return {
        'lr_at_step_0': float(current_lr_0) if current_lr_0 is not None else None,
        'lr_at_step_100': float(current_lr_100) if current_lr_100 is not None else None,
        'lr_trackable': True
    }


# =============================================================================
# Exercise 9: Custom Schedule Function
# =============================================================================
def exercise_custom_schedule():
    """
    Create custom schedules as Python functions.
    """
    # TODO: Implement this function
    # 1. Implement triangle_schedule(peak_value, total_steps):
    #    - Returns a function that takes step and returns lr
    #    - First half: increase linearly from 0 to peak_value
    #    - Second half: decrease linearly from peak_value to 0
    # 2. Create schedule with peak_value=0.001, total_steps=10000
    # 3. Sample at steps [0, 2500, 5000, 7500, 10000]
    def triangle_schedule(peak_value, total_steps):
        """Triangle: increase then decrease."""
        def schedule(step):
            # TODO: Implement the triangle schedule logic
            return None
        return schedule

    schedule = triangle_schedule(0.001, 10000)

    steps = [0, 2500, 5000, 7500, 10000]
    lrs = None

    # One-cycle schedule (popular for training)
    def one_cycle_schedule(max_lr, total_steps, pct_start=0.3):
        """One-cycle: warmup + annealing."""
        def schedule(step):
            # TODO: Implement one-cycle schedule
            # First pct_start fraction: warmup from 0 to max_lr
            # Remaining: cosine annealing from max_lr to 0
            return None
        return schedule

    one_cycle = one_cycle_schedule(0.001, 10000)
    one_cycle_lrs = None

    return {
        'triangle_lrs': lrs,
        'one_cycle_lrs': one_cycle_lrs,
        'triangle_peaks_at_middle': lrs[2] == max(lrs) if lrs is not None else None
    }


# =============================================================================
# Exercise 10: Visualizing Schedules
# =============================================================================
def exercise_visualize_schedules():
    """
    Visualize different schedule curves.
    """
    # TODO: Implement this function
    # 1. Create schedules dict with:
    #    - 'constant': constant_schedule(0.001)
    #    - 'exponential': exponential_decay(0.001, 2000, 0.9)
    #    - 'cosine': cosine_decay_schedule(0.001, total_steps)
    #    - 'warmup_cosine': warmup_cosine_decay_schedule(0.0, 0.001, 1000, total_steps)
    #    - 'piecewise': piecewise_constant_schedule(0.001, {3000: 0.1, 6000: 0.1})
    # 2. Sample each schedule at steps from 0 to total_steps (step 100)
    # 3. Create a plot with all schedules
    total_steps = 10000

    schedules = None

    steps = jnp.arange(0, total_steps, 100)
    curves = {}

    return {
        'schedules_compared': list(schedules.keys()) if schedules is not None else None,
        'num_points': len(steps),
        'plot_saved': '/tmp/lr_schedules.png'
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Learning Rate Schedules Exercises")
    print("=" * 60)

    exercises = [
        ("1. Constant Schedule", exercise_constant_schedule),
        ("2. Exponential Decay", exercise_exponential_decay),
        ("3. Cosine Decay", exercise_cosine_decay),
        ("4. Warmup + Cosine", exercise_warmup_cosine),
        ("5. Linear Schedule", exercise_linear_schedule),
        ("6. Piecewise Constant", exercise_piecewise_constant),
        ("7. Join Schedules", exercise_join_schedules),
        ("8. inject_hyperparams", exercise_inject_hyperparams),
        ("9. Custom Schedule", exercise_custom_schedule),
        ("10. Visualize Schedules", exercise_visualize_schedules),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
