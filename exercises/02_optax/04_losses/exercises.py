"""
Optax Losses - 10 Exercises
===========================

Optax provides common loss functions for training neural networks.
These are designed to work well with JAX transformations.

Reference: https://optax.readthedocs.io/en/latest/api/losses.html
"""

import jax
import jax.numpy as jnp
import optax


# =============================================================================
# Exercise 1: Softmax Cross-Entropy
# =============================================================================
def exercise_softmax_cross_entropy():
    """
    Softmax cross-entropy for multi-class classification.
    Takes logits and one-hot labels.

    Loss = -sum(labels * log(softmax(logits)))
    """
    # Logits (raw network outputs, before softmax)
    logits = jnp.array([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [0.1, 0.2, 3.0]])

    # One-hot labels
    labels = jnp.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=jnp.float32)

    # TODO: Implement this function
    # 1. Compute loss per sample using optax.softmax_cross_entropy
    # 2. Compute mean loss (typical for training)
    # 3. Verify with manual computation using jax.nn.log_softmax
    losses = None
    mean_loss = None
    log_probs = None
    manual_losses = None

    return {
        'per_sample_losses': losses,
        'mean_loss': float(mean_loss) if mean_loss is not None else None,
        'manual_matches': jnp.allclose(losses, manual_losses) if losses is not None and manual_losses is not None else None
    }


# =============================================================================
# Exercise 2: Softmax Cross-Entropy with Integer Labels
# =============================================================================
def exercise_softmax_cross_entropy_int_labels():
    """
    Simpler API: takes integer labels instead of one-hot.
    More memory efficient for large vocabularies.
    """
    logits = jnp.array([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [0.1, 0.2, 3.0]])

    # Integer labels (class indices)
    labels = jnp.array([0, 1, 2])

    # TODO: Implement this function
    # 1. Compute loss using optax.softmax_cross_entropy_with_integer_labels
    # 2. Compare with one-hot version using jax.nn.one_hot and optax.softmax_cross_entropy
    losses = None
    one_hot_labels = None
    losses_onehot = None

    return {
        'losses': losses,
        'matches_onehot': jnp.allclose(losses, losses_onehot) if losses is not None and losses_onehot is not None else None,
        'mean_loss': float(jnp.mean(losses)) if losses is not None else None
    }


# =============================================================================
# Exercise 3: Sigmoid Binary Cross-Entropy
# =============================================================================
def exercise_sigmoid_binary_cross_entropy():
    """
    Binary cross-entropy with sigmoid activation.
    For binary classification or multi-label classification.

    Loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
    """
    # Logits for binary classification
    logits = jnp.array([[-1.0], [0.0], [1.0], [2.0]])

    # Binary labels
    labels = jnp.array([[0.0], [0.0], [1.0], [1.0]])

    # Multi-label: each sample can have multiple labels
    multi_logits = jnp.array([[1.0, -1.0, 2.0],
                              [-2.0, 1.0, 0.5]])

    multi_labels = jnp.array([[1.0, 0.0, 1.0],   # Classes 0 and 2
                              [0.0, 1.0, 0.0]])  # Class 1 only

    # TODO: Implement this function
    # 1. Compute binary loss using optax.sigmoid_binary_cross_entropy
    # 2. Compute multi-label losses and mean over labels
    losses = None
    multi_losses = None
    multi_loss = None

    return {
        'binary_losses': losses.flatten() if losses is not None else None,
        'multi_label_losses': multi_loss,
        'mean_binary_loss': float(jnp.mean(losses)) if losses is not None else None
    }


# =============================================================================
# Exercise 4: L2 Loss (Mean Squared Error)
# =============================================================================
def exercise_l2_loss():
    """
    L2 loss for regression tasks.
    optax.l2_loss returns 0.5 * (pred - target)^2 (per element).
    """
    predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
    targets = jnp.array([1.5, 2.0, 2.5, 4.5])

    # TODO: Implement this function
    # 1. Compute L2 loss (element-wise) using optax.l2_loss
    # 2. Compute MSE (multiply by 2 because optax uses 0.5 factor)
    # 3. Compute manual MSE for comparison
    # 4. Compute total loss (sum)
    losses = None
    mse = None
    manual_mse = None
    total_loss = None

    return {
        'element_losses': losses,
        'mse': float(mse) if mse is not None else None,
        'manual_mse': float(manual_mse) if manual_mse is not None else None,
        'mse_matches': jnp.allclose(mse, manual_mse) if mse is not None and manual_mse is not None else None
    }


# =============================================================================
# Exercise 5: Huber Loss (Robust Regression)
# =============================================================================
def exercise_huber_loss():
    """
    Huber loss: L2 for small errors, L1 for large errors.
    More robust to outliers than pure L2.

    delta controls the transition point.
    """
    predictions = jnp.array([1.0, 2.0, 3.0, 10.0])  # 10.0 is an outlier
    targets = jnp.array([1.0, 2.0, 3.0, 3.0])

    # TODO: Implement this function
    # 1. Compute Huber loss with delta=1.0 using optax.huber_loss
    # 2. Compare with L2 loss using optax.l2_loss
    # 3. Check that Huber is more robust (less affected by outlier)
    delta = 1.0
    huber_losses = None
    l2_losses = None
    huber_total = None
    l2_total = None

    return {
        'huber_losses': huber_losses,
        'l2_losses': l2_losses,
        'huber_total': float(huber_total) if huber_total is not None else None,
        'l2_total': float(l2_total) if l2_total is not None else None,
        'huber_more_robust': huber_total < l2_total if huber_total is not None and l2_total is not None else None
    }


# =============================================================================
# Exercise 6: Cosine Similarity Loss
# =============================================================================
def exercise_cosine_similarity():
    """
    Cosine similarity for learning embeddings.
    Measures angle between vectors (ignores magnitude).
    """
    # Embedding vectors
    pred = jnp.array([[1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    target = jnp.array([[1.0, 0.0, 0.0],   # Same as pred[0]
                        [0.0, 1.0, 0.0],   # Orthogonal to pred[1]
                        [-1.0, 0.0, 0.0]]) # Opposite direction

    # TODO: Implement this function
    # 1. Compute cosine similarity using optax.cosine_similarity
    #    (1 = same, 0 = orthogonal, -1 = opposite)
    # 2. Compute cosine distance (1 - similarity) using optax.cosine_distance
    similarities = None
    distances = None

    return {
        'similarities': similarities,
        'distances': distances,
        'same_vectors': float(similarities[0]) if similarities is not None else None,
        'orthogonal': float(similarities[1]) if similarities is not None else None,
    }


# =============================================================================
# Exercise 7: Custom Loss Functions
# =============================================================================
def exercise_custom_loss():
    """
    Build custom loss functions using JAX primitives.
    """
    def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
        """
        Focal loss for class imbalance.
        Reduces loss for well-classified examples.

        FL = -alpha * (1 - p)^gamma * log(p)  for positive class
        """
        # TODO: Implement focal loss
        # 1. Compute softmax probabilities using jax.nn.softmax
        # 2. Get probability of true class using one_hot and sum
        # 3. Compute focal weight: alpha * (1 - pt) ** gamma
        # 4. Compute cross-entropy: -jnp.log(pt + 1e-8)
        # 5. Return focal_weight * ce
        probs = None
        num_classes = logits.shape[-1]
        one_hot = None
        pt = None
        focal_weight = None
        ce = None

        return None

    # Class imbalanced data
    logits = jnp.array([[3.0, 0.1, 0.1],   # Confident correct (class 0)
                        [0.1, 0.1, 0.1],   # Uncertain
                        [0.5, 2.0, 0.5]])  # Somewhat confident (class 1)

    labels = jnp.array([0, 0, 1])

    focal_losses = focal_loss(logits, labels)
    ce_losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    return {
        'focal_losses': focal_losses,
        'ce_losses': ce_losses,
        'focal_reduces_easy': focal_losses[0] < ce_losses[0] if focal_losses is not None else None
    }


# =============================================================================
# Exercise 8: Label Smoothing
# =============================================================================
def exercise_label_smoothing():
    """
    Label smoothing: regularization technique.
    Converts hard labels (0/1) to soft labels (epsilon/(K-1), 1-epsilon).
    """
    def smooth_labels(labels, num_classes, smoothing=0.1):
        """Convert hard labels to smoothed soft labels."""
        # TODO: Implement label smoothing
        # 1. Convert to one-hot using jax.nn.one_hot
        # 2. Return: one_hot * (1 - smoothing) + smoothing / num_classes
        one_hot = None
        return None

    def cross_entropy_with_smoothing(logits, labels, smoothing=0.1):
        """Cross-entropy with label smoothing."""
        # TODO: Implement cross-entropy with smoothing
        # 1. Get num_classes from logits shape
        # 2. Get soft labels using smooth_labels
        # 3. Return optax.softmax_cross_entropy with soft labels
        num_classes = None
        soft_labels = None
        return None

    logits = jnp.array([[3.0, 1.0, 0.5],
                        [0.5, 2.5, 1.0]])
    labels = jnp.array([0, 1])

    # Without smoothing
    hard_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # With smoothing
    smooth_loss = cross_entropy_with_smoothing(logits, labels, smoothing=0.1)

    # Smoothed labels
    num_classes = 3
    smoothed = smooth_labels(labels, num_classes, smoothing=0.1)

    return {
        'hard_labels': jax.nn.one_hot(labels, num_classes),
        'smoothed_labels': smoothed,
        'hard_loss': float(jnp.mean(hard_loss)),
        'smooth_loss': float(jnp.mean(smooth_loss)) if smooth_loss is not None else None,
        'smoothing_increases_loss': float(jnp.mean(smooth_loss)) > float(jnp.mean(hard_loss)) if smooth_loss is not None else None
    }


# =============================================================================
# Exercise 9: Combining Multiple Losses
# =============================================================================
def exercise_combined_loss():
    """
    Combine multiple loss terms (e.g., task loss + regularization).
    """
    def combined_loss(params, x, y, l2_weight=0.01):
        """
        Total loss = classification loss + L2 regularization.
        """
        # TODO: Implement combined loss
        # 1. Forward pass: logits = params['w'] @ x.T + params['b'][:, None], then transpose
        # 2. Classification loss using optax.softmax_cross_entropy_with_integer_labels
        # 3. L2 regularization: sum of squared parameters using jax.tree.leaves
        # 4. Return total loss and dict with components
        logits = None
        cls_loss = None
        l2_loss = None

        return None, {'cls': cls_loss, 'l2': l2_loss}

    # Setup
    params = {
        'w': jnp.ones((10, 5)) * 0.1,
        'b': jnp.zeros(10)
    }
    x = jnp.ones((32, 5))  # Batch of 32
    y = jnp.zeros(32, dtype=jnp.int32)  # All class 0

    total_loss, loss_components = combined_loss(params, x, y)

    # Get gradients for combined loss
    grad_fn = jax.grad(lambda p: combined_loss(p, x, y)[0])
    grads = grad_fn(params) if total_loss is not None else None

    return {
        'total_loss': float(total_loss) if total_loss is not None else None,
        'classification_loss': float(loss_components['cls']) if loss_components['cls'] is not None else None,
        'regularization_loss': float(loss_components['l2']) if loss_components['l2'] is not None else None,
        'has_gradients': 'w' in grads and 'b' in grads if grads is not None else None
    }


# =============================================================================
# Exercise 10: Loss Functions for Different Tasks
# =============================================================================
def exercise_task_specific_losses():
    """
    Summary of loss functions for different tasks.
    """
    tasks = {
        'binary_classification': {
            'loss': 'sigmoid_binary_cross_entropy',
            'example': 'spam detection, sentiment analysis',
            'output': 'single logit'
        },
        'multi_class': {
            'loss': 'softmax_cross_entropy',
            'example': 'ImageNet, MNIST',
            'output': 'logits for each class'
        },
        'multi_label': {
            'loss': 'sigmoid_binary_cross_entropy (per label)',
            'example': 'image tagging, multi-topic classification',
            'output': 'logits for each label'
        },
        'regression': {
            'loss': 'l2_loss (MSE)',
            'example': 'price prediction, age estimation',
            'output': 'continuous value'
        },
        'robust_regression': {
            'loss': 'huber_loss',
            'example': 'regression with outliers',
            'output': 'continuous value'
        },
        'embedding_learning': {
            'loss': 'cosine_distance or contrastive',
            'example': 'face recognition, semantic similarity',
            'output': 'embedding vectors'
        }
    }

    # Quick demonstration of each
    # Binary
    binary_logits = jnp.array([[1.0]])
    binary_labels = jnp.array([[1.0]])

    # Multi-class
    mc_logits = jnp.array([[2.0, 0.5, 0.1]])
    mc_labels = jnp.array([0])

    # Regression
    pred = jnp.array([2.5])
    target = jnp.array([3.0])

    # TODO: Implement this function
    # 1. Compute binary loss using optax.sigmoid_binary_cross_entropy
    # 2. Compute multi-class loss using optax.softmax_cross_entropy_with_integer_labels
    # 3. Compute regression loss using optax.l2_loss
    binary_loss = None
    mc_loss = None
    reg_loss = None

    return {
        'tasks': list(tasks.keys()),
        'binary_loss': float(binary_loss[0, 0]) if binary_loss is not None else None,
        'multiclass_loss': float(mc_loss[0]) if mc_loss is not None else None,
        'regression_loss': float(reg_loss[0]) if reg_loss is not None else None
    }


# =============================================================================
# Run all exercises
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Losses Exercises")
    print("=" * 60)

    exercises = [
        ("1. Softmax Cross-Entropy", exercise_softmax_cross_entropy),
        ("2. CE with Integer Labels", exercise_softmax_cross_entropy_int_labels),
        ("3. Sigmoid Binary CE", exercise_sigmoid_binary_cross_entropy),
        ("4. L2 Loss", exercise_l2_loss),
        ("5. Huber Loss", exercise_huber_loss),
        ("6. Cosine Similarity", exercise_cosine_similarity),
        ("7. Custom Loss", exercise_custom_loss),
        ("8. Label Smoothing", exercise_label_smoothing),
        ("9. Combined Loss", exercise_combined_loss),
        ("10. Task-Specific Losses", exercise_task_specific_losses),
    ]

    for name, func in exercises:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = func()
            for key, value in result.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Error: {e}")
