"""
Optax Losses - 10 Examples
===========================

Optax provides common loss functions for training neural networks.
These are designed to work well with JAX transformations.

Reference: https://optax.readthedocs.io/en/latest/api/losses.html
"""

import jax
import jax.numpy as jnp
import optax


# =============================================================================
# Example 1: Softmax Cross-Entropy
# =============================================================================
def example_softmax_cross_entropy():
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

    # Compute loss per sample
    losses = optax.softmax_cross_entropy(logits, labels)

    # Mean loss (typical for training)
    mean_loss = jnp.mean(losses)

    # Verify with manual computation
    log_probs = jax.nn.log_softmax(logits)
    manual_losses = -jnp.sum(labels * log_probs, axis=-1)

    return {
        'per_sample_losses': losses,
        'mean_loss': float(mean_loss),
        'manual_matches': jnp.allclose(losses, manual_losses)
    }


# =============================================================================
# Example 2: Softmax Cross-Entropy with Integer Labels
# =============================================================================
def example_softmax_cross_entropy_int_labels():
    """
    Simpler API: takes integer labels instead of one-hot.
    More memory efficient for large vocabularies.
    """
    logits = jnp.array([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3],
                        [0.1, 0.2, 3.0]])

    # Integer labels (class indices)
    labels = jnp.array([0, 1, 2])

    # Compute loss
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # Compare with one-hot version
    one_hot_labels = jax.nn.one_hot(labels, num_classes=3)
    losses_onehot = optax.softmax_cross_entropy(logits, one_hot_labels)

    return {
        'losses': losses,
        'matches_onehot': jnp.allclose(losses, losses_onehot),
        'mean_loss': float(jnp.mean(losses))
    }


# =============================================================================
# Example 3: Sigmoid Binary Cross-Entropy
# =============================================================================
def example_sigmoid_binary_cross_entropy():
    """
    Binary cross-entropy with sigmoid activation.
    For binary classification or multi-label classification.

    Loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
    """
    # Logits for binary classification
    logits = jnp.array([[-1.0], [0.0], [1.0], [2.0]])

    # Binary labels
    labels = jnp.array([[0.0], [0.0], [1.0], [1.0]])

    # Compute loss
    losses = optax.sigmoid_binary_cross_entropy(logits, labels)

    # Multi-label: each sample can have multiple labels
    multi_logits = jnp.array([[1.0, -1.0, 2.0],
                              [-2.0, 1.0, 0.5]])

    multi_labels = jnp.array([[1.0, 0.0, 1.0],   # Classes 0 and 2
                              [0.0, 1.0, 0.0]])  # Class 1 only

    multi_losses = optax.sigmoid_binary_cross_entropy(multi_logits, multi_labels)
    # Mean over labels
    multi_loss = jnp.mean(multi_losses, axis=-1)

    return {
        'binary_losses': losses.flatten(),
        'multi_label_losses': multi_loss,
        'mean_binary_loss': float(jnp.mean(losses))
    }


# =============================================================================
# Example 4: L2 Loss (Mean Squared Error)
# =============================================================================
def example_l2_loss():
    """
    L2 loss for regression tasks.
    optax.l2_loss returns 0.5 * (pred - target)^2 (per element).
    """
    predictions = jnp.array([1.0, 2.0, 3.0, 4.0])
    targets = jnp.array([1.5, 2.0, 2.5, 4.5])

    # L2 loss (element-wise)
    losses = optax.l2_loss(predictions, targets)

    # Mean over elements (MSE)
    mse = jnp.mean(losses) * 2  # Multiply by 2 because optax uses 0.5 factor

    # Standard MSE for comparison
    manual_mse = jnp.mean((predictions - targets) ** 2)

    # Sum for total loss
    total_loss = jnp.sum(losses)

    return {
        'element_losses': losses,
        'mse': float(mse),
        'manual_mse': float(manual_mse),
        'mse_matches': jnp.allclose(mse, manual_mse)
    }


# =============================================================================
# Example 5: Huber Loss (Robust Regression)
# =============================================================================
def example_huber_loss():
    """
    Huber loss: L2 for small errors, L1 for large errors.
    More robust to outliers than pure L2.

    delta controls the transition point.
    """
    predictions = jnp.array([1.0, 2.0, 3.0, 10.0])  # 10.0 is an outlier
    targets = jnp.array([1.0, 2.0, 3.0, 3.0])

    # Huber loss
    delta = 1.0
    huber_losses = optax.huber_loss(predictions, targets, delta=delta)

    # Compare with L2 loss
    l2_losses = optax.l2_loss(predictions, targets)

    # Huber should be less affected by the outlier
    huber_total = jnp.sum(huber_losses)
    l2_total = jnp.sum(l2_losses)

    return {
        'huber_losses': huber_losses,
        'l2_losses': l2_losses,
        'huber_total': float(huber_total),
        'l2_total': float(l2_total),
        'huber_more_robust': huber_total < l2_total
    }


# =============================================================================
# Example 6: Cosine Similarity Loss
# =============================================================================
def example_cosine_similarity():
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

    # Cosine similarity (1 = same, 0 = orthogonal, -1 = opposite)
    similarities = optax.cosine_similarity(pred, target)

    # Cosine distance (1 - similarity)
    distances = optax.cosine_distance(pred, target)

    return {
        'similarities': similarities,
        'distances': distances,
        'same_vectors': float(similarities[0]),  # Should be 1
        'orthogonal': float(similarities[1]),    # Should be 0.5 (45 degrees)
    }


# =============================================================================
# Example 7: Custom Loss Functions
# =============================================================================
def example_custom_loss():
    """
    Build custom loss functions using JAX primitives.
    """
    def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
        """
        Focal loss for class imbalance.
        Reduces loss for well-classified examples.

        FL = -alpha * (1 - p)^gamma * log(p)  for positive class
        """
        probs = jax.nn.softmax(logits)
        # Get probability of true class
        num_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(labels, num_classes)
        pt = jnp.sum(probs * one_hot, axis=-1)

        # Focal weight
        focal_weight = alpha * (1 - pt) ** gamma

        # Cross-entropy
        ce = -jnp.log(pt + 1e-8)

        return focal_weight * ce

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
        'focal_reduces_easy': focal_losses[0] < ce_losses[0]
    }


# =============================================================================
# Example 8: Label Smoothing
# =============================================================================
def example_label_smoothing():
    """
    Label smoothing: regularization technique.
    Converts hard labels (0/1) to soft labels (epsilon/(K-1), 1-epsilon).
    """
    def smooth_labels(labels, num_classes, smoothing=0.1):
        """Convert hard labels to smoothed soft labels."""
        one_hot = jax.nn.one_hot(labels, num_classes)
        return one_hot * (1 - smoothing) + smoothing / num_classes

    def cross_entropy_with_smoothing(logits, labels, smoothing=0.1):
        """Cross-entropy with label smoothing."""
        num_classes = logits.shape[-1]
        soft_labels = smooth_labels(labels, num_classes, smoothing)
        return optax.softmax_cross_entropy(logits, soft_labels)

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
        'smooth_loss': float(jnp.mean(smooth_loss)),
        'smoothing_increases_loss': float(jnp.mean(smooth_loss)) > float(jnp.mean(hard_loss))
    }


# =============================================================================
# Example 9: Combining Multiple Losses
# =============================================================================
def example_combined_loss():
    """
    Combine multiple loss terms (e.g., task loss + regularization).
    """
    def combined_loss(params, x, y, l2_weight=0.01):
        """
        Total loss = classification loss + L2 regularization.
        """
        # Forward pass (simplified)
        logits = params['w'] @ x.T + params['b'][:, None]
        logits = logits.T

        # Classification loss
        cls_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, y)
        )

        # L2 regularization
        l2_loss = sum(
            jnp.sum(p ** 2) for p in jax.tree.leaves(params)
        )

        return cls_loss + l2_weight * l2_loss, {'cls': cls_loss, 'l2': l2_loss}

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
    grads = grad_fn(params)

    return {
        'total_loss': float(total_loss),
        'classification_loss': float(loss_components['cls']),
        'regularization_loss': float(loss_components['l2']),
        'has_gradients': 'w' in grads and 'b' in grads
    }


# =============================================================================
# Example 10: Loss Functions for Different Tasks
# =============================================================================
def example_task_specific_losses():
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
    binary_loss = optax.sigmoid_binary_cross_entropy(binary_logits, binary_labels)

    # Multi-class
    mc_logits = jnp.array([[2.0, 0.5, 0.1]])
    mc_labels = jnp.array([0])
    mc_loss = optax.softmax_cross_entropy_with_integer_labels(mc_logits, mc_labels)

    # Regression
    pred = jnp.array([2.5])
    target = jnp.array([3.0])
    reg_loss = optax.l2_loss(pred, target)

    return {
        'tasks': list(tasks.keys()),
        'binary_loss': float(binary_loss[0, 0]),
        'multiclass_loss': float(mc_loss[0]),
        'regression_loss': float(reg_loss[0])
    }


# =============================================================================
# Run all examples
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Optax Losses Examples")
    print("=" * 60)

    examples = [
        ("1. Softmax Cross-Entropy", example_softmax_cross_entropy),
        ("2. CE with Integer Labels", example_softmax_cross_entropy_int_labels),
        ("3. Sigmoid Binary CE", example_sigmoid_binary_cross_entropy),
        ("4. L2 Loss", example_l2_loss),
        ("5. Huber Loss", example_huber_loss),
        ("6. Cosine Similarity", example_cosine_similarity),
        ("7. Custom Loss", example_custom_loss),
        ("8. Label Smoothing", example_label_smoothing),
        ("9. Combined Loss", example_combined_loss),
        ("10. Task-Specific Losses", example_task_specific_losses),
    ]

    for name, func in examples:
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
