# Lecture Notes: Supervised Learning

---

## I. Supervised Learning Overview

Supervised learning involves learning a function that maps inputs (features) to outputs (labels) using labeled data. It includes two main types:

- **Classification:** Output is a discrete label (e.g., spam/not spam).
- **Regression:** Output is a continuous value (e.g., predicting house price).

---

## II. Decision Trees

Decision Trees are a white-box model used for both classification and regression. They work by recursively partitioning the input space based on feature values, forming a tree-like structure where internal nodes are decision points and leaves are outputs.

### A. Key Concepts

- **Representation vs Algorithm:** The decision tree is a *representation*. Algorithms like ID3, C4.5, or CART are used to learn the tree.
- **Attribute Splitting:**
  - Best attribute chosen based on *information gain* or *Gini impurity*.
  - **Continuous Attributes:** Can be reused along a path by splitting at different thresholds.
  - **Discrete Attributes:** Generally not reused in the same path.
- **Overfitting Risk:**
  - Perfect classification on training data can cause overfitting.
  - Noisy labels can cause infinite loops if not handled properly.

### B. Expressiveness & Complexity

- **Expressiveness:**
  - Rows in a truth table: `2^n`
  - Total boolean functions: `2^(2^n)`
- **Concept Learning Difficulty:**
  - `n-OR` is easy (linear-sized tree).
  - `n-XOR` is hard (exponential-sized tree).
- **Prediction Complexity:** `O(log n)` if tree is balanced.
- **Training Complexity:** `O(n * m * log n)` (n = instances, m = features)

### C. When to Use Decision Trees

**Good for:**
- Small to medium datasets with discrete or mixed-type features.
- Data with missing values or label noise.
- Interpretability is important.

**Less ideal for:**
- High-dimensional data (prone to overfitting).
- Tasks requiring learning of complex functions (e.g., XOR/parity).

---

## III. Decision Tree Algorithms

### A. Generic Decision Tree Learning

1. Pick the attribute that best splits the data.
2. Ask the corresponding question at a node.
3. Create branches for answers.
4. Repeat until stopping conditions are met.

> **Stopping Conditions:**
> - All examples at node have the same label.
> - No remaining attributes.
> - No information gain.

### B. ID3 Algorithm

- **Goal:** Choose attributes that reduce entropy the most.
- **Information Gain:** `Gain(S, A) = Entropy(S) - ∑ (|Sv| / |S|) * Entropy(Sv)`

- **Steps:**
1. Select attribute with highest information gain.
2. Create decision node and split data.
3. Recurse on subsets until pure or no attributes remain.

> **Inductive Bias:** Prefer short, correct trees with good splits at the top.

> **Overfitting Mitigation:**
> - Pruning
> - Cross-validation
> - Breadth-first expansion
> - Stop based on gain thresholds or depth

---

## IV. Decision Tree for Regression

- **Difference:** Instead of classifying, output is a continuous value.
- **Splitting Criteria:** Variance reduction instead of entropy.
- **Output at Leaf:**
- Mean of target values
- Local linear regression (advanced)
- Majority vote (not typical for regression)

> **TODO:** Review exact mechanics of splitting and prediction in regression trees.

---

## V. Classification & Regression Concepts

These apply across all supervised learning models, including trees, SVMs, neural networks, etc.

### A. Key Terms

- **Bias:** Error from incorrect assumptions. High bias = underfitting.
- **Variance:** Error from sensitivity to training data. High variance = overfitting.
- **Bias-Variance Tradeoff:** Balance required for generalization.
- **No Free Lunch Theorem:** No single model performs best across all tasks.
- **Curse of Dimensionality:** Number of feature combinations grows exponentially with dimensionality.

### B. Learning Curves

- **Training Curve:** Accuracy on training set vs data size.
- **Validation Curve:** Accuracy on validation set vs model complexity.
- Used to diagnose overfitting/underfitting.

---

## VI. Vocabulary Summary

### A. Decision Tree Terminology

- **Instance:** One input example.
- **Feature/Attribute:** A variable used for splitting.
- **Concept:** A function that maps inputs to outputs.
- **Target Concept:** The ground truth mapping we want to learn.
- **Hypothesis Class (H):** All functions the learner is willing to consider.
- **Sample (Training Set):** Labeled examples used to train.
- **Candidate Hypothesis:** A specific function under consideration.
- **Testing Set:** Unlabeled examples for evaluation.
- **Entropy Formula:** `H(S) = -∑ p_i * log2(p_i)`
- **Inductive Bias:** Assumptions that guide learning from limited data.
- **Restriction Bias:** Limits on the form of hypotheses in H.
- **Preference Bias:** Ranking of hypotheses within H.

---

## VII. Decision Tree: Pros & Cons

### A. Advantages

- Easy to understand and interpret.
- No need for normalization or scaling.
- Handles categorical and numerical data.
- Robust to missing values and noisy labels.
- Can be used for classification, regression, feature selection.

### B. Disadvantages

- Overfitting risk (especially with deep trees).
- High variance: sensitive to small changes in training data.
- Poor performance on complex patterns (e.g., XOR).
- Trees can look very different if data is rotated or re-scaled.
- No backtracking; greedy algorithms may not find global optimum.
- Prone to bias with imbalanced datasets.
- Requires pruning or ensemble methods (e.g., Random Forest) to improve generalization.

---

## VIII. Splitting Criteria: Entropy vs Gini

- **Entropy (Information Gain):**
- Measures disorder/impurity.
- Tends to yield more balanced trees.
- **Gini Impurity:**
- Simpler and faster to compute.
- Default in scikit-learn.

> In practice, they often produce similar results.

---

## IX. Additional Exercises

- Build decision trees for boolean functions:
- AND, OR, XOR, n-OR (ANY), n-XOR (parity)
- Compare entropy:
- Fair coin vs loaded coin

---