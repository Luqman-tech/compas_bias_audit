import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import warnings
warnings.filterwarnings('ignore')

# Load COMPAS dataset
print("Loading COMPAS dataset...")
dataset = CompasDataset()
print(f"Dataset shape: {dataset.features.shape}")
print(f"Protected attribute: {dataset.protected_attribute_names}")
print(f"Favorable label: {dataset.favorable_label}")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
df['race'] = dataset.protected_attributes[:, 0]  # Race is the protected attribute
df['recid'] = dataset.labels.ravel()  # Recidivism is the target

# Basic dataset statistics
print("\n=== Dataset Overview ===")
print(f"Total samples: {len(df)}")
print(f"Racial distribution:")
race_counts = df['race'].value_counts()
print(race_counts)
print(f"\nRecidivism rate by race:")
recid_by_race = df.groupby('race')['recid'].mean()
print(recid_by_race)

# Define privileged and unprivileged groups
privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

# Calculate bias metrics
print("\n=== Bias Analysis ===")
metric = BinaryLabelDatasetMetric(dataset, 
                                 unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)

print(f"Mean difference: {metric.mean_difference():.4f}")
print(f"Disparate impact: {metric.disparate_impact():.4f}")
print(f"Statistical parity difference: {metric.statistical_parity_difference():.4f}")

# Simulate predictions (using a simple threshold on a feature for demonstration)
# In practice, you'd use actual COMPAS scores or train a model
df['predicted'] = (df['priors_count'] > df['priors_count'].median()).astype(int)

# Create classification metrics
dataset_pred = dataset.copy()
dataset_pred.labels = df['predicted'].values.reshape(-1, 1)

class_metric = ClassificationMetric(dataset, dataset_pred,
                                   unprivileged_groups=unprivileged_groups,
                                   privileged_groups=privileged_groups)

print(f"\n=== Classification Metrics ===")
print(f"Accuracy: {class_metric.accuracy():.4f}")
print(f"Precision: {class_metric.precision():.4f}")
print(f"Recall: {class_metric.recall():.4f}")
print(f"False positive rate difference: {class_metric.false_positive_rate_difference():.4f}")
print(f"False negative rate difference: {class_metric.false_negative_rate_difference():.4f}")
print(f"Equal opportunity difference: {class_metric.equal_opportunity_difference():.4f}")
print(f"Equalized odds difference: {class_metric.equalized_odds_difference():.4f}")

# Visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Recidivism rates by race
race_labels = ['African-American', 'Caucasian']
recid_rates = [recid_by_race[0], recid_by_race[1]]
axes[0, 0].bar(race_labels, recid_rates, color=['#ff7f0e', '#1f77b4'])
axes[0, 0].set_title('Recidivism Rates by Race')
axes[0, 0].set_ylabel('Recidivism Rate')
axes[0, 0].set_ylim(0, 1)
for i, v in enumerate(recid_rates):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# 2. False Positive Rates by Race
fpr_aa = class_metric.false_positive_rate(privileged=False)
fpr_cauc = class_metric.false_positive_rate(privileged=True)
fpr_rates = [fpr_aa, fpr_cauc]
axes[0, 1].bar(race_labels, fpr_rates, color=['#ff7f0e', '#1f77b4'])
axes[0, 1].set_title('False Positive Rates by Race')
axes[0, 1].set_ylabel('False Positive Rate')
axes[0, 1].set_ylim(0, max(fpr_rates) * 1.2)
for i, v in enumerate(fpr_rates):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 3. Confusion Matrix Heatmap for African-American
aa_mask = df['race'] == 0
cm_aa = confusion_matrix(df[aa_mask]['recid'], df[aa_mask]['predicted'])
sns.heatmap(cm_aa, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Recid', 'Recid'], 
            yticklabels=['No Recid', 'Recid'],
            ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - African-American')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Confusion Matrix Heatmap for Caucasian
cauc_mask = df['race'] == 1
cm_cauc = confusion_matrix(df[cauc_mask]['recid'], df[cauc_mask]['predicted'])
sns.heatmap(cm_cauc, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Recid', 'Recid'], 
            yticklabels=['No Recid', 'Recid'],
            ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix - Caucasian')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Additional bias metrics visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
bias_metrics = {
    'Mean Difference': metric.mean_difference(),
    'Disparate Impact': metric.disparate_impact() - 1,  # Subtract 1 to show deviation from fairness
    'Statistical Parity Diff': metric.statistical_parity_difference(),
    'FPR Difference': class_metric.false_positive_rate_difference(),
    'Equal Opportunity Diff': class_metric.equal_opportunity_difference()
}

metrics_names = list(bias_metrics.keys())
metrics_values = list(bias_metrics.values())

colors = ['red' if abs(v) > 0.1 else 'orange' if abs(v) > 0.05 else 'green' for v in metrics_values]
bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High bias threshold')
ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
ax.set_title('Bias Metrics Summary')
ax.set_ylabel('Metric Value')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Demonstrate bias mitigation using Reweighting
print("\n=== Bias Mitigation ===")
print("Applying Reweighting preprocessing...")
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)

# Check metrics after reweighting
metric_transf = BinaryLabelDatasetMetric(dataset_transf,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)

print(f"Mean difference after reweighting: {metric_transf.mean_difference():.4f}")
print(f"Disparate impact after reweighting: {metric_transf.disparate_impact():.4f}")
print(f"Statistical parity difference after reweighting: {metric_transf.statistical_parity_difference():.4f}")

# Summary statistics
print("\n=== Summary ===")
print(f"Original dataset bias (mean difference): {metric.mean_difference():.4f}")
print(f"After reweighting bias (mean difference): {metric_transf.mean_difference():.4f}")
print(f"Bias reduction: {abs(metric.mean_difference()) - abs(metric_transf.mean_difference()):.4f}")

print("\nAnalysis complete. Check the visualizations and report for detailed findings.")