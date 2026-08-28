"""
Evaluation metrics and comparison module for cold-start friend recommendation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import confusion_matrix, classification_report


def calculate_precision_at_k(ranked_df, user_col='user_id', relevant_col='relevant', k=10):
    """
    Calculate Precision@K.
    
    Precision@K = (relevant recommendations in top K) / K
    """
    precisions = []
    
    for user_id in ranked_df[user_col].unique():
        user_data = ranked_df[ranked_df[user_col] == user_id].head(k)
        precision = user_data[relevant_col].sum() / k if k > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions)


def calculate_recall_at_k(ranked_df, all_df, user_col='user_id', relevant_col='relevant', k=10):
    """
    Calculate Recall@K.
    
    Recall@K = (relevant recommendations in top K) / (total relevant recommendations)
    """
    recalls = []
    
    for user_id in ranked_df[user_col].unique():
        user_data = ranked_df[ranked_df[user_col] == user_id].head(k)
        all_user_data = all_df[all_df[user_col] == user_id]
        
        n_relevant_in_top_k = user_data[relevant_col].sum()
        n_total_relevant = all_user_data[relevant_col].sum()
        
        recall = n_relevant_in_top_k / n_total_relevant if n_total_relevant > 0 else 0
        recalls.append(recall)
    
    return np.mean(recalls)


def calculate_hit_rate_at_k(ranked_df, user_col='user_id', relevant_col='relevant', k=10):
    """
    Calculate Hit Rate@K.
    
    Hit Rate@K = 1 if at least one relevant recommendation in top K, else 0
    """
    hit_rates = []
    
    for user_id in ranked_df[user_col].unique():
        user_data = ranked_df[ranked_df[user_col] == user_id].head(k)
        hit = 1 if user_data[relevant_col].sum() > 0 else 0
        hit_rates.append(hit)
    
    return np.mean(hit_rates)


def evaluate_model(ranked_df, all_df, user_col='user_id', relevant_col='relevant', 
                   k_values=[10, 20, 50]):
    """
    Evaluate model with multiple metrics at different K values.
    """
    results = {}
    
    for k in k_values:
        results[k] = {
            'precision': calculate_precision_at_k(ranked_df, user_col, relevant_col, k),
            'recall': calculate_recall_at_k(ranked_df, all_df, user_col, relevant_col, k),
            'hit_rate': calculate_hit_rate_at_k(ranked_df, user_col, relevant_col, k)
        }
    
    return results


def compare_models(baseline_results, proposed_results, k_values=[10, 20, 50]):
    """
    Compare baseline and proposed model results.
    """
    comparison = []
    
    for k in k_values:
        for metric in ['precision', 'recall', 'hit_rate']:
            baseline_val = baseline_results[k][metric]
            proposed_val = proposed_results[k][metric]
            
            comparison.append({
                'K': k,
                'Metric': metric.replace('_', ' ').title(),
                'Baseline': round(baseline_val, 4),
                'Proposed': round(proposed_val, 4),
                'Improvement': round(proposed_val - baseline_val, 4),
                'Percent_Improvement': round(((proposed_val - baseline_val) / baseline_val * 100), 2) if baseline_val > 0 else 0
            })
    
    return pd.DataFrame(comparison)


def plot_comparison(baseline_results, proposed_results, k_values=[10, 20, 50], 
                    save_dir='reports/figures'):
    """
    Plot comparison between baseline and proposed methods.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = ['precision', 'recall', 'hit_rate']
    metric_names = ['Precision', 'Recall', 'Hit Rate']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        baseline_values = [baseline_results[k][metric] for k in k_values]
        proposed_values = [proposed_results[k][metric] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        axes[i].bar(x - width/2, baseline_values, width, label='Baseline', color='#FF6B6B', alpha=0.8)
        axes[i].bar(x + width/2, proposed_values, width, label='Proposed', color='#4ECDC4', alpha=0.8)
        
        axes[i].set_xlabel('K')
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{metric_name}@K Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(k_values)
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_dir}/comparison_plot.png")


def plot_score_distribution(scored_df, save_dir='reports/figures'):
    """
    Plot distribution of relationship scores.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of scores
    axes[0].hist(scored_df['relationship_score'], bins=20, edgecolor='black', alpha=0.7, color='#4ECDC4')
    axes[0].set_xlabel('Relationship Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Relationship Scores')
    axes[0].axvline(x=0.25, color='red', linestyle='--', label='Weak/Moderate')
    axes[0].axvline(x=0.50, color='orange', linestyle='--', label='Moderate/Strong')
    axes[0].axvline(x=0.75, color='green', linestyle='--', label='Strong/Very Strong')
    axes[0].legend()
    
    # Score by relevance
    if 'relevant' in scored_df.columns:
        relevant_scores = scored_df[scored_df['relevant'] == 1]['relationship_score']
        non_relevant_scores = scored_df[scored_df['relevant'] == 0]['relationship_score']
        
        axes[1].boxplot([non_relevant_scores, relevant_scores], labels=['Non-Relevant', 'Relevant'])
        axes[1].set_ylabel('Relationship Score')
        axes[1].set_title('Relationship Score by Relevance')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_dir}/score_distribution.png")


def plot_interaction_distribution(feature_df, save_dir='reports/figures'):
    """
    Plot distribution of interaction counts.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of interaction counts
    axes[0].hist(feature_df['n_interactions'], bins=20, edgecolor='black', alpha=0.7, color='#3498DB')
    axes[0].set_xlabel('Number of Interactions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Interaction Counts')
    
    # Relevance distribution
    if 'relevant' in feature_df.columns:
        relevance_counts = feature_df['relevant'].value_counts()
        axes[1].bar(['Non-Relevant', 'Relevant'], relevance_counts.values, color=['#FF6B6B', '#4ECDC4'])
        axes[1].set_ylabel('Count')
        axes[1].set_title('Relevance Label Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'interaction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_dir}/interaction_distribution.png")


def plot_confusion_matrix(y_true, y_pred, title, save_dir='reports/figures', filename='confusion_matrix.png'):
    """
    Plot confusion matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Relevant', 'Relevant'],
                yticklabels=['Non-Relevant', 'Relevant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_dir}/{filename}")


def save_metrics_to_json(metrics_dict, save_path='reports/metrics.json'):
    """
    Save metrics to JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to {save_path}")


def create_error_analysis(ranked_df, all_df, user_col='user_id', relevant_col='relevant', 
                          save_path='reports/error_analysis.csv'):
    """
    Create error analysis CSV.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Find misclassified cases (relevant contacts ranked low)
    error_rows = []
    
    for user_id in ranked_df[user_col].unique():
        user_data = ranked_df[ranked_df[user_col] == user_id]
        
        # Get relevant contacts that are ranked low
        relevant_contacts = user_data[user_data[relevant_col] == 1]
        non_relevant_contacts = user_data[user_data[relevant_col] == 0]
        
        # Add relevant contacts that might be ranked below non-relevant ones
        for _, row in relevant_contacts.iterrows():
            error_rows.append({
                'user_id': row[user_col],
                'contact_id': row['contact_id'],
                'n_interactions': row['n_interactions'],
                'relationship_score': row.get('relationship_score', 'N/A'),
                'true_label': row[relevant_col],
                'predicted_label': 1 if row.get('relationship_score', 0) >= 0.5 else 0,
                'error_reason': 'Relevant contact ranked lower than expected',
                'possible_fix': 'Adjust weight parameters or threshold'
            })
    
    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(save_path, index=False)
    
    print(f"Error analysis saved to {save_path}")
    print(f"Total error cases: {len(error_df)}")
    
    return error_df


if __name__ == "__main__":
    # Example usage
    pass