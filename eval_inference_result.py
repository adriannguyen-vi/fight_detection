import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score

def analyze_and_visualize(csv_path):
    print(f"Loading data from {csv_path}...")
    
    # 1. Load the data
    df = pd.read_csv(csv_path)
    
    y_true = df['True Label']
    y_pred = df['Predicted Label']
    
    # 2. Calculate Metrics
    # Confusion Matrix: [TN, FP]
    #                   [FN, TP]
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle cases where the dataset might only have one class (like your snippet) gracefully
    if cm.shape == (1, 1):
        # If the snippet only contains 0s, expand it for visualization
        tn = cm[0,0] if y_true.iloc[0] == 0 else 0
        tp = cm[0,0] if y_true.iloc[0] == 1 else 0
        fp = fn = 0
        cm = [[tn, fp], [fn, tp]]
    else:
        tn, fp, fn, tp = cm.ravel()
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n--- Numerical Summary ---")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # 3. Create Visualizations
    # Set up a figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Fight Detection Model Evaluation', fontsize=16, fontweight='bold')
    
    # --- Plot A: Confusion Matrix Heatmap ---
    labels = ['Non-Violence (0)', 'Violence (1)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, 
                annot_kws={"size": 14}, cbar=False, ax=axes[0])
    
    axes[0].set_title('Confusion Matrix', fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Add textual labels inside the heatmap for clarity
    axes[0].text(0.5, 0.2, f'True Negative\n({tn})', ha='center', va='center', color='black' if tn < (cm.max()/2) else 'white')
    axes[0].text(1.5, 0.2, f'False Positive\n({fp})', ha='center', va='center', color='black' if fp < (cm.max()/2) else 'white')
    axes[0].text(0.5, 1.2, f'False Negative\n({fn})', ha='center', va='center', color='black' if fn < (cm.max()/2) else 'white')
    axes[0].text(1.5, 1.2, f'True Positive\n({tp})', ha='center', va='center', color='black' if tp < (cm.max()/2) else 'white')

    # --- Plot B: Metrics Bar Chart ---
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [accuracy, precision, recall, f1]
    
    bars = axes[1].bar(metrics_names, metrics_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    axes[1].set_title('Performance Metrics', fontsize=14)
    axes[1].set_ylim(0, 1.1) # Set Y-axis from 0 to 1.1 for percentage scale
    axes[1].set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add the value text on top of each bar
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # Adjust title spacing
    
    # Save and show the plot
    output_img = 'evaluation_metrics.png'
    plt.savefig(output_img, dpi=300)
    print(f"\n✅ Visualization saved successfully as '{output_img}'")
    
    # If running in a script, you can show the plot interactively
    # plt.show()

if __name__ == "__main__":
    # Change this to the path of your actual CSV file
    CSV_FILE_PATH = "/home/adrian/fight_detection/runs/fight_detection_ViolenceX3D_20260306-100023/inference_results.csv" 

            
    analyze_and_visualize(CSV_FILE_PATH)