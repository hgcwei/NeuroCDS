# main.py (修正版 - 包含完整的评估函数)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, f1_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, precision_score, accuracy_score
import os


def load_all_predictions_and_labels(sequences_dir, label_dir, total_sequences=2000):
    """
    加载所有序列的预测和标签（三个阅读框合并）
    """
    print("加载预测结果和标签（三个阅读框）...")

    all_scores = []
    all_labels = []
    valid_sequences = 0
    for seq_id in range(1, total_sequences + 1):
        seq_dir = os.path.join(sequences_dir, f"seq_{seq_id}")

        # 检查序列文件夹是否存在
        if not os.path.exists(seq_dir):
            continue

        # 检查是否有预测文件（至少第一个阅读框）
        pred_file_0 = os.path.join(seq_dir, "frame_0_pred.csv")
        if not os.path.exists(pred_file_0):
            continue

        try:
            # 加载三个阅读框的预测结果
            frame_scores = []
            frame_labels = []

            for frame_idx in range(3):
                # 加载预测概率
                pred_file = os.path.join(seq_dir, f"frame_{frame_idx}_pred.csv")
                if not os.path.exists(pred_file):
                    continue

                scores = np.loadtxt(pred_file, delimiter=',')

                # 加载对应标签
                label_file = os.path.join(label_dir, f"{seq_id}_label_{frame_idx}.csv")
                if not os.path.exists(label_file):
                    continue

                labels = np.loadtxt(label_file, delimiter=',')

                # 确保长度匹配
                if len(scores) != len(labels):
                    min_len = min(len(scores), len(labels))
                    scores = scores[:min_len]
                    labels = labels[:min_len]

                frame_scores.append(scores)
                frame_labels.append(labels)

            if not frame_scores:
                continue

            # 合并三个阅读框的数据
            seq_scores = np.concatenate(frame_scores)
            seq_labels = np.concatenate(frame_labels)

            all_scores.append(seq_scores)
            all_labels.append(seq_labels)
            valid_sequences += 1

        except Exception as e:
            print(f"警告: 序列 {seq_id} 加载失败: {e}")
            continue

    if not all_scores:
        print("错误: 未找到有效的预测/标签对")
        return None, None

    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels)

    return flat_scores, flat_labels


# ==================== 从原main.py复制的评估函数 ====================
def eval_perf(label, scores):
    """修复后的性能评估函数"""

    # 如果scores是二维的，提取正类概率
    if len(scores.shape) > 1 and scores.shape[1] == 2:
        scores = scores[:, 1]  # 取第二列作为正类概率

    fpr, tpr, thresholds = roc_curve(label, scores)
    pr, re, thresholds2 = precision_recall_curve(label, scores)

    # 找到阈值接近0.5的点
    threshold_idx = np.argmin(np.abs(thresholds - 0.5))
    sn, sp, auROC, auPRC = tpr[threshold_idx], 1 - fpr[threshold_idx], auc(fpr, tpr), auc(re, pr)

    # 生成预测标签
    predictions = (scores > 0.5).astype(np.int32)

    label = np.int32(np.array(label))
    label = label.tolist()
    pre = precision_score(label, predictions, zero_division=0)
    acc = accuracy_score(label, predictions)
    f1 = f1_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)

    perf_scores = (sn, sp, pre, acc, f1, auROC, auPRC, mcc)
    curve_scores = (fpr, tpr, re, pr)

    return perf_scores, curve_scores


def calculate_performance_metrics(scores, labels):
    """计算详细的性能指标"""
    # 如果scores是二维的，提取正类概率
    if len(scores.shape) > 1 and scores.shape[1] == 2:
        scores = scores[:, 1]

    # 计算各种指标
    fpr, tpr, thresholds = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)

    auROC = auc(fpr, tpr)
    auPRC = auc(recall, precision)

    # 使用0.5作为分类阈值
    predictions = (scores > 0.5).astype(np.int32)

    # 确保labels是整数
    labels_int = labels.astype(np.int32)

    accuracy = accuracy_score(labels_int, predictions)
    precision_val = precision_score(labels_int, predictions, zero_division=0)
    f1 = f1_score(labels_int, predictions)
    mcc = matthews_corrcoef(labels_int, predictions)

    # 计算敏感性和特异性 - 修复这里
    # 确保计算用整数
    tn, fp, fn, tp = np.bincount(labels_int * 2 + predictions, minlength=4)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision_val,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'auROC': auROC,
        'auPRC': auPRC,
        'MCC': mcc
    }


# ==================== 评估函数结束 ====================

def main():
    """主评估函数"""
    print("=== NeuroTIS+ Oeu-3UTR 性能评估 ===\n")

    # 设置路径和参数
    predictions_dir = "test_data/predictions"
    label_dir = "test_data/labels"
    total_sequences = 2000

    if not os.path.exists(predictions_dir):
        print(f"错误: 序列目录不存在: {predictions_dir}")
        print("请先运行 main_test.py 进行预测")
        return

    if not os.path.exists(label_dir):
        print(f"错误: 标签目录不存在: {label_dir}")
        return

    # 加载数据（合并三个阅读框）
    scores, labels = load_all_predictions_and_labels(predictions_dir, label_dir, total_sequences)

    if scores is None or labels is None:
        return

    # 计算性能指标
    print("\n计算性能指标...")
    metrics = calculate_performance_metrics(scores, labels)

    # 打印结果
    print("\n=== NeuroTIS+ Oeu-3UTR 性能指标 ===")
    for metric, value in metrics.items():
        if metric in ['TP', 'FP', 'TN', 'FN']:
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    # 保存性能指标
    metrics_file = 'oeu-3UTR_performance_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("NeuroTIS+ Oeu-3UTR Performance Metrics\n")
        f.write("===================================\n")

        for metric, value in metrics.items():
            if metric in ['TP', 'FP', 'TN', 'FN']:
                f.write(f"{metric}: {value}\n")
            else:
                f.write(f"{metric}: {value:.4f}\n")

    print(f"\n性能指标已保存到: {metrics_file}")

    # 绘制ROC曲线
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - NeuroTIS+ (Oeu-3UTR)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('oeu-3UTR_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("ROC曲线已保存到: oeu-3UTR_roc_curve.png")
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")


if __name__ == "__main__":
    main()