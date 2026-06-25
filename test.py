# check_atg_accuracy_advanced.py
import pandas as pd
import os
import numpy as np  # 引入numpy用于MCC计算


def calculate_atg_accuracy(csv_file_path):
    """
    读取TIS-Rover生成的预测结果CSV文件，并计算包括敏感性、特异性等在内的多种性能指标。

    Args:
        csv_file_path (str): 预测结果CSV文件的路径。
    """
    # --- 1. 检查并加载文件 ---
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 未找到。")
        print("请确保下面的 'file_to_analyze' 变量设置了正确的文件名，并且文件与此脚本位于同一目录下。")
        return

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"错误: 读取CSV文件时出错: {e}")
        return

    # --- 2. 检查必需的列是否存在 ---
    required_columns = ['tis_probability', 'is_true_tis']
    if not all(col in df.columns for col in required_columns):
        print(f"错误: CSV文件缺少必需的列。请确保文件包含 {required_columns}。")
        return

    # --- 3. 计算基础指标 ---
    # 根据概率阈值0.5，生成预测标签 (1 代表预测为TIS, 0 代表预测为非TIS)
    df['predicted_label'] = (df['tis_probability'] >= 0.5).astype(int)

    total_predictions = len(df)
    if total_predictions == 0:
        print("CSV文件中没有数据。")
        return

    correct_predictions = (df['predicted_label'] == df['is_true_tis']).sum()
    accuracy = correct_predictions / total_predictions

    # --- 4. 计算混淆矩阵的四个基本量 ---
    true_positives = ((df['predicted_label'] == 1) & (df['is_true_tis'] == 1)).sum()
    true_negatives = ((df['predicted_label'] == 0) & (df['is_true_tis'] == 0)).sum()
    false_positives = ((df['predicted_label'] == 1) & (df['is_true_tis'] == 0)).sum()
    false_negatives = ((df['predicted_label'] == 0) & (df['is_true_tis'] == 1)).sum()

    # ###############################################################
    # ###               【【【核心修改区域】】】                    ###
    # ###############################################################
    #
    # --- 5. 计算高级性能指标 ---
    # 为防止除零错误，在分母上加一个极小值 epsilon
    epsilon = 1e-9

    # 敏感性 (Sensitivity / Recall / TPR): 真正例中被正确预测的比例 (查全率)
    sensitivity = true_positives / (true_positives + false_negatives + epsilon)

    # 特异性 (Specificity / TNR): 真反例中被正确预测的比例
    specificity = true_negatives / (true_negatives + false_positives + epsilon)

    # 精确率 (Precision / PPV): 预测为正例的样本中，真实为正例的比例 (查准率)
    precision = true_positives / (true_positives + false_positives + epsilon)

    # F1分数 (F1-Score): 精确率和敏感性的调和平均数，一个综合性指标
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + epsilon)

    # 马修斯相关系数 (MCC): 一个均衡的指标，即使在类别不平衡的情况下也表现良好
    mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    mcc_denominator = np.sqrt(
        (true_positives + false_positives) *
        (true_positives + false_negatives) *
        (true_negatives + false_positives) *
        (true_negatives + false_negatives)
    )
    mcc = mcc_numerator / (mcc_denominator + epsilon)

    # --- 6. 打印完整的分析报告 ---
    print("\n" + "=" * 60)
    print("      ATG 预测性能综合分析报告")
    print("=" * 60)
    print(f"分析文件: {os.path.basename(csv_file_path)}")
    print(f"总共评估的ATG位点数: {total_predictions}")
    print("-" * 60)

    print("【基础准确率】")
    print(f"  - 整体预测准确率 (Accuracy): {accuracy:.2%}")
    print(f"    (预测正确的位点数 / 总位点数)")
    print("-" * 60)

    print("【混淆矩阵详解】")
    print(f"  - 真正例 (TP): {true_positives:6d}  (模型正确预测为TIS的位点)")
    print(f"  - 真反例 (TN): {true_negatives:6d}  (模型正确预测为非TIS的位点)")
    print(f"  - 假正例 (FP): {false_positives:6d}  (模型错误预测为TIS的位点) - (误报)")
    print(f"  - 假反例 (FN): {false_negatives:6d}  (模型错误预测为非TIS的位点) - (漏报)")
    print("-" * 60)

    print("【高级性能指标】")
    print(f"  - 敏感性 (Sensitivity/Recall): {sensitivity:.2%}")
    print("    (查全率：在所有真实的TIS中，模型成功预测出了多少)")

    print(f"\n  - 特异性 (Specificity):      {specificity:.2%}")
    print("    (在所有非TIS的ATG中，模型成功排除了多少)")

    print(f"\n  - 精确率 (Precision):         {precision:.2%}")
    print("    (查准率：在所有被模型预测为TIS的位点中，有多少是真的)")

    print(f"\n  - F1 分数 (F1-Score):         {f1_score:.4f}")
    print("    (精确率和敏感性的综合平衡指标，越接近1越好)")

    print(f"\n  - 马修斯相关系数 (MCC):   {mcc:.4f}")
    print("    (综合考虑四类结果的平衡指标, 范围[-1, 1], 越接近1越好)")
    print("=" * 60)


# ==============================================================================
#                                 主执行区域
# ==============================================================================
if __name__ == "__main__":
    # --- 【请在这里修改您要分析的文件名】 ---
    # 您只需要修改下面这一行代码中的 "file.csv" 为您实际的预测结果文件名即可。
    # 例如: file_to_analyze = "test_predictions_20240523_103000.csv"

    file_to_analyze = "test_predictions_20260218_221605.csv"

    # -----------------------------------------

    # 调用主函数，开始分析
    calculate_atg_accuracy(file_to_analyze)

