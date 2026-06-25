import numpy as np
import tensorflow as tf
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import re
import random
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split  # 新增导入


# --- 1. 数据准备和编码 ---

def one_hot_encode(seq):
    seq = seq.upper()
    data = np.zeros((len(seq), 4), dtype=np.uint8)
    for i in range(len(seq)):
        if seq[i] == 'A':
            data[i] = [1, 0, 0, 0]
        elif seq[i] == 'C':
            data[i] = [0, 1, 0, 0]
        elif seq[i] == 'G':
            data[i] = [0, 0, 1, 0]
        elif seq[i] == 'T':
            data[i] = [0, 0, 0, 1]
        else:
            data[i] = [0, 0, 0, 0]
    return data


def generate_samples_from_records(records, window_size=401):
    sequences, labels = [], []
    half_window = window_size // 2
    skip_count = 0  # 记录跳过的序列数

    for record in records:
        seq = str(record.seq).upper()
        seq_len = len(seq)

        # 从描述中提取所有数字 [a s t fragment_type ...]
        coords = re.findall(r'\d+', record.description)

        if len(coords) < 2:  # 至少需要s位置
            continue

        # --- 关键修改：提取s位置并检查是否为ATG ---
        # 格式可能是 [a s t] 或 [s t]
        if len(coords) >= 3:
            # 新格式：[a s t]
            s_pos_1based = int(coords[1])  # s位置 (1-based)
            a = int(coords[0]) if len(coords) > 0 else -1  # 阅读框
        else:
            # 旧格式：[s t]
            s_pos_1based = int(coords[0])
            a = (s_pos_1based - 1) % 3  # 计算阅读框

        # 转换为0-based索引
        s_pos_0based = s_pos_1based - 1

        # 检查1: s位置是否在序列范围内
        if s_pos_0based < 0 or s_pos_0based >= seq_len:
            skip_count += 1
            continue

        # 检查2: s位置是否为ATG (关键判断！)
        if seq[s_pos_0based:s_pos_0based + 3] != 'ATG':
            # 不是ATG，跳过这条序列（不提取任何样本）
            skip_count += 1
            continue

        # --- 到这里说明确实有TIS ---
        # 提取正样本（以s位置为中心）
        start = s_pos_0based - half_window
        end = s_pos_0based + half_window + 1
        subseq = ""

        if start < 0:
            subseq += 'N' * abs(start)
            start = 0
        subseq += seq[start:end]
        if end > seq_len:
            subseq += 'N' * (end - seq_len)

        if len(subseq) != window_size:
            continue

        sequences.append(one_hot_encode(subseq))
        labels.append(1)  # 正样本

        # --- 提取负样本（同原逻辑） ---
        # 找到序列中所有ATG位置
        atg_indices = [m.start() for m in re.finditer('ATG', seq)]

        # 负样本候选：排除真实TIS位置
        negative_candidates = [idx for idx in atg_indices if idx != s_pos_0based]

        neg_pos = -1
        if negative_candidates:
            # 策略1: 优先选择相同阅读框的ATG
            same_frame_candidates = [idx for idx in negative_candidates
                                     if idx % 3 == (s_pos_0based % 3)]

            if same_frame_candidates:
                neg_pos = random.choice(same_frame_candidates)
            else:
                # 策略2: 选择任何其他ATG
                neg_pos = random.choice(negative_candidates)

        if neg_pos != -1:
            # 提取负样本窗口
            neg_start = neg_pos - half_window
            neg_end = neg_pos + half_window + 1
            neg_subseq = ""

            if neg_start < 0:
                neg_subseq += 'N' * abs(neg_start)
                neg_start = 0
            neg_subseq += seq[neg_start:neg_end]
            if neg_end > seq_len:
                neg_subseq += 'N' * (neg_end - seq_len)

            if len(neg_subseq) == window_size:
                sequences.append(one_hot_encode(neg_subseq))
                labels.append(0)  # 负样本

    # 输出统计信息
    if skip_count > 0:
        print(f"提示: 跳过了 {skip_count} 条序列（s位置不是ATG或无TIS）")
        print(f"有效序列: {len(records) - skip_count}/{len(records)}")

    if not sequences:
        return np.array([]), np.array([])

    # 打乱数据
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    return np.array(sequences)[indices], np.array(labels)[indices]


# --- 2. 模型构建 ---

def build_neurotis_plus_model(input_shape):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.01)
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 4), padding="valid", activation='relu',
                               kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 1), padding="valid", activation='relu',
                               kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), padding="valid", activation='relu',
                               kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=200, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=2, activation='softmax', kernel_initializer=initializer)
    ])
    return model


# --- 3. 预测测试集中所有ATG位点 ---

def predict_all_atgs_in_testset(model, test_records, window_size=401, batch_size=128):
    """
    批处理版本：完全保留原有代码的logit处理逻辑
    修正列顺序：预测信息在前，真实信息在后
    """
    print("\n--- 开始预测测试集中所有ATG位点 ---")

    half_window = window_size // 2

    # === 完全按照原代码方式创建logit模型 ===
    print("创建logit模型...")

    # 创建logit模型（与原代码完全一致的结构）
    logit_model = tf.keras.Sequential([
        tf.keras.layers.Reshape((window_size, 4, 1), input_shape=(window_size, 4)),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 4), padding="valid", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 1), padding="valid", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(5, 1), padding="valid", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=3),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=200, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=2, activation='linear')  # 注意这里是linear
    ])

    # 复制所有层的权重（除了最后一层）
    for i in range(len(model.layers) - 1):
        try:
            logit_model.layers[i].set_weights(model.layers[i].get_weights())
        except Exception as e:
            print(f"  警告: 第{i}层权重复制失败: {e}")
            continue

    try:
        logit_model.layers[-1].set_weights(model.layers[-1].get_weights())
    except Exception as e:
        print(f"  警告: 最后一层权重复制失败: {e}")

    print("\n收集所有ATG位点...")
    all_sequences = []
    all_metadata = []

    total_atg_count = 0

    for seq_idx, record in enumerate(test_records):
        seq = str(record.seq).upper()
        seq_id = record.id

        # 获取真实TIS信息
        coords = re.findall(r'\d+', record.description)
        true_tis_pos = -1
        true_reading_frame = -1
        tis_position_valid = False

        if len(coords) >= 3:
            potential_tis_pos = int(coords[1]) - 1  # 潜在TIS位置
            true_reading_frame = int(coords[0])
            # 检查该位置是否为ATG
            if potential_tis_pos >= 0 and potential_tis_pos + 2 < len(seq) and seq[
                potential_tis_pos:potential_tis_pos + 3] == 'ATG':
                true_tis_pos = potential_tis_pos
                tis_position_valid = True
        elif len(coords) >= 2:
            potential_tis_pos = int(coords[0]) - 1
            # 检查该位置是否为ATG
            if potential_tis_pos >= 0 and potential_tis_pos + 2 < len(seq) and seq[
                potential_tis_pos:potential_tis_pos + 3] == 'ATG':
                true_tis_pos = potential_tis_pos
                tis_position_valid = True
                if true_tis_pos >= 0:
                    true_reading_frame = true_tis_pos % 3

        # 找到所有ATG
        atg_indices = [m.start() for m in re.finditer('ATG', seq)]

        for atg_idx, atg_pos in enumerate(atg_indices):
            # 提取窗口
            start = atg_pos - half_window
            end = atg_pos + half_window + 1
            subseq = ""

            if start < 0:
                subseq += 'N' * abs(start)
                start = 0
            subseq += seq[start:end]
            if end > len(seq):
                subseq += 'N' * (end - len(seq))

            # 编码
            encoded_seq = one_hot_encode(subseq)
            all_sequences.append(encoded_seq)

            # 保存元数据
            all_metadata.append({
                'sequence_index': seq_idx,
                'sequence_id': seq_id,
                'atg_index': atg_idx,
                'atg_position': atg_pos + 1,
                'reading_frame': atg_pos % 3,
                # 预测信息（先留空，后面填充）
                'tis_probability': None,
                'logit_class_0': None,
                'logit_class_1': None,
                # 真实信息
                'is_true_tis': 1 if atg_pos == true_tis_pos else 0,
                'true_tis_position': true_tis_pos + 1 if true_tis_pos != -1 else -1,
                'true_reading_frame': true_reading_frame
            })

            total_atg_count += 1

        if (seq_idx + 1) % 100 == 0:
            print(f"  已处理 {seq_idx + 1}/{len(test_records)} 条序列，发现 {total_atg_count} 个ATG位点...")

    print(f"\n共发现 {len(all_sequences)} 个ATG位点，开始批处理预测...")

    # 转换为numpy数组
    X = np.array(all_sequences)

    start_time = time.time()

    probabilities = model.predict(X, batch_size=batch_size, verbose=1)

    logits = logit_model.predict(X, batch_size=batch_size, verbose=1)

    prediction_time = time.time() - start_time
    print(f"预测完成！耗时: {prediction_time:.2f} 秒")

    results = []
    for i, metadata in enumerate(all_metadata):
        # 更新预测信息
        metadata['tis_probability'] = float(probabilities[i, 1])
        metadata['logit_class_0'] = float(logits[i, 0])
        metadata['logit_class_1'] = float(logits[i, 1])
        results.append(metadata)

    # 创建DataFrame
    df_results = pd.DataFrame(results)

    # === 确保列的顺序正确 ===
    # 定义希望的列顺序
    desired_columns = [
        'sequence_index',
        'sequence_id',
        'atg_index',
        'atg_position',
        'reading_frame',
        'tis_probability',
        'logit_class_0',
        'logit_class_1',
        'is_true_tis',
        'true_tis_position',
        'true_reading_frame'
    ]

    # 重新排列列
    df_results = df_results[desired_columns]

    return df_results

# --- 4. 训练函数 ---

def train_with_separate_files(train_fasta, test_fasta, window_size=401, model_save_path='tis_predictor_model.h5'):
    """
    使用两个独立的FASTA文件分别进行训练和测试

    Args:
        train_fasta (str): 训练数据FASTA文件路径
        test_fasta (str): 测试数据FASTA文件路径
        window_size (int): 窗口大小
        model_save_path (str): 模型保存路径
    """
    start_time = time.time()

    print(f"正在从 '{train_fasta}' 读取训练转录本记录...")
    if not os.path.exists(train_fasta):
        print(f"错误: 训练FASTA文件 '{train_fasta}' 不存在。")
        return

    print(f"正在从 '{test_fasta}' 读取测试转录本记录...")
    if not os.path.exists(test_fasta):
        print(f"错误: 测试FASTA文件 '{test_fasta}' 不存在。")
        return

    # 读取训练和测试数据
    train_records = list(SeqIO.parse(train_fasta, "fasta"))
    test_records = list(SeqIO.parse(test_fasta, "fasta"))

    # 打乱训练数据
    random.shuffle(train_records)

    print(f"训练转录本: {len(train_records)} 条")
    print(f"测试转录本: {len(test_records)} 条")

    print("正在生成训练样本(采用'诱饵ATG'策略)...")
    X_train, y_train = generate_samples_from_records(train_records, window_size)
    print("正在生成测试样本(采用'诱饵ATG'策略)...")
    X_test, y_test = generate_samples_from_records(test_records, window_size)

    if len(X_train) == 0:
        print("错误: 训练数据集为空，无法训练。")
        return
    if len(X_test) == 0:
        print("错误: 测试数据集为空。")
        return

    print(f"样本生成完成！")
    print(f"训练集总样本数: {len(X_train)} 条 (正样本: {np.sum(y_train == 1)}, 负样本: {np.sum(y_train == 0)})")
    print(f"测试集总样本数: {len(X_test)} 条 (正样本: {np.sum(y_test == 1)}, 负样本: {np.sum(y_test == 0)})")

    # ========== 关键修改：将训练集划分为训练集和验证集 ==========
    # 从训练集中分出20%作为验证集
    print("\n" + "=" * 60)
    print("划分训练集和验证集...")
    print("=" * 60)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train,
        test_size=0.2,  # 20%作为验证集
        random_state=42,  # 固定随机种子以确保可重复性
        stratify=y_train  # 保持正负样本比例
    )

    print(f"划分完成:")
    print(
        f"  训练集: {len(X_train_split)} 条样本 (正样本: {np.sum(y_train_split == 1)}, 负样本: {np.sum(y_train_split == 0)})")
    print(
        f"  验证集: {len(X_val_split)} 条样本 (正样本: {np.sum(y_val_split == 1)}, 负样本: {np.sum(y_val_split == 0)})")
    print(f"  测试集: {len(X_test)} 条样本 (正样本: {np.sum(y_test == 1)}, 负样本: {np.sum(y_test == 0)})")

    # 构建模型
    input_shape = (window_size, 4)
    model = build_neurotis_plus_model(input_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # 简化：只使用准确率指标
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n" + "=" * 60)
    print("模型结构摘要:")
    print("=" * 60)
    model.summary()
    print("=" * 60)

    print("\n开始训练模型...")

    # 设置回调函数
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # 监控验证集准确率
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # 学习率调度器
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 监控验证集损失
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # 模型检查点
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',  # 根据验证集准确率保存最佳模型
        save_best_only=True,
        verbose=1
    )

    # 训练模型（使用验证集）
    print(f"\n开始训练，共 {len(X_train_split)} 个训练样本，{len(X_val_split)} 个验证样本，{len(X_test)} 个测试样本")
    print(f"批次大小: 64, 初始学习率: 0.0005")

    history = model.fit(
        X_train_split, y_train_split,
        epochs=30,
        batch_size=64,
        validation_data=(X_val_split, y_val_split),  # 使用验证集
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    # 加载最佳模型
    if os.path.exists('best_model.h5'):
        print("加载最佳模型权重...")
        model.load_weights('best_model.h5')

    # 评估模型 - 在测试集上评估
    print("\n" + "=" * 60)
    print("在测试集上评估模型:")
    print("=" * 60)

    # 预测测试集
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 计算各项指标
    accuracy = np.mean(y_pred == y_test)

    # 计算精确率、召回率和F1分数
    true_pos = np.sum((y_pred == 1) & (y_test == 1))
    false_pos = np.sum((y_pred == 1) & (y_test == 0))
    false_neg = np.sum((y_pred == 0) & (y_test == 1))
    true_neg = np.sum((y_pred == 0) & (y_test == 0))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集精确率: {precision:.4f}")
    print(f"测试集召回率: {recall:.4f}")
    print(f"测试集F1分数: {f1_score:.4f}")
    print(f"混淆矩阵:")
    print(f"  真正例 (TP): {true_pos}")
    print(f"  假正例 (FP): {false_pos}")
    print(f"  真反例 (TN): {true_neg}")
    print(f"  假反例 (FN): {false_neg}")

    # 保存最终模型（使用新格式避免警告）
    print(f"\n保存最终模型到 '{model_save_path}'...")
    if model_save_path.endswith('.h5'):
        # 使用新格式
        model_save_path_new = model_save_path.replace('.h5', '.keras')
        model.save(model_save_path_new)
        print(f"模型已保存为新格式: {model_save_path_new}")
        print("注意: 为避免警告，已使用.keras格式保存模型")
    else:
        model.save(model_save_path)

    print("模型保存成功。")

    # 预测测试集中所有ATG位点
    print("\n" + "=" * 60)
    print("开始预测测试集中所有ATG位点...")
    print("=" * 60)

    df_predictions = predict_all_atgs_in_testset(model, test_records, window_size)

    # 保存预测结果到CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"test_predictions_{timestamp}.csv"
    df_predictions.to_csv(output_csv, index=False)

    print(f"\n预测结果已保存到: {output_csv}")
    print(f"总记录数: {len(df_predictions)}")
    print(f"预测为TIS的概率分布:")
    print(f"  均值: {df_predictions['tis_probability'].mean():.4f}")
    print(f"  标准差: {df_predictions['tis_probability'].std():.4f}")
    print(f"  最小值: {df_predictions['tis_probability'].min():.4f}")
    print(f"  最大值: {df_predictions['tis_probability'].max():.4f}")

    # 显示真实TIS的预测结果
    if 'is_true_tis' in df_predictions.columns:
        true_tis_predictions = df_predictions[df_predictions['is_true_tis'] == 1]
        if len(true_tis_predictions) > 0:
            print(f"\n真实TIS位点的预测:")
            print(f"  数量: {len(true_tis_predictions)}")
            print(f"  TIS概率均值: {true_tis_predictions['tis_probability'].mean():.4f}")
            print(f"  TIS概率中位数: {true_tis_predictions['tis_probability'].median():.4f}")

            # 统计预测正确的TIS
            correct_tis = true_tis_predictions[true_tis_predictions['tis_probability'] > 0.5]
            incorrect_tis = true_tis_predictions[true_tis_predictions['tis_probability'] <= 0.5]
            print(
                f"  预测正确的TIS数 (概率>0.5): {len(correct_tis)}/{len(true_tis_predictions)} ({len(correct_tis) / len(true_tis_predictions) * 100:.1f}%)")

            if len(correct_tis) > 0:
                print(f"    正确预测的TIS平均概率: {correct_tis['tis_probability'].mean():.4f}")
            if len(incorrect_tis) > 0:
                print(f"    错误预测的TIS平均概率: {incorrect_tis['tis_probability'].mean():.4f}")

    # 按阅读框分组统计
    print(f"\n按阅读框分组的预测结果:")
    for frame in [0, 1, 2]:
        frame_data = df_predictions[df_predictions['reading_frame'] == frame]
        if len(frame_data) > 0:
            tis_count = len(frame_data[frame_data['tis_probability'] > 0.5])
            tis_percentage = tis_count / len(frame_data) * 100 if len(frame_data) > 0 else 0
            print(
                f"  阅读框 {frame}: {len(frame_data)} 个ATG, TIS预测数: {tis_count} ({tis_percentage:.1f}%), TIS概率均值: {frame_data['tis_probability'].mean():.4f}")

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n总训练和预测时间: {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟)")

    return model, df_predictions


if __name__ == "__main__":
    # 配置参数
    train_fasta = "Plants-partial-train.fa"  # 训练文件
    test_fasta = "oeu-3UTR-test.fa"  # 测试文件
    window_size = 401
    model_path = 'tis_predictor_model.keras'  # 使用新格式

    print("=" * 60)
    print("TIS预测模型训练和测试")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(train_fasta):
        print(f"错误: 训练文件 '{train_fasta}' 不存在。")
        print(f"请确保文件在当前目录下:")
        print(f"  训练文件: {train_fasta}")
        print(f"  测试文件: {test_fasta}")
        exit(1)

    if not os.path.exists(test_fasta):
        print(f"错误: 测试文件 '{test_fasta}' 不存在。")
        print(f"请确保文件在当前目录下:")
        print(f"  训练文件: {train_fasta}")
        print(f"  测试文件: {test_fasta}")
        exit(1)

    print(f"找到训练文件: {train_fasta}")
    print(f"找到测试文件: {test_fasta}")

    # 检查文件大小
    train_size = os.path.getsize(train_fasta) / 1024 / 1024  # MB
    test_size = os.path.getsize(test_fasta) / 1024 / 1024  # MB
    print(f"训练文件大小: {train_size:.2f} MB")
    print(f"测试文件大小: {test_size:.2f} MB")

    # 读取序列数量
    train_records = list(SeqIO.parse(train_fasta, "fasta"))
    test_records = list(SeqIO.parse(test_fasta, "fasta"))

    print(f"训练序列数量: {len(train_records)}")
    print(f"测试序列数量: {len(test_records)}")

    if len(train_records) == 0:
        print("错误: 训练文件为空！")
        exit(1)

    if len(test_records) == 0:
        print("错误: 测试文件为空！")
        exit(1)

    # 显示示例序列信息
    print(f"\n训练集第一条序列:")
    print(f"  ID: {train_records[0].id}")
    print(f"  描述: {train_records[0].description}")
    print(f"  长度: {len(train_records[0].seq)} bp")

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练过程...")
    print("=" * 60)

    # 如果模型不存在或强制重新训练
    if not os.path.exists(model_path):
        model, predictions = train_with_separate_files(
            train_fasta=train_fasta,
            test_fasta=test_fasta,
            window_size=window_size,
            model_save_path=model_path
        )
    else:
        print(f"模型文件 '{model_path}' 已存在。")
        print("如果要重新训练，请先删除现有模型文件。")

        # 加载现有模型进行预测
        print("\n加载现有模型进行预测...")
        model = tf.keras.models.load_model(model_path)

        # 预测测试集中所有ATG位点
        test_records = list(SeqIO.parse(test_fasta, "fasta"))
        df_predictions = predict_all_atgs_in_testset(model, test_records, window_size)

        # 保存预测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"test_predictions_{timestamp}.csv"
        df_predictions.to_csv(output_csv, index=False)
        print(f"\n预测结果已保存到: {output_csv}")

    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)