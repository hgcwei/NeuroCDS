# main_test.py (完整修改版 - 支持三个阅读框单独预测)
from uuid import uuid4
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Input, Add, Activation, Dense
from tensorflow.keras import Model
import os


# 保持原有的 compiled_tcn 函数
def compiled_tcn(num_feat=64, num_classes=2, nb_filters=15, kernel_size=20,
                 dilations=[1, 2, 4], nb_stacks=2, max_len=500, use_skip_connections=False,
                 opt='rmsprop', padding='same', lr=1e-3, use_weight_norm=True, return_sequences=True):
    """替代原TCN函数的简单CNN实现"""

    inputs = Input(shape=(max_len, num_feat))
    x = inputs

    for i in range(nb_stacks):
        for dilation in dilations:
            conv_out = Conv1D(nb_filters, kernel_size, dilation_rate=dilation, padding=padding)(x)
            conv_out = BatchNormalization()(conv_out)
            conv_out = Activation('relu')(conv_out)
            conv_out = Dropout(0.1)(conv_out)

            if use_skip_connections:
                if x.shape[-1] != conv_out.shape[-1]:
                    x = Conv1D(nb_filters, 1, padding=padding)(x)
                x = Add()([x, conv_out])
            else:
                x = conv_out

    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    if opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_model_with_logits_output():
    """创建返回logits和softmax概率的模型"""
    inputs = Input(shape=(500, 64))
    x = inputs

    # 构建网络结构
    for i in range(2):  # nb_stacks=2
        for dilation in [1, 2, 4]:
            conv_out = Conv1D(15, 20, dilation_rate=dilation, padding='same')(x)
            conv_out = BatchNormalization()(conv_out)
            conv_out = Activation('relu')(conv_out)
            conv_out = Dropout(0.1)(conv_out)
            x = conv_out

    # 获取softmax前的logits
    logits = Dense(2)(x)  # 注意：这里没有activation，输出两个logits值

    # softmax概率
    probabilities = Activation('softmax')(logits)

    # 创建两个模型
    # 1. 用于训练的模型（输出概率）
    train_model = Model(inputs=inputs, outputs=probabilities)

    # 2. 用于推理的模型（输出logits和概率）
    inference_model = Model(inputs=inputs, outputs=[logits, probabilities])

    return train_model, inference_model


def load_and_prepare_sequence(feature_file, segment_len=500):
    """
    加载单个序列特征并填充到segment_len的倍数
    """
    # 加载特征
    features = np.loadtxt(feature_file, delimiter=',')
    original_length = features.shape[0]

    # 计算需要填充的长度
    remainder = original_length % segment_len
    if remainder > 0:
        padding = segment_len - remainder
        features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        padded_length = original_length + padding
    else:
        padded_length = original_length

    # 重塑为多个段
    n_segments = padded_length // segment_len
    if n_segments > 0:
        features_reshaped = features.reshape((n_segments, segment_len, 64))
    else:
        # 如果序列太短，填充到一个段
        padding = segment_len - original_length
        features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        features_reshaped = features.reshape((1, segment_len, 64))
        n_segments = 1

    print(f"  长度: {original_length} -> {padded_length} ({n_segments}个段)")
    return features_reshaped, original_length, n_segments


def predict_single_frame(model, features_reshaped, original_length, segment_len=500):
    """
    对单个阅读框进行预测，返回logits和softmax概率
    """
    n_segments = features_reshaped.shape[0]
    all_logits = []  # 保存完整的logits矩阵
    all_probs = []  # 保存正类的概率

    for i in range(n_segments):
        segment = features_reshaped[i:i + 1, :, :]  # (1, segment_len, 64)
        logits, probabilities = model.predict(segment, verbose=0)

        # 保存完整的logits
        all_logits.append(logits[0])  # (segment_len, 2)

        # 只保存正类的概率
        pos_probs = probabilities[0, :, 1]  # (segment_len,)
        all_probs.append(pos_probs)

    # 合并所有段
    if n_segments > 1:
        full_logits = np.vstack(all_logits)  # (总长度, 2)
        full_probs = np.concatenate(all_probs)  # (总长度,)
    else:
        full_logits = all_logits[0]
        full_probs = all_probs[0]

    # 只保留原始长度部分
    return full_logits[:original_length], full_probs[:original_length]


def save_frame_predictions(seq_dir, frame_idx, logits, probabilities):
    """
    保存单个阅读框的预测结果到序列文件夹
    """
    # 确保序列文件夹存在
    os.makedirs(seq_dir, exist_ok=True)

    # 1. 保存softmax概率（正类）
    prob_file = os.path.join(seq_dir, f"frame_{frame_idx}_pred.csv")
    np.savetxt(prob_file, probabilities, delimiter=',', fmt='%.6f')

    # 2. 保存logits（两个原始值）
    logits_file = os.path.join(seq_dir, f"frame_{frame_idx}_logits.csv")
    np.savetxt(logits_file, logits, delimiter=',', fmt='%.6f')

    return prob_file, logits_file


def run_task():
    # 1. 创建模型（现在有两个输出：logits和probabilities）
    print("创建模型...")
    train_model, inference_model = create_model_with_logits_output()

    # 2. 编译训练模型（用于加载权重）
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    train_model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. 加载权重到训练模型
    weights_filename = 'plants-partial.tcn.weights.0.8974.weights.h5'
    print(f"加载权重文件: {weights_filename}")

    try:
        # 先构建模型（确保所有层都已创建）
        dummy_input = tf.random.normal([1, 500, 64])
        _ = train_model(dummy_input)
        train_model.load_weights(weights_filename)

        # 将权重转移到推理模型
        for i in range(len(train_model.layers)):
            inference_model.layers[i].set_weights(train_model.layers[i].get_weights())

        print("权重加载成功")
    except Exception as e:
        print(f"权重加载失败: {e}")
        return

    # 4. 设置路径
    feature_dir = "test_data/features"
    sequences_dir = "test_data/predictions"

    if not os.path.exists(feature_dir):
        print(f"错误: 特征目录不存在: {feature_dir}")
        print("请先运行 separate_test_data.py")
        return

    os.makedirs(sequences_dir, exist_ok=True)

    # 5. 逐个序列预测（三个阅读框）
    print("\n开始逐个序列预测（三个阅读框）...")

    processed_count = 0
    total_sequences = 2000

    for seq_id in range(1, total_sequences + 1):
        # 检查是否有这个序列的特征文件（只检查frame_0）
        feature_file_0 = os.path.join(feature_dir, f"{seq_id}_feature_0.csv")

        if not os.path.exists(feature_file_0):
            continue

        print(f"预测序列 {seq_id}/{total_sequences}...")

        # 创建序列文件夹路径
        seq_dir = os.path.join(sequences_dir, f"seq_{seq_id}")

        # 检查是否已经预测过（如果有任意一个文件存在，则跳过）
        pred_file_0 = os.path.join(seq_dir, "frame_0_pred.csv")
        if os.path.exists(pred_file_0):
            print(f"  已存在，跳过")
            processed_count += 1
            continue

        try:
            # 对三个阅读框分别进行预测
            for frame_idx in range(3):
                feature_file = os.path.join(feature_dir, f"{seq_id}_feature_{frame_idx}.csv")

                if not os.path.exists(feature_file):
                    print(f"  警告: 阅读框{frame_idx}特征文件不存在")
                    continue

                print(f"  处理阅读框 {frame_idx}...")

                # 加载并准备序列
                features_reshaped, original_length, n_segments = load_and_prepare_sequence(
                    feature_file, segment_len=500
                )

                # 预测（返回logits和概率）
                logits, probabilities = predict_single_frame(
                    inference_model, features_reshaped, original_length
                )

                # 保存预测结果到序列文件夹
                prob_file, logits_file = save_frame_predictions(
                    seq_dir, frame_idx, logits, probabilities
                )

                print(f"    ✓ 保存: {prob_file}")
                print(f"    ✓ 保存: {logits_file}")

            processed_count += 1
            print(f"  ✓ 序列 {seq_id} 预测完成")

        except Exception as e:
            print(f"  ✗ 序列 {seq_id} 预测失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n预测完成! 成功处理 {processed_count}/{total_sequences} 个序列")

    return processed_count


if __name__ == '__main__':
    processed = run_task()
    print(f"\n总共处理了 {processed} 个序列")