# separate_data_final.py (适配 >[offset s t] 三维信息行)
import numpy as np
from Bio import SeqIO
import pandas as pd
import os
import sys
import re

# 确保能找到 cuRNA.py
sys.path.append('.')
import cuRNA


# --- 辅助函数 (无需修改) ---
def extract_atg_indices_with_frames(seq_record):
    seq_str = str(seq_record.seq).upper()
    atg_positions, atg_frames = [], []
    start = 0
    while True:
        pos = seq_str.find("ATG", start)
        if pos == -1: break
        atg_positions.append(pos + 1)
        atg_frames.append(pos % 3)
        start = pos + 1
    return atg_positions, atg_frames


def clean_sequence(seq_str):
    return re.sub(r'[^a-zA-Z]', '', seq_str).upper()


# --- 主分离函数 ---
def separate_data_streaming(fasta_file, output_dir="test_data"):
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features")
    labels_path = os.path.join(output_dir, "labels")
    os.makedirs(features_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    print(f"开始流式处理并分离FASTA文件: {fasta_file}")

    cu = cuRNA.CuRNA(['A', 'C', 'G', 'T'], 90)
    atg_data, processed_count = [], 0

    try:
        for i, rec in enumerate(SeqIO.parse(fasta_file, "fasta")):
            seq_id = str(i + 1)
            header_to_parse = rec.description
            print(f"处理片段 {i + 1}: id={header_to_parse}")

            try:
                # --- 1. 解析三维Header [offset s t] ---
                coords = re.findall(r'-?\d+', header_to_parse)
                if len(coords) < 3:
                    print(f"  跳过: Header '{header_to_parse}' 数字不足3个。")
                    continue

                # --- 2. 直接读取 correct_offset ---
                correct_offset = int(coords[0])
                start_1based_s = int(coords[1])
                stop_1based_t = int(coords[2])

                if correct_offset not in [0, 1, 2]:
                    print(f"  跳过: 无效的 offset 值 {correct_offset}。")
                    continue

                # CDS坐标处理
                cds_1based = [-1, -1]
                if start_1based_s != -1 and start_1based_s <= stop_1based_t:
                    cds_1based = [start_1based_s, stop_1based_t]
                cds_0based = [cds_1based[0] - 1, cds_1based[1] - 1] if cds_1based[0] != -1 else [-1, -1]

                # 3. 清理序列 (逻辑不变)
                cleaned_seq_str = clean_sequence(str(rec.seq))
                if len(cleaned_seq_str) < 6:
                    print(f"  跳过: 序列太短。")
                    continue

                # 4. 【核心】为所有三个可能的offset提取特征和标签
                for offset_id in range(3):
                    orf = cu.get_orf(cleaned_seq_str, offset_id)
                    features = cu.get_slid_cu(orf)

                    # 生成标签 (逻辑与之前版本一致)
                    if offset_id == correct_offset:
                        labels = cu.get_orf_lab(features.shape[0], cds_0based, offset_id)
                    else:
                        labels = np.zeros(features.shape[0], dtype=int)

                    # 保存文件 (逻辑不变)
                    if features.shape[0] > 0:
                        feature_file = os.path.join(features_path, f"{seq_id}_feature_{offset_id}.csv")
                        label_file = os.path.join(labels_path, f"{seq_id}_label_{offset_id}.csv")
                        np.savetxt(feature_file, features, delimiter=',', fmt='%.4f')
                        np.savetxt(label_file, labels, delimiter=',', fmt='%d')
                        print(f"    -> 已保存 offset {offset_id} 的数据。")

                # 5. 提取ATG信息 (逻辑不变)
                atg_positions, atg_frames = extract_atg_indices_with_frames(rec)
                atg_info_pairs = [f"{pos}|{frame}" for pos, frame in zip(atg_positions, atg_frames)]
                atg_data.append(
                    {'seq_id': seq_id, 'atg_count': len(atg_positions), 'atg_info': ",".join(atg_info_pairs)})
                processed_count += 1

            except Exception as e:
                import traceback
                print(f"  处理片段 {seq_id} 失败: {e}")
                traceback.print_exc()
    except FileNotFoundError:
        print(f"错误: 文件 '{fasta_file}' 未找到！")
        return

    if atg_data:
        df_atg = pd.DataFrame(atg_data)
        df_atg.to_csv(os.path.join(output_dir, "atg_indices.csv"), index=False)

    print(f"\n片段分离完成! 成功处理: {processed_count} 个片段")


def main():
    """主函数入口"""
    # 【重要】确保这里的输入文件是使用 create_fragments_final.py 生成的
    fasta_to_process = 'oeu-3UTR-test.fa'

    if not os.path.exists(fasta_to_process):
        print(f"错误: 输入的FASTA文件 '{fasta_to_process}' 不存在!")
        sys.exit(1)

    separate_data_streaming(fasta_to_process, output_dir="test_data")


if __name__ == "__main__":
    main()
