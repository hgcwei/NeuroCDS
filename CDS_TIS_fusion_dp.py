import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import re
from typing import Dict, List, Tuple, Any
import csv
import math

BASE_DIR = os.getcwd()
TEST_DATA_DIR = os.path.join(BASE_DIR, 'test_data')
PRED_DIR = os.path.join(TEST_DATA_DIR, 'predictions')
LABELS_DIR = os.path.join(TEST_DATA_DIR, 'labels')
FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, 'CDS_TIS_fusion_dp')
TEMP_FASTA_FILE = os.path.join(BASE_DIR, 'oeu-3UTR-test.fa')

W_CDS = 0.5
W_TIS = 0.5

STOP_CODONS = ['TAA', 'TAG', 'TGA']
MIN_ORF_LENGTH_BP = 10
INF = 1e18

def load_custom_fasta(fasta_path: str) -> Dict[str, str]:
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Sequence file not found: {fasta_path}")
    sequences = {}
    current_seq_id_numeric = 0
    current_seq_key = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                current_seq_id_numeric += 1
                current_seq_key = f"seq_{current_seq_id_numeric}"
                sequences[current_seq_key] = ""
            elif current_seq_key:
                sequences[current_seq_key] += line.upper()
    return sequences


def load_atg_candidates() -> Dict[int, List[Tuple[int, int]]]:
    file_path = os.path.join(TEST_DATA_DIR, 'atg_indices.csv')
    atg_data = {}
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if not row or len(row) < 3: continue
                try:
                    seq_id = int(row[0].strip())
                    atg_info_string = row[2].strip()
                    atg_parts = [part.strip() for part in atg_info_string.split(',') if part.strip()]
                    atg_list = []
                    for atg_info in atg_parts:
                        try:
                            pos_str, frame_str = atg_info.split('|')
                            atg_list.append((int(pos_str.strip()), int(frame_str.strip())))
                        except ValueError:
                            pass
                    if atg_list: atg_data[seq_id] = atg_list
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return atg_data


def load_tis_scores(filename: str) -> Dict[Tuple[int, int], float]:
    file_path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(file_path)
    tis_scores = {}
    for _, row in df.iterrows():
        key = (int(row['sequence_index']), int(row['atg_position']))
        tis_scores[key] = row['logit_class_1']
    return tis_scores


def viterbi_cds(cds_pos: np.ndarray, cds_neg: np.ndarray, tis: np.ndarray, is_stop: np.ndarray) -> Dict[str, Any]:
    n = len(cds_pos)
    if n == 0:
        return {"segments": [], "final_score": -INF}

    dp = np.full((4, n + 1, 3), -INF)
    cds_sum = np.zeros((4, n + 1, 3))
    cds_len = np.zeros((4, n + 1, 3), dtype=int)
    tis_sum = np.zeros((4, n + 1, 3))

    pre_k = np.full((4, n + 1, 3), -1, dtype=int)
    pre_c = np.full((4, n + 1, 3), -1, dtype=int)

    START_PENALTY = 50

    dp[1][1][0] = cds_neg[0]
    dp[1][1][1] = cds_pos[0] - START_PENALTY
    cds_sum[1][1][1] = cds_pos[0]
    cds_len[1][1][1] = 1

    dp[1][1][2] = cds_neg[0]

    for i in range(2, n + 1):
        score_neg = cds_neg[i - 1]
        score_pos = cds_pos[i - 1]
        curr_tis = tis[i - 1]

        for k in range(1, 4):
            if dp[k][i - 1][0] > -INF + 1:
                dp[k][i][0] = dp[k][i - 1][0] + score_neg
                pre_k[k][i][0], pre_c[k][i][0] = k, 0
                cds_sum[k][i][0] = cds_sum[k][i - 1][0]
                cds_len[k][i][0] = cds_len[k][i - 1][0]
                tis_sum[k][i][0] = tis_sum[k][i - 1][0]

            v_cur_1 = dp[k][i - 1][1] if dp[k][i - 1][1] > -INF + 1 and not is_stop[i - 2] else -INF

            v_prev_1 = dp[k - 1][i - 1][0] + curr_tis if k > 1 and curr_tis != 0 and dp[k - 1][i - 1][
                0] > -INF + 1 else -INF

            if v_prev_1 > v_cur_1 and v_prev_1 > -INF + 1:
                dp[k][i][1] = v_prev_1 + score_pos
                pre_k[k][i][1], pre_c[k][i][1] = k - 1, 0
                cds_sum[k][i][1] = cds_sum[k - 1][i - 1][0] + score_pos
                cds_len[k][i][1] = cds_len[k - 1][i - 1][0] + 1
                tis_sum[k][i][1] = tis_sum[k - 1][i - 1][0] + curr_tis
            elif v_cur_1 > -INF + 1:
                dp[k][i][1] = v_cur_1 + score_pos
                pre_k[k][i][1], pre_c[k][i][1] = k, 1
                cds_sum[k][i][1] = cds_sum[k][i - 1][1] + score_pos
                cds_len[k][i][1] = cds_len[k][i - 1][1] + 1
                tis_sum[k][i][1] = tis_sum[k][i - 1][1]

            v_cur_2 = dp[k][i - 1][2] if dp[k][i - 1][2] > -INF + 1 else -INF
            v_prev_2 = dp[k - 1][i - 1][1] if k > 1 and is_stop[i - 2] and dp[k - 1][i - 1][1] > -INF + 1 else -INF

            if v_prev_2 > v_cur_2 and v_prev_2 > -INF + 1:
                dp[k][i][2] = v_prev_2 + score_neg
                pre_k[k][i][2], pre_c[k][i][2] = k - 1, 1
                cds_sum[k][i][2] = cds_sum[k - 1][i - 1][1]
                cds_len[k][i][2] = cds_len[k - 1][i - 1][1]
                tis_sum[k][i][2] = tis_sum[k - 1][i - 1][1]
            elif v_cur_2 > -INF + 1:
                dp[k][i][2] = v_cur_2 + score_neg
                pre_k[k][i][2], pre_c[k][i][2] = k, 2
                cds_sum[k][i][2] = cds_sum[k][i - 1][2]
                cds_len[k][i][2] = cds_len[k][i - 1][2]
                tis_sum[k][i][2] = tis_sum[k][i - 1][2]

    valid_endpoints = [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)]

    best_k, best_c = -1, -1
    max_dp_score = -INF

    for vk, vc in valid_endpoints:
        if (vc == 1 or vc == 2) and cds_len[vk][n][vc] * 3 < MIN_ORF_LENGTH_BP:
            continue
        if dp[vk][n][vc] > max_dp_score:
            max_dp_score = dp[vk][n][vc]
            best_k = vk
            best_c = vc

    if best_k == -1:
        return {"segments": [], "final_score": -INF}

    c_len = cds_len[best_k][n][best_c]
    if c_len == 0:
        final_s = 0.0
    else:
        baseline_score = dp[1][n][0]

        net_advantage = max_dp_score - baseline_score

        t_score = tis_sum[best_k][n][best_c]

        pure_cds_adv_sum = net_advantage - t_score
        avg_relative_cds = pure_cds_adv_sum / c_len

        final_s = (W_CDS * avg_relative_cds * math.log1p(c_len)) + (W_TIS * t_score)

    state_seq = np.zeros(n + 1, dtype=int)
    curr_k, curr_c = best_k, best_c

    for i in range(n, 0, -1):
        state_seq[i] = curr_c
        nxt_k, nxt_c = pre_k[curr_k][i][curr_c], pre_c[curr_k][i][curr_c]
        curr_k, curr_c = nxt_k, nxt_c

    segments = []
    in_cds = False
    start = -1
    for i in range(1, n + 1):
        if state_seq[i] == 1 and not in_cds:
            in_cds, start = True, i
        elif state_seq[i] != 1 and in_cds:
            segments.append((start, i - 1))
            in_cds, start = False, -1

    if in_cds:
        segments.append((start, n))

    return {"segments": segments, "final_score": final_s}


def main():
    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    try:
        tis_pred_files = [f for f in os.listdir(BASE_DIR) if f.startswith('test_predictions_') and f.endswith('.csv')]
        latest_tis_file = sorted(tis_pred_files)[-1]
        sequences = load_custom_fasta(TEMP_FASTA_FILE)
        atg_candidates_map = load_atg_candidates()
        tis_scores_map = load_tis_scores(latest_tis_file)
    except Exception as e:
        print(f"\nError: Data loading failed - {e}")
        return

    prediction_summary_list = []
    sorted_seq_ids = sorted(atg_candidates_map.keys())

    # ================= 插入点 1 =================
    fp_list = []
    fn_list = []
    # ============================================

    for seq_id in sorted_seq_ids:
        seq_key = f"seq_{seq_id}"
        if seq_key not in sequences: continue
        sequence_str = sequences[seq_key]
        seq_len = len(sequence_str)

        for frame in range(3):
            num_codons = (seq_len - frame) // 3
            if num_codons <= 0: continue

            logit_file = os.path.join(PRED_DIR, f"seq_{seq_id}", f"frame_{frame}_logits.csv")
            cds_pos_scores, cds_neg_scores = np.full(num_codons, -10.0), np.full(num_codons, 0.0)
            if os.path.exists(logit_file):
                try:
                    logits_df = pd.read_csv(logit_file, header=None)
                    len_to_copy = min(len(logits_df), num_codons)
                    cds_neg_scores[:len_to_copy] = logits_df.iloc[:len_to_copy, 0].to_numpy()
                    cds_pos_scores[:len_to_copy] = logits_df.iloc[:len_to_copy, 1].to_numpy()
                except pd.errors.EmptyDataError:
                    pass

            tis_scores_for_frame = np.zeros(num_codons)
            for atg_pos, atg_frame in atg_candidates_map.get(seq_id, []):
                if atg_frame == frame:
                    codon_idx = (atg_pos - 1 - frame) // 3
                    if 0 <= codon_idx < num_codons:
                        tis_scores_for_frame[codon_idx] = tis_scores_map.get((seq_id - 1, atg_pos), 0.0)

            is_stop = np.zeros(num_codons, dtype=bool)
            for i in range(num_codons):
                codon = sequence_str[frame + i * 3: frame + i * 3 + 3]
                if codon in STOP_CODONS: is_stop[i] = True

            prediction = viterbi_cds(cds_pos_scores, cds_neg_scores, tis_scores_for_frame, is_stop)
            best_segment_info = {'start': -1, 'stop': -1, 'score': -INF}

            if prediction["segments"]:
                best_seg = max(prediction["segments"], key=lambda s: s[1] - s[0])
                best_segment_info['score'] = prediction['final_score']
                best_segment_info['start'] = frame + (best_seg[0] - 1) * 3 + 1
                best_segment_info['stop'] = frame + (best_seg[1] - 1) * 3 + 3

            prediction_summary_list.append({
                'seq_id': seq_id, 'frame': frame, 'pred_start': best_segment_info['start'],
                'pred_stop': best_segment_info['stop'],
                'score': best_segment_info['score'] if best_segment_info['score'] != -INF else np.nan
            })

    summary_df = pd.DataFrame(prediction_summary_list)
    best_indices = summary_df.loc[summary_df.groupby('seq_id')['score'].idxmax().dropna()].index
    summary_df['is_best_frame'] = False
    summary_df.loc[best_indices, 'is_best_frame'] = True
    summary_df.to_csv(os.path.join(FINAL_OUTPUT_DIR, 'prediction_summary.csv'), index=False, float_format='%.4f')

    tp, fp, tn, fn = 0, 0, 0, 0
    seq_y_true_for_roc = []
    seq_y_score_for_roc = []

    for seq_id in sorted_seq_ids:
        seq_key = f"seq_{seq_id}"
        if seq_key not in sequences: continue

        sequence_str = sequences[seq_key]
        best_pred_row = summary_df[(summary_df['seq_id'] == seq_id) & (summary_df['is_best_frame'] == True)]

        best_frame = -1
        best_score = -9999.0
        if not best_pred_row.empty:
            best_frame = best_pred_row.iloc[0]['frame']
            if not np.isnan(best_pred_row.iloc[0]['score']):
                best_score = best_pred_row.iloc[0]['score']

        seq_true_all = []
        seq_pred_all = []

        for frame in range(3):
            num_codons = (len(sequence_str) - frame) // 3
            if num_codons <= 0: continue

            pred_labels = np.zeros(num_codons, dtype=int)
            if frame == best_frame:
                pred_start = best_pred_row.iloc[0]['pred_start']
                pred_stop = best_pred_row.iloc[0]['pred_stop']
                if pred_start != -1:
                    start_idx = (int(pred_start) - 1 - frame) // 3
                    end_idx = (int(pred_stop) - 1 - frame) // 3
                    if 0 <= start_idx <= end_idx < num_codons:
                        pred_labels[start_idx: end_idx + 1] = 1

            true_labels = np.zeros(num_codons, dtype=int)
            label_file = os.path.join(LABELS_DIR, f"{seq_id}_label_{frame}.csv")
            if os.path.exists(label_file):
                try:
                    loaded_labels = pd.read_csv(label_file, header=None).values.flatten()
                    len_to_copy = min(len(loaded_labels), num_codons)
                    true_labels[:len_to_copy] = loaded_labels[:len_to_copy]
                except Exception:
                    pass

            # ================= 请在这里添加以下代码 =================
            def format_labels(labels):
                if len(labels) == 0: return "无"
                res, start, val = [], 0, labels[0]
                for i in range(1, len(labels)):
                    if labels[i] != val:
                        res.append(f"【{start + 1}-{i}】为{val}")
                        start, val = i, labels[i]
                res.append(f"【{start + 1}-{len(labels)}】为{val}")
                return "，".join(res)

            print(
                f"序列 seq_{seq_id} | 阅读框 {frame} | 真实标签：{format_labels(true_labels)} | 预测标签：{format_labels(pred_labels)}")
            # ========================================================

            seq_true_all.append(true_labels)
            seq_pred_all.append(pred_labels)

            seq_output_dir = os.path.join(FINAL_OUTPUT_DIR, f"seq_{seq_id}")
            os.makedirs(seq_output_dir, exist_ok=True)
            pd.DataFrame(pred_labels).to_csv(os.path.join(seq_output_dir, f"frame_{frame}_codon_labels.csv"),
                                             index=False, header=False)

        if not seq_true_all: continue

        seq_true_arr = np.concatenate(seq_true_all)
        seq_pred_arr = np.concatenate(seq_pred_all)

        exact_match = np.array_equal(seq_true_arr, seq_pred_arr)
        true_all_zeros = np.all(seq_true_arr == 0)
        pred_all_zeros = np.all(seq_pred_arr == 0)

        if exact_match and not true_all_zeros:
            tp += 1
        elif not exact_match and not pred_all_zeros:
            fp += 1
            fp_list.append(seq_id)  # 记录 FP
        elif pred_all_zeros and not true_all_zeros:
            fn += 1
            fn_list.append(seq_id)  # 记录 FN
        elif exact_match and true_all_zeros:
            tn += 1

        seq_y_true_for_roc.append(0 if true_all_zeros else 1)
        seq_y_score_for_roc.append(best_score)

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * sn) / (precision + sn) if (precision + sn) > 0 else 0.0

    try:
        auroc = roc_auc_score(seq_y_true_for_roc, seq_y_score_for_roc) if len(
            np.unique(seq_y_true_for_roc)) > 1 else 0.0
    except Exception:
        auroc = 0.0

    print("\n================ Viterbi Sequence-Level Performance ================")
    print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"  SN (Sensitivity):     {sn:.4f}")
    print(f"  SP (Specificity):     {sp:.4f}")
    print(f"  Precision:            {precision:.4f}")
    print(f"  F1-score:             {f1_score:.4f}")
    print(f"  auROC:                {auroc:.4f}")
    print("====================================================================")

    # ============================================
    # 在所有打印结束的最后，输出这两行：
    print(f"\n[调试信息] 共有 {len(fp_list)} 个 FP，前 20 个序列 ID 为: {fp_list[:20]}")
    print(f"[调试信息] 共有 {len(fn_list)} 个 FN，前 20 个序列 ID 为: {fn_list[:20]}")
    # ============================================

if __name__ == '__main__':
    main()
