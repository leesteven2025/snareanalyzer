import os
import subprocess
import numpy as np

def parse_fasta(file_path):
    """
    讀取 FASTA 檔案，回傳蛋白質序列字串（去除標頭與換行）
    """
    sequence = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence.append(line)
    return ''.join(sequence)


def run_psiblast(fasta_path, output_pssm_path, db= "C:/Users/USER/yourdb", num_iterations=3):
    """
    使用 psiblast 將 fasta 檔比對產生 PSSM 檔案（需系統已安裝 psiblast 並設定資料庫）
    """
    command = [
        "psiblast",
        "-query", fasta_path,
        "-db", db,
        "-num_iterations", str(num_iterations),
        "-out_ascii_pssm", output_pssm_path,
        "-out", os.devnull
    ]
    subprocess.run(command, check=True)


def parse_pssm(pssm_path):
    """
    解析 PSSM 檔案內容為 shape=(序列長度, 20) 的 numpy array
    """
    pssm_data = []
    start_read = False
    with open(pssm_path, 'r') as f:
        for line in f:
            if start_read:
                if line.strip() == '' or line.startswith('Lambda'):
                    break
                parts = line.strip().split()
                if len(parts) >= 22 and parts[0].isdigit():
                    scores = list(map(int, parts[2:22]))
                    pssm_data.append(scores)
            elif line.startswith('Last position-specific scoring matrix computed'):#開頭是這個即開始'
                start_read = True
    return np.array(pssm_data)


def pssm_to_features_for_svm(pssm_matrix):
    """
    將 PSSM 矩陣轉換為 SVM 使用的 40 維特徵向量（20維平均 + 20維標準差）
    """
    mean_feat = np.mean(pssm_matrix, axis=0)
    std_feat = np.std(pssm_matrix, axis=0)
    return np.concatenate([mean_feat, std_feat])


def pssm_to_features_for_cnn(pssm_matrix, max_len=500):
    """
    將 PSSM 矩陣整理成 CNN 使用的輸入格式 (max_len, 20)
    - 若太短補零，太長則截斷
    """
    if pssm_matrix.shape[0] > max_len:
        pssm_matrix = pssm_matrix[:max_len, :]
    else:
        pad_len = max_len - pssm_matrix.shape[0]
        padding = np.zeros((pad_len, 20))
        pssm_matrix = np.vstack((pssm_matrix, padding))
    return pssm_matrix
