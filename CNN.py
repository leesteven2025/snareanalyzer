from TurntoPSSM import parse_pssm, pssm_to_features_for_cnn
import os
import numpy as np
import tensorflow as tf

def load_cnn_dataset(pssm_dir, labels_dict, max_len=500):
    X = []
    y = []
    for fname in os.listdir(pssm_dir):
        if fname.endswith('.pssm'):
            seq_id = fname.replace('.pssm', '')
            pssm_path = os.path.join(pssm_dir, fname)
            pssm_matrix = parse_pssm(pssm_path)
            cnn_input = pssm_to_features_for_cnn(pssm_matrix, max_len)
            X.append(cnn_input)
            y.append(labels_dict.get(seq_id, 0))  # 預設為 0
    return np.array(X), np.array(y)

# 範例用法
if __name__ == "__main__":
    pssm_folder = "C:/Users/USER/pssm_test"  # 你的 PSSM 資料夾
    # 範例標籤字典，key是序列ID(檔名前綴)，value是標籤
    labels = {
        "001": 1,
        "002": 1,
        "003": 1,
        "004": 1,
        "005": 1,
        "006": 1,
        "007": 1,
        "008": 1,
        "009": 1,
        "010": 1,
        "011": 1,
        "012": 1,
        "013": 1,
        "014": 1,
        "015": 1,
        "016": 1,
        "017": 1,
        "018": 1,
        "019": 1,
        "020": 1,
        "021": 0,
        "022": 0,
        "023": 0,
        "024": 0,
        "025": 0,
        "026": 0,
        "027": 0,
        "028": 0,
        "029": 0,
        "030": 0,
        "031": 0,
        "032": 0,
        "033": 0,
        "034": 0,
        "035": 0,
        "036": 0,
        "037": 0,
        "038": 0,
        "039": 0,
        "040": 0,
        # 其他序列ID與標籤
    }

    X, y = load_cnn_dataset(pssm_folder, labels, max_len=500)

    print("X shape:", X.shape)  # (樣本數, 500, 20)
    print("y shape:", y.shape)  # (樣本數,)



# 建立 CNN 模型


import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(500, 20)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=16, validation_split=0.4)


model.save("biopython_CNN_model.h5")  