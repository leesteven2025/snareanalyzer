from TurntoPSSM import parse_pssm, pssm_to_features_for_svm
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import joblib

# 讀取資料夾中所有 pssm 並產生 X, y
def load_dataset_for_svm(pssm_folder, labels_dict):
    """
    讀取 PSSM 資料夾，轉成 SVM 使用的 40 維特徵向量（20 維平均 + 20 維標準差）

    回傳：
    - X.shape = (樣本數, 40)
    - y.shape = (樣本數,)
    """
    X = []
    y = []
    for fname in os.listdir(pssm_folder):
        if not fname.endswith('.pssm'):
            continue
        seq_id = fname.replace('.pssm', '')
        pssm_path = os.path.join(pssm_folder, fname)
        pssm_matrix = parse_pssm(pssm_path)
        feature_vector = pssm_to_features_for_svm(pssm_matrix)
        X.append(feature_vector)
        y.append(labels_dict.get(seq_id, 0))  # 預設沒標籤就為 0
    return np.array(X), np.array(y)


if __name__ == "__main__":
    pssm_folder = "C:/Users/USER/pssm_test"

    # 你要事先準備好這個標籤字典：key 是序列ID(檔名前綴)，value 是 0/1 標籤
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
        # 請補足全部序列ID與標籤
    }

    X, y = load_dataset_for_svm(pssm_folder, labels)
    
    # 分割訓練測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

    
    svc = svm.SVC(probability=True)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1, 10]
    }

    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)

    best_svc = grid_search.best_estimator_

    y_pred = best_svc.predict(X_test)
    y_prob = best_svc.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    # 建立 SVM 模型
    #clf = svm.SVC(kernel='rbf', probability=True)

    # 訓練
    #clf.fit(X_train, y_train)

    # 測試
    #y_pred = clf.predict(X_test)
    #y_prob = clf.predict_proba(X_test)[:, 1]

    #print(classification_report(y_test, y_pred))
    
    joblib.dump(grid_search.best_estimator_, "biopython_SVM_model")
