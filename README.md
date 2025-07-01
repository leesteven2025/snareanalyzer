# 🔬 生物資訊預測平台 snareanalyzer

本專案為一個基於 Flask 的生物資訊預測網站，可上傳 FASTA 檔案、轉換成 PSSM，再透過 CNN 或 SVM 模型進行snare蛋白質序列的分類預測。

---

## 🧩 專案功能

- 上傳 `.fasta` 檔案
- 自動執行 PSI-BLAST 轉換為 PSSM
- 選擇預測模型：SVM 或 CNN
- 回傳預測結果並顯示在網頁上

---

## 📁 專案結構

snareanalyzer/
app.py # Flask 主程式

TurntoPSSM.py # 工具函式（FASTA → PSSM 解析）

CNN.py / SVM.py # 模型訓練腳本（可選放入 training/）

model/ # 儲存模型檔（.pkl / .keras）

templates/ # 前端頁面（index.html）

uploads/ # 暫存使用者上傳的檔案

requirements.txt # Python 套件清單

render.yaml # Render 雲端部署設定（可選）

.gitignore # 忽略上傳檔案與模型

---

## ⚙️ 依賴套件
Flask

numpy

joblib

tensorflow

scikit-learn

biopython（如有）
