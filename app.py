from flask import Flask, request, render_template, jsonify
from TurntoPSSM import parse_fasta, run_psiblast, parse_pssm, pssm_to_features_for_svm, pssm_to_features_for_cnn
from keras.models import load_model
import os
import numpy as np
import joblib
import uuid

app= Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 載入模型
svm_model = joblib.load('model/biopython_SVM_model')
cnn_model = load_model('model/biopython_CNN_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'fasta_file' not in request.files:
        return jsonify({'error': '請上傳.FASTA 檔'}), 400
    uploaded_file = request.files['fasta_file']
    file_type = request.form.get('model_type', 'svm')

    uid = str(uuid.uuid4())
    fasta_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}.fasta')
    uploaded_file.save(fasta_path)

    pssm_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}.pssm')
    run_psiblast(fasta_path, pssm_path)
    pssm_matrix = parse_pssm(pssm_path)

    if file_type == 'svm':
        features = pssm_to_features_for_svm(pssm_matrix)
        result = svm_model.predict([features])[0]
    elif file_type == 'cnn':
        features = pssm_to_features_for_cnn(pssm_matrix)
        features = np.expand_dims(features, axis=0)
        result = cnn_model.predict(features)[0]
    else:
        return jsonify({'error': '未知模型類型'}), 400

    return jsonify({'result': str(result)})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=False) 
