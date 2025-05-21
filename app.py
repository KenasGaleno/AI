from flask import Flask, render_template, request
from deepface import DeepFace
import os

app = Flask(__name__)

# Folder untuk menyimpan gambar sementara
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fungsi untuk memeriksa similarity antara dua gambar dan analisis wajah
def check_similarity(image1_path, image2_path):
    # Periksa similaritas gambar menggunakan DeepFace
    result = DeepFace.verify(image1_path, image2_path)
    
    # Facial Attribute Analysis (Usia, Jenis Kelamin, Emosi, Ras)
    analysis1 = DeepFace.analyze(image1_path, actions=['age', 'gender', 'emotion', 'race'])
    analysis2 = DeepFace.analyze(image2_path, actions=['age', 'gender', 'emotion', 'race'])
    
    return result, analysis1, analysis2

@app.route('/')
def index():
    return render_template('index.html')  # Halaman utama

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file gambar
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    # Simpan file gambar ke direktori upload sementara
    image1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image1.jpg')
    image2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image2.jpg')
    
    image1.save(image1_path)
    image2.save(image2_path)
    
    # Periksa similaritas antara kedua gambar dan analisis wajah
    result, analysis1, analysis2 = check_similarity(image1_path, image2_path)
    
    # Tampilkan hasil
    similarity = result['verified']
    return render_template('result.html', similarity=similarity, analysis1=analysis1, analysis2=analysis2)

if __name__ == '__main__':
    app.run(debug=True)
