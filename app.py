from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os
import tensorflow as tf
from datetime import datetime
import gdown
import psycopg2

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)  # removed custom template path (use /templates)
app.secret_key = os.environ.get("SECRET_KEY", "secret123")

# -----------------------------
# Admin credentials
# -----------------------------
ADMIN_USERNAME = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASS", "admin123")

# -----------------------------
# Load model (Google Drive)
# -----------------------------
MODEL_PATH = "vgg16_best.h5"
MODEL_URL = "https://drive.google.com/uc?id=1sq-Cz_Jvtyns3bxx8_kqdt8dfZDInZMr"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=True, fuzzy=True)

print("Loading model...")
model = load_model(MODEL_PATH)

class_labels = [
    'adenocarcinoma',
    'large.cell.carcinoma',
    'normal',
    'squamous.cell.carcinoma'
]

# -----------------------------
# PostgreSQL (Render DB)
# -----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    db = psycopg2.connect(DATABASE_URL, sslmode='require')
    db.autocommit = True
    cursor = db.cursor()
else:
    print("⚠️ DATABASE_URL not set. Running without DB.")
    cursor = None

# -----------------------------
# Grad-CAM
# -----------------------------
def get_gradcam(img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer('block5_conv3').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

# -----------------------------
# Stage calculation
# -----------------------------
def calculate_stage(heatmap):
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    coverage = np.mean(heatmap > 0.5) * 100

    if coverage <= 10:
        return coverage, "Stage I"
    elif coverage <= 25:
        return coverage, "Stage II"
    elif coverage <= 45:
        return coverage, "Stage III"
    return coverage, "Stage IV"

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

# -----------------------------
# Admin
# -----------------------------
@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect('/admin')
        return render_template('admin_login.html', error="Invalid")

    return render_template('admin_login.html')

@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect('/admin_login')

    if cursor:
        cursor.execute("SELECT * FROM patients ORDER BY timestamp DESC")
        data = cursor.fetchall()
    else:
        data = []

    return render_template('admin.html', patients=data)

@app.route('/admin_logout')
def logout():
    session.clear()
    return redirect('/')

# -----------------------------
# Prediction
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['scan']
    patient_name = request.form.get('patient_name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    smoking = request.form.get('smoking')

    img = image.load_img(BytesIO(file.read()), target_size=(224,224))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    idx = np.argmax(preds)

    label = class_labels[idx]
    confidence = float(np.max(preds) * 100)

    if confidence < 90:
        return render_template('results.html', prediction="Invalid Image")

    heatmap = get_gradcam(arr)

    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    coverage, stage = calculate_stage(heatmap)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    upload_dir = os.path.join(app.root_path, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    scan_filename = f"scan_{timestamp}.png"
    gradcam_filename = f"gradcam_{timestamp}.png"

    cv2.imwrite(os.path.join(upload_dir, scan_filename), orig)
    cv2.imwrite(os.path.join(upload_dir, gradcam_filename), superimposed)

    # Save to DB (safe)
    if cursor:
        cursor.execute("""
            INSERT INTO patients 
            (patient_name, age, gender, smoking, scan_path, gradcam_path, prediction, confidence, stage)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,(patient_name, age, gender, smoking, scan_filename, gradcam_filename, label, confidence, stage))

    return render_template('results.html',
        prediction=label,
        confidence=f"{confidence:.2f}%",
        stage=stage,
        coverage=f"{coverage:.2f}%",
        gradcam=gradcam_filename,
        original=scan_filename
    )

# -----------------------------
# Serve uploads
# -----------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    upload_dir = os.path.join(app.root_path, 'uploads')
    return send_from_directory(upload_dir, filename)

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)