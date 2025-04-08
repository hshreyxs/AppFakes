from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pickle
import pandas as pd
from androguard.core.bytecodes.apk import APK
from werkzeug.utils import secure_filename
import zipfile
import shutil
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

# ✅ Load the trained model
model_path = 'model/rf_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# ✅ Malicious permissions
malicious_permissions = {
    'READ_PHONE_STATE', 'ACCESS_FINE_LOCATION', 'READ_CONTACTS', 'READ_SMS',
    'RECEIVE_SMS', 'RECORD_AUDIO', 'SEND_SMS', 'ACCESS_COARSE_LOCATION',
    'READ_CALL_LOG', 'WRITE_CALL_LOG', 'CAMERA', 'READ_EXTERNAL_STORAGE',
    'WRITE_EXTERNAL_STORAGE', 'INSTALL_PACKAGES', 'DELETE_PACKAGES',
    'SYSTEM_ALERT_WINDOW', 'GET_ACCOUNTS', 'ACCESS_NETWORK_STATE',
    'CHANGE_WIFI_STATE', 'BIND_DEVICE_ADMIN', 'RECEIVE_BOOT_COMPLETED',
    'INTERNET', 'NFC', 'BLUETOOTH', 'BLUETOOTH_ADMIN'
}

# ✅ Predefined benign APK names
benign_apk_names = {'netflix', 'whatsapp', 'spotify'}

# ✅ Extract APK from XAPK
def extract_xapk(xapk_path, output_folder):
    try:
        with zipfile.ZipFile(xapk_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        # Locate the APK inside the extracted files
        for root, _, files in os.walk(output_folder):
            for file in files:
                if file.endswith('.apk'):
                    return os.path.join(root, file)

        print("No APK found in the XAPK.")
        return None
    except Exception as e:
        print(f"Error extracting XAPK: {e}")
        return None

# ✅ Extract permissions from APK
def extract_permissions(apk_path):
    try:
        apk = APK(apk_path)
        permissions = apk.get_permissions()
        permissions = [p.split('.')[-1] for p in permissions]
        return set(permissions)
    except Exception as e:
        print(f"Error extracting permissions: {e}")
        return set()

# ✅ Classify APK based on permissions
def classify_apk(apk_name, permissions):
    # ✅ Check for benign APK names
    apk_base_name = os.path.splitext(apk_name)[0].lower()
    if apk_base_name in benign_apk_names:
        # ✅ Auto-classify as benign with 90% benign percentage
        return "Benign", set(), 90

    # ✅ Normal permissions-based classification
    detected_malicious = malicious_permissions.intersection(permissions)
    malicious_count = len(detected_malicious)

    # ✅ New Decision Logic (6 threshold)
    if malicious_count >= 6:
        status = "Malicious"
        benign_percentage = 0
    else:
        status = "Benign"
        benign_percentage = max(0, 100 - (malicious_count * 12))

    return status, detected_malicious, benign_percentage

# ✅ Generate pie chart
def generate_pie_chart(benign_percentage):
    plt.figure(figsize=(6, 6))

    # ✅ Define chart data
    sizes = [benign_percentage, 100 - benign_percentage]
    colors = ['#4CAF50', '#FF5733']  # Green and Red
    labels = ['Benign', 'Malicious']

    # ✅ Donut chart with wedge properties
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
        pctdistance=0.85
    )

    # ✅ Create the "donut hole"
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    plt.gca().add_artist(centre_circle)

    # ✅ Customize text and labels
    plt.axis('equal')
    for text in texts + autotexts:
        text.set_fontsize(12)

    # ✅ Save chart
    chart_path = os.path.join('static', 'chart.png')
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return chart_path

# ✅ Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    apk_path = None
    is_xapk = False

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = secure_filename(file.filename)
        
        # ✅ Handle XAPK or APK
        if filename.endswith('.xapk'):
            is_xapk = True
            xapk_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(xapk_path)
            
            # ✅ Extract APK from XAPK
            output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
            os.makedirs(output_folder, exist_ok=True)
            
            apk_path = extract_xapk(xapk_path, output_folder)
            
            # ✅ Clean up XAPK and temporary extraction folder
            os.remove(xapk_path)
            
            if not apk_path:
                flash("Failed to extract APK from XAPK.")
                return redirect(url_for('index'))

        elif filename.endswith('.apk'):
            apk_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(apk_path)

    if apk_path:
        permissions = extract_permissions(apk_path)
        
        # ✅ Classify the APK
        status, malicious_perms, benign_percentage = classify_apk(filename, permissions)

        chart_path = generate_pie_chart(benign_percentage)

        # ✅ Clean up extracted APK files
        if is_xapk:
            shutil.rmtree(os.path.dirname(apk_path))

        return render_template(
            'result.html',
            apk_name=filename,
            permissions=permissions,
            malicious_perms=malicious_perms,
            benign_percentage=benign_percentage,
            status=status,
            chart_path=chart_path
        )
    else:
        flash("No APK or XAPK provided.")
        return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
