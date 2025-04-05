# 🦁 Animal Species Recognition (Custom KNN)

A web-based machine learning project that classifies animal species based on uploaded data using a **K-Nearest Neighbors (KNN) algorithm built from scratch** in Python. The application is powered by a lightweight **HTML frontend** and a custom backend built for speed and simplicity.

---

## 🚀 Features
- Upload animal feature data and get predicted species instantly
- Built entirely from scratch without using external ML libraries
- Clean and interactive frontend using **HTML**
- KNN algorithm implemented from scratch in `knn.py` and `knn2.py`
- Uses CSV data for classification logic (`zoo.csv`, `class.csv`)

---

## 🧠 Tech Stack

| Layer         | Tools                            |
|--------------|----------------------------------|
| Frontend      | HTML, CSS (in `templates/`)      |
| Backend       | Python (Flask-based `app.py`)     |
| ML Model      | Custom KNN (`knn.py`, `knn2.py`) |
| Dataset       | zoo.csv, class.csv               |

---

## 📁 Folder Structure
```
📁 animal_species_knn/
├── 📁 templates/           # HTML frontend
├── 📁 uploads/             # Uploaded files directory
├── app.py                 # Flask app for running the server
├── knn.py                 # Custom KNN classifier
├── knn2.py                # Alternate KNN version
├── id3.py                 # ID3 decision tree logic (if used)
├── zoo.csv                # Zoo dataset
├── class.csv              # Class labels
└── README.md              # Project info
```

---

## 💡 How to Run

### 1. Install Requirements (if any)
This project has no heavy dependencies. Just make sure Python is installed.

### 2. Run the App
```bash
python app.py
```

### 3. Open in Browser
Go to [http://localhost:5000](http://localhost:5000) and start uploading animal data!

---

## 🐾 Dataset Information
- **zoo.csv**: Contains animal features (like hair, feathers, legs)
- **class.csv**: Maps class numbers to animal species

---

## 📌 Notes
- KNN is implemented **without using scikit-learn or other ML libraries**
- The HTML frontend is manually crafted in the `templates/` folder
- Uploaded files are stored in the `uploads/` directory temporarily

---

## 🛠️ Future Improvements
- Add live image-based animal detection
- UI revamp using Tailwind or Bootstrap
- Model accuracy tuning and visualization

---

## 🧬 Made with ❤️ by Ankit Bhaumik
