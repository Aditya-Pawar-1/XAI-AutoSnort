# AutoSnort

AutoSnort is a Python-based web application that uses a "real-world" machine learning model to detect network intrusions. Its primary feature is the use of Explainable AI (XAI) to not only identify an attack but to also **automatically generate a corresponding Snort rule** based on the features the model found most important.

This project is trained on the CIC-IDS-2017 dataset and uses a Random Forest model specifically balanced to provide **high recall on minority attack classes** (like Bots, Infiltration, and Web Attacks), making it more effective than a standard accuracy-focused model.

## 🚀 Key Features

  * **High-Recall IDS:** Uses a Random Forest model trained with `class_weight='balanced'` to effectively catch rare and dangerous attacks.
  * **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to explain *why* the model classified a packet as malicious.
  * **Automatic Snort Rule Generation:** Uses the top feature from the SHAP explanation to automatically generate a dynamic, behavior-based Snort rule.
  * **Web-Based Interface:** A simple Flask app provides a UI for entering packet data and viewing the prediction, XAI explanation, and generated rule.
  * **Complete ML Pipeline:** Includes the full Jupyter Notebook (`CIC_IDS_2017_Preprocessing.ipynb`) used to clean, preprocess, and train the final model.
  * **Detailed Reporting:** The training script automatically saves a confusion matrix, classification reports, and raw test files for easy analysis.

## 💻 Technology Stack

  * **Backend:** Flask
  * **Machine Learning:** scikit-learn, pandas, numpy
  * **Explainable AI:** SHAP
  * **Data Visualization:** matplotlib, seaborn

## 📂 Project Structure

```
AutoSnort/
├── app/
│   ├── app.py                # The main Flask web server
│   ├── static/style.css      # CSS for the web app
│   └── templates/index.html   # HTML template for the web app
│
├── Datasets/
│   └── CICIDS-2017/          # Directory for the 8 raw CIC-IDS-2017 CSVs
│
├── model_cic/
│   ├── random_forest_model_cic.pkl    # The final trained Random Forest model
│   ├── encoder_cic.pkl                # The LabelEncoder object
│   ├── scaler_values_cic.pkl          # The dictionary of normalization values
│   ├── feature_names_cic.pkl          # The list of 15 features the model expects
│   │
│   ├── test_set_raw_values.csv        # The UN-NORMALIZED test set (for demo)
│   ├── demo_samples_raw_by_class.csv  # One RAW median sample for each attack type
│   │
│   ├── confusion_matrix_cicids.png    # Final confusion matrix of the model
│   ├── classification_report_per_class.csv # Model's performance (Recall, Precision)
│   └── metrics_per_class.csv          # Model's TP, TN, FP, FN counts
│
├── notebooks/
│   └── CIC_IDS_2017_Preprocessing.ipynb # Jupyter Notebook to run the full training pipeline
│
└── requirements.txt          # All Python libraries required for the project
```

## ⚙️ Setup & Installation

**1. Clone the Repository**

```bash
git clone https://github.com/Aditya-Pawar-1/AutoSnort.git
cd autosnort
```

**2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Download the Dataset (Crucial Step)**
This model requires the CIC-IDS-2017 dataset.

  * Download all 8 `.csv` files (Monday-Friday) from a reliable source (e.g., [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/cic-ids-2017/cicids2017)).
  * Place all 8 CSV files inside the `Datasets/CICIDS-2017/` directory. The training notebook (`notebooks/CIC_IDS_2017_Preprocessing.ipynb`) is pre-configured to read from this folder.

## 🚀 How to Run

There are two ways to use this project:

### 1\. Run the Pre-Trained Web App (Recommended)

This project comes with a pre-trained model, so you can run the web app immediately.

1.  **Navigate to the app directory:**

    ```bash
    cd app
    ```

2.  **Run the Flask server:**

    ```bash
    python app.py
    ```

3.  **Open the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

4.  **How to Test:**
    To demo the app, you need to provide **raw, un-normalized** data. The easiest way is to use the provided demo file:

      * Open `model_cic/demo_samples_raw_by_class.csv`.
      * Copy the 15 feature values from one of the rows (e.g., the `Bot` row).
      * Paste these values into the form on the web page to see the prediction and generated Snort rule.

### 2\. Train a New Model (Optional)

If you want to re-train the model from scratch:

1.  **Ensure the dataset is downloaded** and placed in the `Datasets/CICIDS-2017/` folder (see Setup step 4).
2.  **Run the Jupyter Notebook:**
    Open and run all cells in `notebooks/CIC_IDS_2017_Preprocessing.ipynb`.
3.  **Done\!** The script will automatically:
      * Load and preprocess all 8 CSVs.
      * Train the high-recall Random Forest model.
      * Save all the new model (`.pkl`) files, reports, and demo CSVs into the `model_cic/` directory, overwriting the old ones.
      * Your `app.py` will automatically use this new model the next time you run it.

## 📊 Model Performance

The final model is a Random Forest (`max_depth=10`) trained on the "Top 15" features from the XAI-IDS paper. It uses `class_weight='balanced'` to prioritize attack detection.

The goal is **High Recall** (catching attacks), not just high accuracy.

### Final Confusion Matrix

### Key Performance Metrics

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Bot** | 0.0753 | **0.9830** | 0.1398 |
| **Brute Force** | 1.0000 | **0.9858** | 0.9929 |
| **Dos/Ddos** | 0.9985 | 0.9978 | 0.9981 |
| **Infiltration**| 1.0000 | **0.8182** | 0.9000 |
| **PortScan** | 0.6810 | **0.9896** | 0.8068 |
| **Web Attack** | 0.0474 | **0.6341** | 0.0883 |
| **BENIGN** | 0.9998 | 0.9856 | 0.9926 |
| | | | |
| **Overall** | **98.65% Accuracy** | | |

*For a full breakdown of True Positives, False Positives, etc., see `model_cic/metrics_per_class.csv`.*
