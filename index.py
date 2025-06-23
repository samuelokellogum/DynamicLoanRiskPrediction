# Authors: Samuel Okello, Omollo Tanga

# SECTION 1: Install + Setup (first run only)
!pip install kaggle shap scikit-learn tensorflow pandas matplotlib seaborn --quiet

from google.colab import files
files.upload()  # Upload kaggle.json manually

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d wordsforthewise/lending-club
!unzip -o lending-club.zip -d lending_data

# SECTION 2: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# SECTION 3: Load & Prepare Data
df = pd.read_csv('lending_data/accepted_2007_to_2018Q4.csv', low_memory=False)

# Select and clean features
df = df[['loan_amnt', 'issue_d', 'last_pymnt_d', 'loan_status']].dropna()
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
df['defaulted'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

# Parse dates and calculate delay
df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], errors='coerce')
df['days_late'] = (df['last_pymnt_d'] - df['issue_d']).dt.days
df = df.dropna(subset=['days_late'])
df = df[df['days_late'] >= 0]

# Normalize features
scaler = MinMaxScaler()
df[['loan_amnt', 'days_late']] = scaler.fit_transform(df[['loan_amnt', 'days_late']])

# SECTION 4: Create Sequences for LSTM
features = ['loan_amnt', 'days_late']
sequence_length = 5

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        seq_x = data.iloc[i:i+window][features].values
        seq_y = data.iloc[i+window]['defaulted']
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# SECTION 5: Logistic Regression (Flat Features)
Xf = df[features].values
yf = df['defaulted'].values
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(Xf_train, yf_train)
y_pred = logreg.predict(Xf_test)

print("\n📊 Logistic Regression Results")
print(classification_report(yf_test, y_pred))
print("AUC-ROC:", roc_auc_score(yf_test, logreg.predict_proba(Xf_test)[:, 1]))

# SECTION 6: LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
y_lstm_pred = model.predict(X_test).flatten()
y_lstm_label = (y_lstm_pred > 0.5).astype(int)

print("\n📊 LSTM Results")
print(classification_report(y_test, y_lstm_label))
print("AUC-ROC:", roc_auc_score(y_test, y_lstm_pred))

# SECTION 7: SHAP Explanation (Logistic Regression)
explainer = shap.Explainer(logreg, Xf_test)
shap_values = explainer(Xf_test)
print("\n📈 SHAP Summary Plot (Logistic Regression)")
shap.summary_plot(shap_values, Xf_test, feature_names=features)

# SECTION 8: Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_lstm_label), annot=True, fmt='d')
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# SECTION 9: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_lstm_pred)
plt.plot(fpr, tpr, label='LSTM (AUC = {:.2f})'.format(roc_auc_score(y_test, y_lstm_pred)))
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()
