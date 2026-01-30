# Task 2: Chatbot Performance & Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Load Chatbot Log File
# ---------------------------------------------------------
df = pd.read_csv('../healthcare_chatbot_log_40.csv')

print("Dataset loaded successfully")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", list(df.columns))

# ---------------------------------------------------------
# 2. Data Cleaning & Preparation
# ---------------------------------------------------------
df['Fallback_Triggered'] = df['Fallback_Triggered'].astype(bool)

df['NLU_Confidence'] = pd.to_numeric(
    df['NLU_Confidence'], errors='coerce'
)

df['Response_Time (s)'] = pd.to_numeric(
    df['Response_Time (s)'], errors='coerce'
)

df = df.dropna(subset=['Intent_Recognized', 'Actual_Intent'])

# ---------------------------------------------------------
# 3. Intent Recognition Performance 
# ---------------------------------------------------------
print("\n--- Intent Recognition Performance ---")

y_true = df['Actual_Intent']
y_pred = df['Intent_Recognized']

cm = confusion_matrix(y_true, y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sorted(y_pred.unique()),
    yticklabels=sorted(y_true.unique())
)
plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("Intent Recognition Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Core Performance Metrics 
# ---------------------------------------------------------
fallback_rate = df['Fallback_Triggered'].mean()
avg_confidence = df['NLU_Confidence'].mean()
avg_response_time = df['Response_Time (s)'].mean()

completion_rate = (
    df.dropna(subset=['Completion_Rate'])
      .groupby('Session_ID')['Completion_Rate']
      .last()
      .str.replace('%', '')
      .astype(float)
      .mean()
) / 100

print("\n--- Core Metrics ---")
print(f"Fallback Rate: {fallback_rate:.2f}")
print(f"Average NLU Confidence: {avg_confidence:.2f}")
print(f"Average Response Time (s): {avg_response_time:.2f}")
print(f"Session Completion Rate: {completion_rate:.2f}")

# ---------------------------------------------------------
# 5. Exploratory Data Analysis 
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(df['NLU_Confidence'], bins=20, kde=True)
plt.axvline(0.4, color='red', linestyle='--', label='Fallback Threshold')
plt.title("NLU Confidence Distribution")
plt.xlabel("Confidence Score")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.boxplot(
    x=df['Fallback_Triggered'],
    y=df['NLU_Confidence']
)
plt.title("NLU Confidence vs Fallback Trigger")
plt.xlabel("Fallback Triggered")
plt.ylabel("NLU Confidence")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Session-Level Fallback Analysis 
# ---------------------------------------------------------
session_fallbacks = df.groupby('Session_ID')['Fallback_Triggered'].sum()

plt.figure(figsize=(7, 4))
sns.histplot(session_fallbacks, bins=10)
plt.title("Fallback Count per Session")
plt.xlabel("Number of Fallbacks")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. Deep-Dive Metrics 
# ---------------------------------------------------------
successful_sessions = (
    df.groupby('Session_ID')['Fallback_Triggered']
      .max()
      .eq(False)
      .mean()
)

print("\n--- Deep-Dive Metrics ---")
print(f"Conversion Rate (Proxy): {successful_sessions:.2f}")

csat_proxy = np.where(
    (df['Fallback_Triggered'] == False) &
    (df['NLU_Confidence'] >= 0.6),
    1, 0
).mean()

print(f"CSAT Proxy Score: {csat_proxy:.2f}")

df['NPS_Category'] = np.where(
    df['NLU_Confidence'] >= 0.7, 'Promoter',
    np.where(df['NLU_Confidence'] >= 0.4, 'Passive', 'Detractor')
)

nps = (
    (df['NPS_Category'] == 'Promoter').mean()
    - (df['NPS_Category'] == 'Detractor').mean()
) * 100

print(f"NPS Proxy Score: {nps:.2f}")

# ---------------------------------------------------------
# 8. Healthcare Safety-Critical Analysis 
# ---------------------------------------------------------
critical_cases = df[df['Actual_Intent'].str.contains('critical', case=False, na=False)]

critical_fallback_rate = critical_cases['Fallback_Triggered'].mean()

print("\n--- Safety-Critical Analysis ---")
print(f"Fallback Rate for Critical Health Queries: {critical_fallback_rate:.2f}")

print("\nTask 2 Analysis Completed Successfully.")
