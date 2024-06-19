import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, confusion_matrix

# Load the data
data = pd.read_csv("C:/Users/Manikandan/Desktop/class/Data Management and Big Data/finalproj/merged_dataset.csv")

# Handling missing values
data['Credit_Product'] = data['Credit_Product'].fillna('No')

# Drop rows where 'Is_Lead' is missing assuming these rows are split between train and test set
data.dropna(subset=['Is_Lead'], inplace=True)

# Label encoding for categorical variables
encoder = LabelEncoder()
categorical_columns = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active']
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# Splitting the data into features and target
X = data.drop(['ID', 'Is_Lead'], axis=1)
y = data['Is_Lead']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression(max_iter=5000, random_state=42)
random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

# List of models for iteration
models = {
    'Logistic Regression': log_reg,
    'Random Forest': random_forest,
    'Gradient Boosting': gradient_boosting
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append((name, accuracy, roc_auc, report, y_pred_proba, cm))

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy and ROC AUC
model_names = [result[0] for result in results]
accuracies = [result[1] for result in results]
roc_aucs = [result[2] for result in results]

axes[0, 0].bar(model_names, accuracies, color='skyblue')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_ylabel('Accuracy')

axes[0, 1].bar(model_names, roc_aucs, color='salmon')
axes[0, 1].set_title('Model ROC AUC')
axes[0, 1].set_ylabel('ROC AUC')

# ROC Curves
for name, _, _, _, y_pred_proba, _ in results:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1, 0].plot(fpr, tpr, label=name)
axes[1, 0].plot([0, 1], [0, 1], 'k--')
axes[1, 0].set_title('ROC Curves')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend()

# Confusion Matrices
for name, _, _, _, _, cm in results:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title(f'{name} Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('True')

plt.tight_layout()
plt.show()

# Feature Importances for Random Forest and Gradient Boosting
for name, model in models.items():
    if name in ['Random Forest', 'Gradient Boosting']:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        features = X.columns

        plt.figure(figsize=(12, 6))
        plt.title(f'{name} Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), features[indices], rotation=90)
        plt.tight_layout()
        plt.show()
