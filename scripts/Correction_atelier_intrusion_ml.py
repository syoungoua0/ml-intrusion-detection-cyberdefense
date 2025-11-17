# atelier_intrusion_ml.py
# Implémentation d'un pipeline ML pour la détection d'intrusion avec scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

def generate_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)

    data = {
        'packet_size': np.random.normal(500, 200, n_samples),        # bytes
        'duration': np.random.exponential(2, n_samples),             # seconds
        'src_bytes': np.random.lognormal(8, 2, n_samples),           # bytes
        'dst_bytes': np.random.lognormal(7, 2, n_samples),           # bytes
        'num_failed_logins': np.random.poisson(0.1, n_samples),      # count
        'protocol_type': np.random.choice([0, 1, 2], n_samples)      # categorical encoded
    }

    df = pd.DataFrame(data)

    # Règles simples pour créer des labels d'intrusion
    intrusion_mask = (
        (df['packet_size'] > 800) |
        (df['duration'] > 5) |
        (df['num_failed_logins'] > 2) |
        ((df['src_bytes'] > np.percentile(df['src_bytes'], 90)) &
         (df['dst_bytes'] > np.percentile(df['dst_bytes'], 90)))
    )

    df['is_intrusion'] = intrusion_mask.astype(int)

    return df

def explore_data(df):
    print("Distribution des classes:")
    print(df['is_intrusion'].value_counts(), "\n")

    print("Proportions:")
    print((df['is_intrusion'].value_counts(normalize=True) * 100).round(2), "%\n")

    print("Statistiques descriptives par classe:")
    print(df.groupby('is_intrusion').describe().transpose(), "\n")

    # Visualisations
    features_to_plot = ['packet_size', 'duration', 'src_bytes', 'dst_bytes', 'num_failed_logins']

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(features_to_plot, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=df, x=col, hue='is_intrusion', common_norm=False)
        plt.title(f'Distribution de {col}')
    plt.tight_layout()
    plt.savefig('distributions_features.png')
    print("Graphique sauvegardé: distributions_features.png")
    plt.close()

    # Corrélation (hors target)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.drop(columns=['is_intrusion']).corr(), annot=False, cmap='coolwarm', center=0)
    plt.title('Corrélation des features')
    plt.savefig('correlation_matrix.png')
    print("Graphique sauvegardé: correlation_matrix.png\n")
    plt.close()

def build_and_evaluate(df, random_state=42):
    X = df.drop('is_intrusion', axis=1)
    y = df['is_intrusion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Standardisation (utile pour certains modèles et pour homogénéiser les échelles)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modèle
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',  # aide si déséquilibre
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Prédictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

    # Évaluation
    cm = confusion_matrix(y_test, y_pred)
    print("Matrice de confusion:")
    print(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Pred 0', 'Pred 1']), "\n")

    print("Rapport de classification:")
    print(classification_report(y_test, y_pred, digits=4))

    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_score:.4f}")

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'RandomForest (AUC={auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    print("\nGraphique sauvegardé: roc_curve.png")
    plt.close()

    # Importance des features (sur données originales, l'ordre correspond aux colonnes de X)
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nImportance des features:")
    print(importances, "\n")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, orient='h')
    plt.title('Importance des features - RandomForest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Graphique sauvegardé: feature_importance.png\n")
    plt.close()

    return {
        'model': rf_model,
        'scaler': scaler,
        'metrics': {
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'auc_roc': auc_score
        }
    }

def main():
    df_network = generate_dataset(n_samples=1000, random_state=42)
    print("Aperçu des données:")
    print(df_network.head(), "\n")

    explore_data(df_network)
    results = build_and_evaluate(df_network)

if __name__ == "__main__":
    main()
