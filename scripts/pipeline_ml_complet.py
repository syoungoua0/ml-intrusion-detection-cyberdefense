# pipeline_ml_complet.py
# Pipeline ML complet pour la detection d'intrusion reseau
# Inclut: exploration, modelisation, optimisation et techniques avancees

# ================================================
# 1) Imports
# ================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

np.random.seed(42)

# ================================================
# 2) Generation du dataset reseau synthetique
# ================================================
print("="*80)
print("PIPELINE ML COMPLET - DETECTION D'INTRUSION")
print("="*80)
print()

print("[*] Generation du dataset reseau synthetique...")
n_samples = 1000

data = {
    'packet_size': np.random.normal(500, 200, n_samples),
    'duration': np.random.exponential(2, n_samples),
    'src_bytes': np.random.lognormal(8, 2, n_samples),
    'dst_bytes': np.random.lognormal(7, 2, n_samples),
    'num_failed_logins': np.random.poisson(0.1, n_samples),
    'protocol_type': np.random.choice([0, 1, 2], n_samples)
}

df_network = pd.DataFrame(data)
df_network['packet_size'] = df_network['packet_size'].clip(lower=1)
df_network['duration'] = df_network['duration'].clip(lower=0.001)

intrusion_mask = (
    (df_network['packet_size'] > 800) |
    (df_network['duration'] > 5) |
    (df_network['num_failed_logins'] > 2)
)

df_network['is_intrusion'] = intrusion_mask.astype(int)

print("[OK] Dataset cree avec succes !")
print(f"   Taille: {len(df_network)} echantillons")
print()
print("Apercu des donnees:")
print(df_network.head())
print()

# ================================================
# 3) Exploration des donnees
# ================================================
print("="*80)
print("EXPLORATION DES DONNEES")
print("="*80)
print()

print("[*] Distribution des classes:")
print(df_network['is_intrusion'].value_counts())
print()
print("Proportions:")
print(df_network['is_intrusion'].value_counts(normalize=True))
print()

# Visualisation: Distribution des classes
plt.figure(figsize=(6,4))
sns.countplot(x='is_intrusion', data=df_network, palette='Set2')
plt.title("Distribution des classes")
plt.xlabel("Is Intrusion (0=Normal, 1=Intrusion)")
plt.ylabel("Nombre d'echantillons")
plt.savefig('01_distribution_classes.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 01_distribution_classes.png")
plt.close()

# Histogrammes par feature
print()
print("[*] Generation des histogrammes par feature...")
features_to_plot = ['packet_size', 'duration', 'num_failed_logins']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, f in enumerate(features_to_plot):
    sns.histplot(df_network, x=f, hue='is_intrusion', bins=40,
                 stat="density", common_norm=False, ax=axes[idx], palette='Set1')
    axes[idx].set_title(f"Distribution de {f}")
    axes[idx].legend(title='Intrusion', labels=['Normal', 'Intrusion'])

plt.tight_layout()
plt.savefig('02_distributions_features.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 02_distributions_features.png")
plt.close()

# Statistiques descriptives
print()
print("Statistiques descriptives par classe:")
print(df_network.groupby('is_intrusion').describe().transpose())
print()

# ================================================
# 4) Pipeline ML complet avec Random Forest
# ================================================
print("="*80)
print("MODELISATION - RANDOM FOREST BASELINE")
print("="*80)
print()

X = df_network.drop('is_intrusion', axis=1)
y = df_network['is_intrusion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"[*] Split des donnees:")
print(f"   Train: {len(X_train)} echantillons")
print(f"   Test: {len(X_test)} echantillons")
print()

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[OK] Standardisation effectuee")
print()

# Entrainement Random Forest
print("[*] Entrainement du Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
print("[OK] Modele entraine")
print()

# Predictions
y_pred = rf.predict(X_test_scaled)
y_proba = rf.predict_proba(X_test_scaled)[:, 1]

# ================================================
# 5) Evaluation du modele
# ================================================
print("="*80)
print("EVALUATION DU MODELE RANDOM FOREST")
print("="*80)
print()

# Matrice de confusion
print("[*] Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Intrusion'],
            yticklabels=['Normal', 'Intrusion'])
plt.title("Matrice de Confusion - Random Forest")
plt.ylabel('Verite')
plt.xlabel('Prediction')
plt.savefig('03_confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 03_confusion_matrix_rf.png")
plt.close()

# Rapport de classification
print()
print("[*] Rapport de classification:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Intrusion']))

# AUC-ROC
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC = {auc:.4f}")
print()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC={auc:.3f})', linewidth=2)
plt.plot([0,1], [0,1], '--', color='gray', label='Aleatoire')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Courbe ROC - Random Forest")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('04_roc_curve_rf.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 04_roc_curve_rf.png")
plt.close()

# ================================================
# 6) Importance des features
# ================================================
print()
print("="*80)
print("IMPORTANCE DES FEATURES")
print("="*80)
print()

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Importance des features (Random Forest):")
print(feat_imp)
print()

plt.figure(figsize=(8,4))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Importance des Features - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig('05_feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 05_feature_importance_rf.png")
plt.close()

# ================================================
# 7) CHALLENGE 1 - GridSearchCV (Optimisation)
# ================================================
print()
print("="*80)
print("CHALLENGE 1 - OPTIMISATION HYPERPARAMETRES (GridSearchCV)")
print("="*80)
print()

print("[*] Recherche des meilleurs hyperparametres...")
print("   Parametres testes:")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
for k, v in param_grid.items():
    print(f"   - {k}: {v}")
print()

gs = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=0
)

gs.fit(X_train_scaled, y_train)

print("[OK] GridSearchCV termine")
print()
print("Meilleurs hyperparametres trouves:")
for k, v in gs.best_params_.items():
    print(f"   {k}: {v}")
print(f"\nMeilleur score F1 (CV): {gs.best_score_:.4f}")
print()

# Evaluation du modele optimise
y_pred_gs = gs.predict(X_test_scaled)
y_proba_gs = gs.predict_proba(X_test_scaled)[:, 1]
auc_gs = roc_auc_score(y_test, y_proba_gs)

print("Performance sur le test set:")
print(f"   AUC-ROC: {auc_gs:.4f}")
print()
print(classification_report(y_test, y_pred_gs, target_names=['Normal', 'Intrusion']))

# ================================================
# 8) CHALLENGE 2 - Comparaison SVM & Logistic Regression
# ================================================
print("="*80)
print("CHALLENGE 2 - COMPARAISON DES MODELES")
print("="*80)
print()

print("[*] Entrainement Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
y_proba_lr = lr.predict_proba(X_test_scaled)[:,1]
auc_lr = roc_auc_score(y_test, y_proba_lr)
print(f"[OK] AUC-ROC Logistic Regression: {auc_lr:.4f}")

print()
print("[*] Entrainement SVM...")
svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train_scaled, y_train)
y_proba_svm = svm.predict_proba(X_test_scaled)[:,1]
auc_svm = roc_auc_score(y_test, y_proba_svm)
print(f"[OK] AUC-ROC SVM: {auc_svm:.4f}")
print()

# Comparaison visuelle
print("[*] Generation de la comparaison visuelle...")
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc:.3f})', linewidth=2)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.3f})', linewidth=2)
plt.plot([0,1], [0,1], '--', color='gray', label='Aleatoire')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Comparaison des Courbes ROC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('06_comparison_roc_curves.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 06_comparison_roc_curves.png")
plt.close()

print()
print("Tableau comparatif:")
comparison_df = pd.DataFrame({
    'Modele': ['Random Forest', 'Logistic Regression', 'SVM'],
    'AUC-ROC': [auc, auc_lr, auc_svm]
}).sort_values('AUC-ROC', ascending=False)
print(comparison_df.to_string(index=False))
print()

# ================================================
# 9) CHALLENGE 3 - SMOTE (Oversampling)
# ================================================
print("="*80)
print("CHALLENGE 3 - GESTION DU DESEQUILIBRE DES CLASSES (SMOTE)")
print("="*80)
print()

print("[*] Application de SMOTE (Synthetic Minority Over-sampling)...")
print(f"   Avant SMOTE: {y_train.value_counts().to_dict()}")

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

print(f"   Apres SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
print()

print("[*] Entrainement Random Forest avec donnees equilibrees...")
rf_sm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sm.fit(X_res, y_res)

pred_sm = rf_sm.predict(X_test_scaled)
proba_sm = rf_sm.predict_proba(X_test_scaled)[:, 1]
auc_sm = roc_auc_score(y_test, proba_sm)

print("[OK] Modele entraine avec SMOTE")
print()

print("Rapport de classification apres SMOTE:")
print(classification_report(y_test, pred_sm, target_names=['Normal', 'Intrusion']))
print(f"AUC-ROC apres SMOTE: {auc_sm:.4f}")
print()

# Comparaison avant/apres SMOTE
print("Comparaison Random Forest - Avec/Sans SMOTE:")
print(f"   Sans SMOTE - AUC: {auc:.4f}")
print(f"   Avec SMOTE - AUC: {auc_sm:.4f}")
print()

# Matrice de confusion apres SMOTE
cm_sm = confusion_matrix(y_test, pred_sm)
plt.figure(figsize=(6,4))
sns.heatmap(cm_sm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Intrusion'],
            yticklabels=['Normal', 'Intrusion'])
plt.title("Matrice de Confusion - Random Forest avec SMOTE")
plt.ylabel('Verite')
plt.xlabel('Prediction')
plt.savefig('07_confusion_matrix_smote.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: 07_confusion_matrix_smote.png")
plt.close()

# ================================================
# 10) Resume final
# ================================================
print()
print("="*80)
print("RESUME FINAL - TOUS LES RESULTATS")
print("="*80)
print()

summary = pd.DataFrame({
    'Approche': [
        'Random Forest (baseline)',
        'Random Forest (GridSearch)',
        'Logistic Regression',
        'SVM',
        'Random Forest + SMOTE'
    ],
    'AUC-ROC': [auc, auc_gs, auc_lr, auc_svm, auc_sm]
}).sort_values('AUC-ROC', ascending=False)

print(summary.to_string(index=False))
print()

# Sauvegarde du resume
summary.to_csv('resume_resultats.csv', index=False)
print("[OK] Resume sauvegarde: resume_resultats.csv")
print()

print("="*80)
print("GRAPHIQUES GENERES:")
print("="*80)
print("  1. 01_distribution_classes.png")
print("  2. 02_distributions_features.png")
print("  3. 03_confusion_matrix_rf.png")
print("  4. 04_roc_curve_rf.png")
print("  5. 05_feature_importance_rf.png")
print("  6. 06_comparison_roc_curves.png")
print("  7. 07_confusion_matrix_smote.png")
print()

print("="*80)
print("[OK] Script execute avec succes !")
print("="*80)
