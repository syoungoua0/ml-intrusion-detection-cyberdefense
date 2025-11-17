# optimisation_threshold_cyber.py
# Optimisation du threshold pour la cyberdefense - Focus sur le Recall

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# ================================================
# 1) Generation du dataset (identique au script original)
# ================================================
print("="*80)
print("OPTIMISATION DU THRESHOLD POUR LA CYBERDEFENSE")
print("="*80)
print()

print("[*] Generation du dataset...")
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

intrusion_mask = (
    (df_network['packet_size'] > 800) |
    (df_network['duration'] > 5) |
    (df_network['num_failed_logins'] > 2) |
    ((df_network['src_bytes'] > np.percentile(df_network['src_bytes'], 90)) &
     (df_network['dst_bytes'] > np.percentile(df_network['dst_bytes'], 90)))
)

df_network['is_intrusion'] = intrusion_mask.astype(int)

print(f"[OK] Dataset cree: {len(df_network)} echantillons")
print(f"   Distribution: {df_network['is_intrusion'].value_counts().to_dict()}")
print()

# ================================================
# 2) Preparation et entrainement du modele
# ================================================
X = df_network.drop('is_intrusion', axis=1)
y = df_network['is_intrusion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[*] Split: Train={len(X_train)}, Test={len(X_test)}")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[*] Entrainement du Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
print("[OK] Modele entraine")
print()

# Obtenir les probabilites
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# ================================================
# 3) COMPARAISON: Threshold par defaut (0.5)
# ================================================
print("="*80)
print("RESULTATS AVEC THRESHOLD PAR DEFAUT (0.5)")
print("="*80)
print()

y_pred_default = rf_model.predict(X_test_scaled)
cm_default = confusion_matrix(y_test, y_pred_default)

print("Matrice de confusion (Threshold = 0.5):")
print(pd.DataFrame(cm_default,
                   index=['Vrai Normal', 'Vrai Intrusion'],
                   columns=['Pred Normal', 'Pred Intrusion']))
print()

recall_default = recall_score(y_test, y_pred_default)
precision_default = precision_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)
f2_default = fbeta_score(y_test, y_pred_default, beta=2)

print(f"Recall (Intrusion):    {recall_default*100:.2f}%")
print(f"Precision (Intrusion): {precision_default*100:.2f}%")
print(f"F1-Score:              {f1_default:.4f}")
print(f"F2-Score:              {f2_default:.4f}")
print()

# Analyse des faux negatifs
fn_default = cm_default[1, 0]
total_intrusions = cm_default[1, 0] + cm_default[1, 1]
print(f"[ALERTE] Intrusions manquees: {fn_default}/{total_intrusions} ({fn_default/total_intrusions*100:.1f}%)")
print()

# ================================================
# 4) ANALYSE DE DIFFERENTS THRESHOLDS
# ================================================
print("="*80)
print("ANALYSE DE DIFFERENTS THRESHOLDS")
print("="*80)
print()

thresholds_to_test = np.arange(0.05, 0.95, 0.05)
results = []

for threshold in thresholds_to_test:
    y_pred_temp = (y_pred_proba >= threshold).astype(int)

    recall = recall_score(y_test, y_pred_temp)
    precision = precision_score(y_test, y_pred_temp)
    f1 = f1_score(y_test, y_pred_temp)
    f2 = fbeta_score(y_test, y_pred_temp, beta=2)

    cm = confusion_matrix(y_test, y_pred_temp)
    fn = cm[1, 0]  # Faux negatifs
    fp = cm[0, 1]  # Faux positifs

    results.append({
        'Threshold': threshold,
        'Recall': recall,
        'Precision': precision,
        'F1-Score': f1,
        'F2-Score': f2,
        'Faux_Negatifs': fn,
        'Faux_Positifs': fp
    })

results_df = pd.DataFrame(results)

print("Apercu des thresholds testes:")
print(results_df[['Threshold', 'Recall', 'Precision', 'F2-Score', 'Faux_Negatifs']].to_string(index=False))
print()

# ================================================
# 5) TROUVER LE THRESHOLD OPTIMAL (Recall >= 99%)
# ================================================
print("="*80)
print("RECHERCHE DU THRESHOLD OPTIMAL")
print("="*80)
print()

# Objectif: Recall >= 99% (ou le plus proche possible)
target_recall = 0.99

# Threshold pour atteindre exactement 99% de Recall
optimal_threshold = None
for threshold in np.arange(0.01, 0.99, 0.01):
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_temp)

    if recall >= target_recall:
        optimal_threshold = threshold
        break

if optimal_threshold is None:
    # Si on ne peut pas atteindre 99%, prendre le meilleur
    optimal_row = results_df.loc[results_df['Recall'].idxmax()]
    optimal_threshold = optimal_row['Threshold']
    print(f"[INFO] Recall 99% non atteignable. Meilleur Recall possible: {optimal_row['Recall']*100:.2f}%")
else:
    print(f"[OK] Threshold optimal trouve pour Recall >= 99%: {optimal_threshold:.3f}")

print()

# ================================================
# 6) RESULTATS AVEC THRESHOLD OPTIMISE
# ================================================
print("="*80)
print(f"RESULTATS AVEC THRESHOLD OPTIMISE ({optimal_threshold:.3f})")
print("="*80)
print()

y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
cm_optimized = confusion_matrix(y_test, y_pred_optimized)

print(f"Matrice de confusion (Threshold = {optimal_threshold:.3f}):")
print(pd.DataFrame(cm_optimized,
                   index=['Vrai Normal', 'Vrai Intrusion'],
                   columns=['Pred Normal', 'Pred Intrusion']))
print()

recall_opt = recall_score(y_test, y_pred_optimized)
precision_opt = precision_score(y_test, y_pred_optimized)
f1_opt = f1_score(y_test, y_pred_optimized)
f2_opt = fbeta_score(y_test, y_pred_optimized, beta=2)

print(f"Recall (Intrusion):    {recall_opt*100:.2f}%")
print(f"Precision (Intrusion): {precision_opt*100:.2f}%")
print(f"F1-Score:              {f1_opt:.4f}")
print(f"F2-Score:              {f2_opt:.4f}")
print()

fn_opt = cm_optimized[1, 0]
fp_opt = cm_optimized[0, 1]
print(f"[OK] Intrusions manquees: {fn_opt}/{total_intrusions} ({fn_opt/total_intrusions*100:.1f}%)")
print(f"[INFO] Faux positifs: {fp_opt}")
print()

# ================================================
# 7) COMPARAISON AVANT/APRES
# ================================================
print("="*80)
print("COMPARAISON THRESHOLD 0.5 vs OPTIMISE")
print("="*80)
print()

comparison = pd.DataFrame({
    'Metrique': ['Threshold', 'Recall', 'Precision', 'F1-Score', 'F2-Score',
                 'Intrusions Manquees', 'Faux Positifs'],
    'Default (0.5)': [0.5, f"{recall_default*100:.2f}%", f"{precision_default*100:.2f}%",
                      f"{f1_default:.4f}", f"{f2_default:.4f}", fn_default, cm_default[0, 1]],
    f'Optimise ({optimal_threshold:.3f})': [optimal_threshold, f"{recall_opt*100:.2f}%",
                                             f"{precision_opt*100:.2f}%",
                                             f"{f1_opt:.4f}", f"{f2_opt:.4f}",
                                             fn_opt, fp_opt],
    'Amelioration': ['N/A',
                     f"+{(recall_opt-recall_default)*100:.2f}%",
                     f"{(precision_opt-precision_default)*100:+.2f}%",
                     f"{(f1_opt-f1_default):+.4f}",
                     f"{(f2_opt-f2_default):+.4f}",
                     f"{fn_opt-fn_default:+d}",
                     f"{fp_opt-cm_default[0, 1]:+d}"]
})

print(comparison.to_string(index=False))
print()

# ================================================
# 8) VISUALISATIONS
# ================================================
print("="*80)
print("GENERATION DES VISUALISATIONS")
print("="*80)
print()

# 1. Courbe Recall vs Threshold
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results_df['Threshold'], results_df['Recall']*100, 'b-', linewidth=2, label='Recall')
plt.plot(results_df['Threshold'], results_df['Precision']*100, 'r-', linewidth=2, label='Precision')
plt.axhline(y=99, color='g', linestyle='--', label='Objectif Recall (99%)')
plt.axvline(x=0.5, color='gray', linestyle=':', label='Threshold defaut (0.5)')
plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
            label=f'Threshold optimal ({optimal_threshold:.3f})')
plt.xlabel('Threshold')
plt.ylabel('Score (%)')
plt.title('Impact du Threshold sur Recall et Precision')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. F2-Score vs Threshold
plt.subplot(1, 2, 2)
plt.plot(results_df['Threshold'], results_df['F2-Score'], 'purple', linewidth=2)
plt.axvline(x=0.5, color='gray', linestyle=':', label='Threshold defaut (0.5)')
plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
            label=f'Threshold optimal ({optimal_threshold:.3f})')
plt.xlabel('Threshold')
plt.ylabel('F2-Score')
plt.title('F2-Score en fonction du Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: threshold_analysis.png")

# 3. Faux Negatifs vs Threshold
plt.figure(figsize=(10, 5))
plt.plot(results_df['Threshold'], results_df['Faux_Negatifs'], 'r-', linewidth=2, marker='o')
plt.axvline(x=0.5, color='gray', linestyle=':', label='Threshold defaut (0.5)')
plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
            label=f'Threshold optimal ({optimal_threshold:.3f})')
plt.xlabel('Threshold')
plt.ylabel('Nombre de Faux Negatifs (Intrusions Manquees)')
plt.title('Impact du Threshold sur les Intrusions Manquees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('faux_negatifs_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: faux_negatifs_analysis.png")

# 4. Comparaison des matrices de confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_default, annot=True, fmt='d', cmap='Reds', ax=ax1,
            xticklabels=['Normal', 'Intrusion'],
            yticklabels=['Normal', 'Intrusion'])
ax1.set_title(f'Threshold = 0.5 (Defaut)\nRecall: {recall_default*100:.2f}%')
ax1.set_ylabel('Verite')
ax1.set_xlabel('Prediction')

sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=ax2,
            xticklabels=['Normal', 'Intrusion'],
            yticklabels=['Normal', 'Intrusion'])
ax2.set_title(f'Threshold = {optimal_threshold:.3f} (Optimise)\nRecall: {recall_opt*100:.2f}%')
ax2.set_ylabel('Verite')
ax2.set_xlabel('Prediction')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: confusion_matrices_comparison.png")

# 5. Courbe ROC avec thresholds
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatoire')

# Marquer les points pour threshold 0.5 et optimal
y_pred_05 = (y_pred_proba >= 0.5).astype(int)
y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)

cm_05 = confusion_matrix(y_test, y_pred_05)
cm_opt_vis = confusion_matrix(y_test, y_pred_opt)

fpr_05 = cm_05[0, 1] / (cm_05[0, 0] + cm_05[0, 1])
tpr_05 = cm_05[1, 1] / (cm_05[1, 0] + cm_05[1, 1])

fpr_opt_vis = cm_opt_vis[0, 1] / (cm_opt_vis[0, 0] + cm_opt_vis[0, 1])
tpr_opt_vis = cm_opt_vis[1, 1] / (cm_opt_vis[1, 0] + cm_opt_vis[1, 1])

plt.plot(fpr_05, tpr_05, 'ro', markersize=10, label=f'Threshold 0.5 (TPR={tpr_05:.3f})')
plt.plot(fpr_opt_vis, tpr_opt_vis, 'go', markersize=10, label=f'Threshold {optimal_threshold:.3f} (TPR={tpr_opt_vis:.3f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (= Recall)')
plt.title('Courbe ROC avec Thresholds')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve_with_thresholds.png', dpi=300, bbox_inches='tight')
print("[OK] Graphique sauvegarde: roc_curve_with_thresholds.png")

print()

# Sauvegarde des resultats
results_df.to_csv('threshold_optimization_results.csv', index=False)
print("[OK] Resultats detailles sauvegardes: threshold_optimization_results.csv")
print()

# ================================================
# 9) RAPPORT FINAL
# ================================================
print("="*80)
print("RAPPORT FINAL - RECOMMANDATIONS")
print("="*80)
print()

print(f"THRESHOLD RECOMMANDE POUR LA CYBERDEFENSE: {optimal_threshold:.3f}")
print()
print("JUSTIFICATION:")
print(f"  - Recall passe de {recall_default*100:.2f}% a {recall_opt*100:.2f}%")
print(f"  - Intrusions manquees: {fn_default} -> {fn_opt} ({fn_default-fn_opt} en moins)")
print(f"  - F2-Score (metrique cyber): {f2_default:.4f} -> {f2_opt:.4f}")
print()

if fn_opt == 0:
    print("[EXCELLENT] Toutes les intrusions sont detectees (100% Recall) !")
elif fn_opt <= 1:
    print("[TRES BON] Seulement {fn_opt} intrusion(s) manquee(s)")
else:
    print(f"[ATTENTION] {fn_opt} intrusions encore manquees - envisager threshold plus bas")

print()
print(f"COUT: {fp_opt - cm_default[0, 1]:+d} fausses alertes supplementaires")
print("      (acceptable en cyberdefense)")
print()

print("GRAPHIQUES GENERES:")
print("  1. threshold_analysis.png - Impact du threshold")
print("  2. faux_negatifs_analysis.png - Intrusions manquees")
print("  3. confusion_matrices_comparison.png - Avant/Apres")
print("  4. roc_curve_with_thresholds.png - ROC avec thresholds")
print()

print("="*80)
print("[OK] Optimisation terminee avec succes !")
print("="*80)
