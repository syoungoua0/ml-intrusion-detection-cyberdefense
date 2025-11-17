# comparaison_modeles_intrusion.py
# Comparaison de plusieurs algorithmes ML pour la détection d'intrusion réseau

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost (si disponible)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[AVERTISSEMENT] XGBoost n'est pas installe. Installation recommandee: pip install xgboost\n")

# Import LightGBM (si disponible)
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[AVERTISSEMENT] LightGBM n'est pas installe. Installation recommandee: pip install lightgbm\n")


def generate_dataset(n_samples=1000, random_state=42):
    """Génère un dataset synthétique pour la détection d'intrusion"""
    np.random.seed(random_state)

    data = {
        'packet_size': np.random.normal(500, 200, n_samples),
        'duration': np.random.exponential(2, n_samples),
        'src_bytes': np.random.lognormal(8, 2, n_samples),
        'dst_bytes': np.random.lognormal(7, 2, n_samples),
        'num_failed_logins': np.random.poisson(0.1, n_samples),
        'protocol_type': np.random.choice([0, 1, 2], n_samples)
    }

    df = pd.DataFrame(data)

    # Règles pour créer des labels d'intrusion
    intrusion_mask = (
        (df['packet_size'] > 800) |
        (df['duration'] > 5) |
        (df['num_failed_logins'] > 2) |
        ((df['src_bytes'] > np.percentile(df['src_bytes'], 90)) &
         (df['dst_bytes'] > np.percentile(df['dst_bytes'], 90)))
    )

    df['is_intrusion'] = intrusion_mask.astype(int)

    return df


def get_models():
    """
    Retourne un dictionnaire de modèles à comparer avec leurs explications.

    EXPLICATIONS DES MODÈLES:

    1. RANDOM FOREST (Baseline)
       - Ensemble de nombreux arbres de décision
       - Chaque arbre vote, et la majorité l'emporte
       - Robuste, peu sensible au surapprentissage
       - Bon pour ce problème: gère bien les relations non-linéaires

    2. XGBOOST (eXtreme Gradient Boosting) ⭐ RECOMMANDÉ
       - Boosting: construit des arbres séquentiellement
       - Chaque arbre corrige les erreurs du précédent
       - Très performant en pratique, gagnant de nombreux Kaggle
       - Excellent pour la détection d'intrusion: rapide et précis
       - Gère bien les données déséquilibrées

    3. LIGHTGBM (Light Gradient Boosting Machine)
       - Similaire à XGBoost mais plus rapide sur gros datasets
       - Construit les arbres différemment (leaf-wise vs level-wise)
       - Très efficace en mémoire
       - Idéal pour des millions de lignes

    4. GRADIENT BOOSTING (sklearn)
       - Version de base du boosting par sklearn
       - Moins optimisé que XGBoost/LightGBM mais stable
       - Bon pour comprendre le principe du boosting

    5. LOGISTIC REGRESSION
       - Modèle linéaire simple et interprétable
       - Rapide à entraîner
       - Bon baseline, mais limité pour relations complexes

    6. SVM (Support Vector Machine)
       - Trouve l'hyperplan optimal séparant les classes
       - Kernel RBF permet de capturer la non-linéarité
       - Peut être lent sur gros datasets
       - Efficace si les données sont bien séparables

    7. K-NEAREST NEIGHBORS (KNN)
       - Classifie selon les K voisins les plus proches
       - Simple conceptuellement
       - Peut être lent en prédiction
       - Sensible à l'échelle des features (d'où la standardisation)

    8. NAIVE BAYES
       - Basé sur le théorème de Bayes
       - Assume l'indépendance des features (souvent faux, mais marche bien)
       - Très rapide
       - Bon pour données avec beaucoup de features

    9. DECISION TREE
       - Un seul arbre de décision
       - Très interprétable
       - Tend à sur-apprendre (Random Forest résout ce problème)

    10. ADABOOST
        - Adaptive Boosting: donne plus de poids aux exemples mal classés
        - Plus ancien que XGBoost
        - Sensible au bruit et outliers
    """

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
            algorithm='SAMME'
        )
    }

    # Ajout de XGBoost si disponible
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=5  # Pour gérer le déséquilibre
        )

    # Ajout de LightGBM si disponible
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Évalue un modèle et retourne ses métriques"""

    # Entraînement et mesure du temps
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Prédictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Probabilités pour AUC
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = None

    # Métriques
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': auc,
        'Training Time (s)': training_time,
        'Prediction Time (s)': prediction_time
    }

    return metrics, y_pred, model


def plot_comparison(results_df):
    """Crée des visualisations comparant les modèles"""

    # 1. Comparaison des métriques principales
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        data = results_df.sort_values(metric, ascending=True)
        bars = ax.barh(data['Model'], data[metric])

        # Coloration: vert pour les meilleures valeurs
        colors = plt.cm.RdYlGn(data[metric] / data[metric].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel(metric)
        ax.set_title(f'Comparaison: {metric}')
        ax.set_xlim([0, 1])

        # Ajouter les valeurs
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')

    plt.tight_layout()
    plt.savefig('comparaison_metriques.png', dpi=300, bbox_inches='tight')
    print("[OK] Graphique sauvegarde: comparaison_metriques.png")
    plt.close()

    # 2. AUC-ROC comparison
    if results_df['AUC-ROC'].notna().any():
        plt.figure(figsize=(12, 6))
        data = results_df[results_df['AUC-ROC'].notna()].sort_values('AUC-ROC', ascending=True)
        bars = plt.barh(data['Model'], data['AUC-ROC'])
        colors = plt.cm.RdYlGn(data['AUC-ROC'] / data['AUC-ROC'].max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('AUC-ROC Score')
        plt.title('Comparaison AUC-ROC (Area Under Curve)')
        plt.xlim([0.5, 1.0])

        for i, v in enumerate(data['AUC-ROC']):
            plt.text(v + 0.005, i, f'{v:.4f}', va='center')

        plt.tight_layout()
        plt.savefig('comparaison_auc.png', dpi=300, bbox_inches='tight')
        print("[OK] Graphique sauvegarde: comparaison_auc.png")
        plt.close()

    # 3. Temps d'entrainement vs Performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training Time
    data = results_df.sort_values('Training Time (s)', ascending=True)
    ax1.barh(data['Model'], data['Training Time (s)'], color='steelblue')
    ax1.set_xlabel('Temps (secondes)')
    ax1.set_title("Temps d'Entrainement")
    for i, v in enumerate(data['Training Time (s)']):
        ax1.text(v + 0.01, i, f'{v:.3f}s', va='center')

    # F1-Score vs Training Time (scatter)
    ax2.scatter(results_df['Training Time (s)'], results_df['F1-Score'],
                s=200, alpha=0.6, c=results_df['F1-Score'], cmap='RdYlGn')
    for idx, row in results_df.iterrows():
        ax2.annotate(row['Model'], (row['Training Time (s)'], row['F1-Score']),
                    fontsize=9, ha='right')
    ax2.set_xlabel('Temps d\'Entrainement (s)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Performance vs Vitesse')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparaison_temps.png', dpi=300, bbox_inches='tight')
    print("[OK] Graphique sauvegarde: comparaison_temps.png")
    plt.close()


def main():
    print("="*80)
    print("COMPARAISON DE MODELES ML POUR LA DETECTION D'INTRUSION")
    print("="*80)
    print()

    # 1. Generation des donnees
    print("[*] Generation du dataset...")
    df = generate_dataset(n_samples=1000, random_state=42)
    print(f"   Dataset: {len(df)} echantillons")
    print(f"   Classes: {df['is_intrusion'].value_counts().to_dict()}")
    print()

    # 2. Preparation des donnees
    X = df.drop('is_intrusion', axis=1)
    y = df['is_intrusion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Recuperation des modeles
    models = get_models()
    print(f"[*] {len(models)} modeles a evaluer:")
    for name in models.keys():
        print(f"   - {name}")
    print()

    # 4. Evaluation de tous les modeles
    print("[*] Entrainement et evaluation en cours...\n")
    results = []

    for name, model in models.items():
        print(f"   Entrainement: {name}...", end=' ')
        try:
            metrics, y_pred, trained_model = evaluate_model(
                model, X_train_scaled, X_test_scaled, y_train, y_test, name
            )
            results.append(metrics)
            print("[OK]")
        except Exception as e:
            print(f"[ERREUR: {e}]")

    print()

    # 5. Creation du DataFrame de resultats
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)

    # 6. Affichage des resultats
    print("="*80)
    print("RESULTATS DE LA COMPARAISON")
    print("="*80)
    print()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.to_string(index=False))
    print()

    # 7. Meilleur modele
    best_model = results_df.iloc[0]
    print("="*80)
    print("*** MEILLEUR MODELE ***")
    print("="*80)
    print(f"Modele: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.4f}")
    print(f"Precision: {best_model['Precision']:.4f}")
    print(f"Recall: {best_model['Recall']:.4f}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    if pd.notna(best_model['AUC-ROC']):
        print(f"AUC-ROC: {best_model['AUC-ROC']:.4f}")
    print(f"Temps d'entrainement: {best_model['Training Time (s)']:.4f}s")
    print()

    # 8. Visualisations
    print("[*] Generation des graphiques comparatifs...")
    plot_comparison(results_df)
    print()

    # 9. Sauvegarde des resultats
    results_df.to_csv('resultats_comparaison.csv', index=False)
    print("[OK] Resultats sauvegardes: resultats_comparaison.csv")
    print()

    # 10. Recommandations
    print("="*80)
    print("*** RECOMMANDATIONS ***")
    print("="*80)
    print()

    if XGBOOST_AVAILABLE and 'XGBoost' in results_df['Model'].values:
        xgb_row = results_df[results_df['Model'] == 'XGBoost'].iloc[0]
        print("[+] XGBoost est generalement le meilleur choix pour la detection d'intrusion:")
        print(f"  - Excellent equilibre performance/vitesse")
        print(f"  - F1-Score: {xgb_row['F1-Score']:.4f}")
        print(f"  - Temps: {xgb_row['Training Time (s)']:.3f}s")
        print()

    print("Pour la PRODUCTION:")
    print("  1. Si performance maximale requise: XGBoost ou LightGBM")
    print("  2. Si interpretabilite importante: Random Forest ou Decision Tree")
    print("  3. Si ressources limitees: Logistic Regression ou Naive Bayes")
    print("  4. Si donnees en temps reel: modeles legers (LogReg, NB)")
    print()

    print("Pour l'APPRENTISSAGE:")
    print("  - Commencer avec Decision Tree (le plus simple a comprendre)")
    print("  - Puis Random Forest (amelioration par ensemble)")
    print("  - Ensuite XGBoost (boosting sequentiel)")
    print()

    print("="*80)
    print("[OK] Analyse terminee!")
    print("="*80)


if __name__ == "__main__":
    main()
