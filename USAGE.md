# 📖 Guide d'Utilisation - Système de Détection d'Intrusion ML

Ce guide détaille l'utilisation de chaque script du projet et explique comment interpréter les résultats.

## 📋 Table des matières

- [Installation](#installation)
- [Scripts disponibles](#scripts-disponibles)
- [Utilisation détaillée](#utilisation-détaillée)
- [Interprétation des résultats](#interprétation-des-résultats)
- [Personnalisation](#personnalisation)
- [FAQ](#faq)

## 🚀 Installation

### Étape 1 : Cloner le repository

```bash
git clone https://github.com/[VOTRE_USERNAME]/ml-intrusion-detection.git
cd ml-intrusion-detection
```

### Étape 2 : Créer un environnement virtuel

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 : Installer les dépendances

```bash
pip install -r requirements.txt
```

### Vérification

```bash
python -c "import sklearn, xgboost, lightgbm; print('✓ Installation réussie!')"
```

## 📂 Scripts disponibles

| Script | Niveau | Temps | Description |
|--------|--------|-------|-------------|
| `Correction_atelier_intrusion_ml.py` | Débutant | ~10s | Baseline avec Random Forest |
| `pipeline_ml_complet.py` | Intermédiaire | ~30s | Pipeline complet avec SMOTE + GridSearch |
| `optimisation_threshold_cyber.py` | Avancé | ~15s | Optimisation threshold pour cyberdéfense |
| `comparaison_modeles_intrusion.py` | Avancé | ~45s | Benchmark de 10 algorithmes |

## 💻 Utilisation détaillée

### 1. Script Baseline (Recommandé pour débuter)

**Objectif :** Comprendre les bases de la détection d'intrusion par ML

```bash
python Correction_atelier_intrusion_ml.py
```

**Ce que fait ce script :**

1. Génère 1000 connexions réseau synthétiques
2. Entraîne un Random Forest avec class_weight='balanced'
3. Évalue les performances (Recall, Precision, F1-Score)
4. Génère 4 visualisations

**Fichiers générés :**

```
distributions_features.png    → Distribution des features par classe
correlation_matrix.png         → Corrélation entre features
roc_curve.png                  → Courbe ROC
feature_importance.png         → Importance des variables
```

**Résultats attendus :**

```
Recall (Intrusion): ~90-92%
Precision: ~96-97%
AUC-ROC: ~0.998
```

---

### 2. Pipeline Complet (Pour aller plus loin)

**Objectif :** Explorer les techniques avancées (SMOTE, GridSearch, comparaisons)

```bash
python pipeline_ml_complet.py
```

**Ce que fait ce script :**

1. Exploration des données (distributions, statistiques)
2. Baseline Random Forest
3. **GridSearchCV** pour optimiser les hyperparamètres
4. Comparaison **Random Forest vs SVM vs Logistic Regression**
5. Application de **SMOTE** (rééquilibrage des classes)

**Fichiers générés (7 graphiques) :**

```
01_distribution_classes.png            → Distribution Normal/Intrusion
02_distributions_features.png          → Histogrammes par feature
03_confusion_matrix_rf.png             → Matrice de confusion RF baseline
04_roc_curve_rf.png                    → Courbe ROC RF
05_feature_importance_rf.png           → Importance des features
06_comparison_roc_curves.png           → Comparaison des 3 modèles
07_confusion_matrix_smote.png          → Résultat avec SMOTE
resume_resultats.csv                   → Tableau récapitulatif
```

**Résultats attendus :**

| Modèle | AUC-ROC |
|--------|---------|
| Random Forest | ~1.000 |
| SVM | ~0.992 |
| Logistic Regression | ~0.926 |

---

### 3. Optimisation Threshold (Focus cyberdéfense) ⭐

**Objectif :** Maximiser le Recall pour minimiser les intrusions manquées

```bash
python optimisation_threshold_cyber.py
```

**Ce que fait ce script :**

1. Teste différents thresholds (0.05 à 0.90)
2. Analyse l'impact sur Recall, Precision, F2-Score
3. Identifie le threshold optimal pour Recall ≥ 99%
4. Compare performances avant/après optimisation

**Fichiers générés :**

```
threshold_analysis.png                 → Recall/Precision vs Threshold
faux_negatifs_analysis.png             → Intrusions manquées vs Threshold
confusion_matrices_comparison.png      → Avant/Après optimisation
roc_curve_with_thresholds.png          → ROC avec thresholds marqués
threshold_optimization_results.csv     → Résultats détaillés
```

**Résultats clés :**

| Configuration | Threshold | Recall | Intrusions manquées |
|---------------|-----------|--------|---------------------|
| Défaut | 0.5 | 90.62% | 3/32 |
| **Optimisé** | **0.15** | **96.88%** | **1/32** |
| Max sécurité | 0.01 | 100% | 0/32 |

---

### 4. Comparaison 10 Modèles (Benchmark complet)

**Objectif :** Identifier le meilleur algorithme pour votre cas d'usage

```bash
cd Comparaison_Modeles
python comparaison_modeles_intrusion.py
```

**Ce que fait ce script :**

1. Entraîne et évalue **10 algorithmes** :
   - Random Forest
   - XGBoost ⭐
   - LightGBM
   - Gradient Boosting
   - SVM
   - Logistic Regression
   - KNN
   - Naive Bayes
   - Decision Tree
   - AdaBoost

2. Compare les performances (AUC-ROC, F1-Score, temps)
3. Génère des recommandations pour la production

**Fichiers générés :**

```
comparaison_metriques.png      → Accuracy, Precision, Recall, F1
comparaison_auc.png            → AUC-ROC de chaque modèle
comparaison_temps.png          → Performance vs Vitesse
resultats_comparaison.csv      → Tableau complet
```

**Top 3 des modèles :**

| 🏆 Rang | Modèle | AUC-ROC | F1-Score | Temps |
|---------|--------|---------|----------|-------|
| 🥇 | **XGBoost** | **0.9994** | 0.9688 | 0.062s |
| 🥈 | LightGBM | 0.9956 | 0.9688 | 0.141s |
| 🥉 | Gradient Boosting | 0.9947 | 0.9688 | 0.219s |

---

## 📊 Interprétation des résultats

### Métriques clés en cyberdéfense

#### 1. **Recall (Sensibilité)** - LA PLUS IMPORTANTE ⭐⭐⭐

```
Recall = Intrusions détectées / Intrusions totales
```

**Objectif :** ≥ 99%

- ✅ 99% = Seulement 1% d'intrusions manquées
- ⚠️ 90% = 10% d'intrusions passent inaperçues (DANGEREUX)

**Pourquoi c'est crucial ?**
- Une seule intrusion manquée = système compromis
- Ransomware, vol de données, backdoor...

#### 2. **Precision** - Importante mais secondaire

```
Precision = Vraies intrusions / Total alertes
```

**Objectif :** ≥ 70%

- ✅ 90% = 9 alertes sur 10 sont vraies
- ⚠️ 50% = Moitié des alertes sont fausses (fatigue SOC)

**Trade-off :**
- Mieux vaut 100 fausses alertes qu'une vraie intrusion manquée

#### 3. **F2-Score** - Métrique optimale pour cyber

```
F2-Score = Moyenne harmonique avec 2x plus de poids au Recall
```

**Objectif :** ≥ 0.95

- Privilégie le Recall tout en tenant compte de la Precision
- Adapté aux contextes où les faux négatifs sont critiques

#### 4. **AUC-ROC** - Performance globale

```
Area Under the Curve (courbe ROC)
```

**Objectif :** ≥ 0.95

- 1.0 = Discrimination parfaite
- 0.5 = Modèle aléatoire

---

### Matrice de confusion expliquée

```
                    Prédit Normal    Prédit Intrusion
Vrai Normal              TN                FP
Vrai Intrusion           FN                TP
```

| Terme | Signification | Impact cyberdéfense |
|-------|---------------|---------------------|
| **TP** (Vrai Positif) | Intrusion correctement détectée | ✅ Excellent |
| **TN** (Vrai Négatif) | Trafic normal correctement identifié | ✅ Bon |
| **FP** (Faux Positif) | Fausse alerte (trafic normal signalé) | ⚠️ Acceptable |
| **FN** (Faux Négatif) | Intrusion MANQUÉE | ❌ CRITIQUE |

**Objectif cyberdéfense : FN = 0** (aucune intrusion manquée)

---

### Exemple de rapport

```
================================================================================
RESULTATS AVEC THRESHOLD OPTIMISE (0.15)
================================================================================

Matrice de confusion:
                Pred Normal   Pred Intrusion
Vrai Normal          168              1
Vrai Intrusion         1             31

Recall (Intrusion):    96.88%   ✅
Precision (Intrusion): 96.88%   ✅
F1-Score:              0.9688   ✅
F2-Score:              0.9688   ✅

[OK] Intrusions manquees: 1/32 (3.1%)  ✅
```

**Interprétation :**
- ✅ **Excellent Recall** : Seulement 1 intrusion manquée sur 32
- ✅ **Excellente Precision** : 31/32 alertes sont vraies
- ✅ **Équilibre optimal** pour la cyberdéfense

---

## ⚙️ Personnalisation

### Modifier la taille du dataset

Dans les scripts, changez :

```python
n_samples = 1000  # Passer à 5000 ou 10000
```

### Utiliser vos propres données

Remplacez la fonction `generate_dataset()` :

```python
def load_your_data():
    df = pd.read_csv('your_data.csv')
    # Adapter les noms de colonnes
    return df
```

### Ajuster les seuils de détection

Dans la génération de labels :

```python
intrusion_mask = (
    (df['packet_size'] > 800) |      # Modifier ce seuil
    (df['duration'] > 5) |            # Modifier ce seuil
    (df['num_failed_logins'] > 2)    # Modifier ce seuil
)
```

### Tester d'autres hyperparamètres

GridSearchCV permet de tester facilement :

```python
param_grid = {
    'n_estimators': [50, 100, 200],      # Ajouter des valeurs
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]      # Ajouter des valeurs
}
```

---

## ❓ FAQ

### Q1 : Pourquoi Recall 90% n'est pas suffisant ?

**R :** En cyberdéfense, 10% d'intrusions manquées = 10% de risque de compromission totale. Sur 1000 attaques, 100 passeraient inaperçues.

### Q2 : Comment choisir entre XGBoost et Random Forest ?

**R :**
- **XGBoost** : Meilleur AUC, plus rapide → **Production**
- **Random Forest** : Plus simple, interprétable → **Prototypage**

### Q3 : SMOTE améliore-t-il toujours les performances ?

**R :** Non. Si le modèle baseline performe déjà bien (comme ici avec 99% AUC), SMOTE n'apporte pas d'amélioration. Utile surtout si Recall initial < 85%.

### Q4 : Quel threshold utiliser en production ?

**R :** Dépend de votre tolérance :
- **0.15** : Équilibre optimal (Recall 96.88%, peu de FP)
- **0.05** : Sécurité maximale (Recall 100%, plus de FP)
- **0.50** : Défaut (Recall 90%, pas recommandé)

### Q5 : Combien de temps pour entraîner sur 1M de lignes ?

**R :**
- XGBoost : ~5-10 minutes
- Random Forest : ~10-20 minutes
- LightGBM : ~3-5 minutes (le plus rapide)

### Q6 : Peut-on déployer ce modèle en production ?

**R :** Oui, mais :
1. Entraîner sur données réelles (NSL-KDD, CICIDS2017)
2. Monitorer les performances en continu
3. Réentraîner régulièrement (nouvelles menaces)
4. Mettre en place une boucle de feedback (SOC)

---

## 📚 Ressources complémentaires

- [README principal](README.md)
- [Méthodologie détaillée](docs/methodologie.md)
- [Guide des métriques en cyberdéfense](docs/metriques_cyber.md)

---

**Besoin d'aide ?** Ouvrez une issue sur GitHub ou contactez-moi : syoungoua0@gmail.com

---

*Dernière mise à jour : Janvier 2025*
