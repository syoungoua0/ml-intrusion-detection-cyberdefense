# 📊 Data Directory

Ce dossier contient les données utilisées pour la détection d'intrusion.

## Données synthétiques

Les scripts génèrent automatiquement des données réseau synthétiques avec les caractéristiques suivantes :

### Features (6 variables)

| Feature | Description | Type | Distribution |
|---------|-------------|------|--------------|
| `packet_size` | Taille des paquets (bytes) | Numérique | Normale (μ=500, σ=200) |
| `duration` | Durée de la connexion (secondes) | Numérique | Exponentielle (λ=2) |
| `src_bytes` | Données envoyées (bytes) | Numérique | Log-normale (μ=8, σ=2) |
| `dst_bytes` | Données reçues (bytes) | Numérique | Log-normale (μ=7, σ=2) |
| `num_failed_logins` | Tentatives de connexion échouées | Entier | Poisson (λ=0.1) |
| `protocol_type` | Type de protocole | Catégorielle | Uniforme {0, 1, 2} |

### Target (variable cible)

| Variable | Description | Valeurs |
|----------|-------------|---------|
| `is_intrusion` | Indicateur d'intrusion | 0 (Normal) / 1 (Intrusion) |

### Règles de détection

Une connexion est classée comme **intrusion** si :

```python
(packet_size > 800) OR
(duration > 5) OR
(num_failed_logins > 2) OR
((src_bytes > P90) AND (dst_bytes > P90))
```

où P90 = 90ème percentile

## Utilisation de données réelles

Pour utiliser vos propres données :

1. Placez votre fichier CSV dans ce dossier
2. Assurez-vous que les colonnes correspondent aux features ci-dessus
3. Modifiez la fonction `generate_dataset()` dans les scripts pour charger votre fichier

Exemple :

```python
def load_custom_data(filepath):
    df = pd.read_csv(filepath)
    # Adapter les noms de colonnes si nécessaire
    return df
```

## Datasets publics recommandés

Pour aller plus loin, vous pouvez utiliser :

- **NSL-KDD** : Version améliorée du dataset KDD Cup 1999
- **CICIDS2017** : Canadian Institute for Cybersecurity IDS Dataset
- **UNSW-NB15** : University of New South Wales Network-Based Dataset

---

*Note : Les données synthétiques sont générées aléatoirement avec une seed fixe (42) pour assurer la reproductibilité.*
