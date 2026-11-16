# Métriques d'évaluation en NLP : résumé automatique, question-réponse, génération de texte

Ce dépôt contient le notebook `NLP_Evaluation_Metrics.ipynb` et sa documentation.

Pour chaque métrique, le notebook suit la même logique : *ce qu'elle mesure → formule → calcul pas à pas → score final → interprétation → avantages et limites*. Tous les scores sont **calculés par le code**.

---

## 1. Contenu du dépôt

```
.
├── NLP_Evaluation_Metrics.ipynb   # code + explications + résultats
└── README.md                      # documentation du code
```

## 2. Métriques implémentées

| Tâche | Métriques implémentées à la main |
|---|---|
| **A. Résumé automatique** | ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (P, R, F1) |
| **B. Question-réponse** | Exact Match (EM), précision, rappel, F1 par tokens |
| **C. Génération de texte** | Perplexité, Distinct-1, Distinct-2 (+ évaluation humaine, expliquée) |

## 3. Installation et exécution

### Option 1 : Google Colab (recommandée)
1. Ouvrir `NLP_Evaluation_Metrics.ipynb` dans [Google Colab](https://colab.research.google.com/) (*Fichier → Importer* ou via GitHub).
2. Exécuter **Exécution → Tout exécuter**.

`torch` est déjà installé sur Colab.

### Option 2 : en local
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install rouge-score bert-score transformers gensim torch numpy pandas matplotlib jupyter
jupyter notebook NLP_Evaluation_Metrics.ipynb
```

### Dépendances

| Bibliothèque | Rôle |
|---|---|
| `numpy`, `pandas`, `matplotlib` | calculs, tableaux, graphiques |
| `rouge-score` | **vérification** de notre ROUGE manuel |
| `gensim` | chargement des vecteurs GloVe (`glove-wiki-gigaword-100`, ~128 Mo) |
| `bert-score`, `transformers`, `torch` | vrai BERTScore (DistilBERT) et perplexité GPT-2 (cellules optionnelles) |

> Les téléchargements de modèles (GloVe, DistilBERT, GPT-2) se font à la première exécution. Les cellules **optionnelles** affichent un message d'erreur sans bloquer le notebook si le téléchargement échoue.

---

## 4. Structure du notebook et documentation du code

### Configuration
- `tokenize(text)` : met en minuscules et garde uniquement lettres et chiffres (même convention que la bibliothèque ROUGE).
- `prf(overlap, n_pred, n_ref)` : calcule **Précision** (recouvrement / taille de la prédiction), **Rappel** (recouvrement / taille de la référence) et **F1** (moyenne harmonique). Fonction réutilisée par ROUGE et par la QA.

### Section A — Résumé automatique

**Exemple :** une référence et deux candidats.
- *Candidat 1* : paraphrase (même sens, mots différents).
- *Candidat 2* : mêmes mots, **sens erroné** (*cut* au lieu de *raised*, *unemployment* au lieu de *inflation*).

**A.1 ROUGE (recouvrement lexical)**

| Fonction | Rôle |
|---|---|
| `ngrams(tokens, n)` | génère les n-grammes d'une liste de tokens |
| `rouge_n(ref, cand, n)` | ROUGE-N : intersection de `Counter` (comptes plafonnés), puis P, R, F1 |
| `lcs(a, b)` | plus longue sous-séquence commune par programmation dynamique, avec remontée pour retrouver la séquence |
| `rouge_l(ref, cand)` | ROUGE-L : P, R, F1 à partir de la LCS |

Le F1 manuel est **vérifié par `assert`** contre `rouge_score.RougeScorer` (écart < 1e-9).

**A.2 BERTScore (similarité sémantique)**

| Fonction | Rôle |
|---|---|
| `cosine(a, b)` | similarité cosinus entre deux vecteurs |
| `bertscore_greedy(ref, cand, kv)` | matrice de cosinus entre les mots, **appariement glouton** : Rappel = moyenne des max par mot de la référence, Précision = moyenne des max par mot du candidat |

- L'exemple pas à pas utilise des vecteurs **GloVe** (non contextuels) pour montrer chaque étape de l'algorithme.
- Une cellule optionnelle lance le **vrai BERTScore** (`bert_score`, DistilBERT, embeddings contextuels).
- Le notebook affiche une heatmap de la matrice des cosinus et un graphique comparant ROUGE-1 et BERTScore.

**Résultat clé :** le candidat 2 (sens faux) obtient un score plus élevé que le candidat 1 (bonne paraphrase) avec les deux familles de métriques. Aucune ne vérifie les faits.

### Section B — Question-réponse

| Fonction | Rôle |
|---|---|
| `normalize(text)` | normalisation façon SQuAD : minuscules, suppression de la ponctuation et des articles *a, an, the* |
| `exact_match(pred, gold)` | 1 si les textes normalisés sont identiques, 0 sinon |
| `qa_scores(pred, gold)` | EM, précision, rappel et F1 par tokens |

- 6 exemples (`qa_examples`) couvrant : réponse exacte, bonne réponse avec mots en trop, réponse correcte formulée autrement, réponse incomplète, réponse fausse.
- Calcul pas à pas sur la question 2, tableau des 6 exemples, score global du jeu (EM moyen vs F1 moyen) et graphique.

**Résultat clé :** EM ≈ 0.33 contre F1 ≈ 0.58 : l'écart mesure combien de réponses sont « presque justes ».

### Section C — Génération de texte

**Trois textes de test :**
- **A** : phrase répétée (répétitif)
- **B** : 11 mots différents formant une phrase sensée (divers, fluide)
- **C** : les mêmes 11 mots que B, mélangés (divers, sans sens)

**C.1 Perplexité**

| Élément | Rôle |
|---|---|
| `corpus`, `uni`, `bi`, `V` | mini-corpus d'entraînement, comptes d'unigrammes et de bigrammes, taille du vocabulaire (+1 pour `<unk>`) |
| `bigram_prob(prev, w)` | probabilité bigramme avec **lissage add-one** : `(count(prev,w)+1) / (count(prev)+V)` |
| `perplexity(sentence, verbose)` | `PPL = exp(−moyenne des ln P)`, avec affichage du détail pas à pas |

Une cellule optionnelle calcule la perplexité d'un vrai modèle (**GPT-2**, 124 M de paramètres). Ces valeurs ne sont pas comparables à celles du modèle bigramme jouet.

**C.2 Distinct-1 / Distinct-2**
- `distinct_n(text, n)` = n-grammes uniques / n-grammes totaux.

**C.3 Synthèse :** tableau et graphique des 3 textes × 3 métriques.
- A vs B : Distinct-n détecte la répétition, pas la perplexité.
- B vs C : Distinct-n identique (mêmes mots), seule la perplexité détecte le charabia.
- **Aucune métrique ne suffit seule.**

**C.4 Évaluation humaine** (partie théorique) : protocole de notation, MOS, accord entre évaluateurs (kappa de Cohen, alpha de Krippendorff), limites.

---

## 5. Résumé des résultats principaux

| Constat | Métrique(s) |
|---|---|
| Un résumé faux avec les bons mots est mieux noté qu'une bonne paraphrase | ROUGE, BERTScore |
| Les scores BERTScore ne sont pas comparables à ceux de ROUGE (seuil de base élevé) | BERTScore |
| Une réponse juste mais trop longue ou incomplète a EM = 0 mais F1 > 0 | EM, F1 |
| Texte répétitif et texte fluide ont une perplexité similaire | Perplexité |
| Texte aléatoire = diversité maximale | Distinct-n |

## 6. Références

- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
- Zhang, T. et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020. arXiv:1904.09675.
- Pennington, J., Socher, R., Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- Rajpurkar, P. et al. (2016). *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP 2016. arXiv:1606.05250.
- Li, J. et al. (2016). *A Diversity-Promoting Objective Function for Neural Conversation Models.* NAACL-HLT 2016.
- Jurafsky, D., Martin, J. H. *Speech and Language Processing* (3rd ed. draft), chapitre sur les modèles de langage n-grammes.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI.
