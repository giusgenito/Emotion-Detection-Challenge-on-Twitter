# Twitter Emotion Detection via Ensemble Learning 🎭

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLP](https://img.shields.io/badge/Task-NLP%20Classification-orange)
![Sklearn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Project Overview

Questo progetto affronta una sfida di **Natural Language Processing (NLP)**: classificare l'emozione dominante in una serie di Tweet. A differenza della classica *Sentiment Analysis* binaria (positivo/negativo), questo modello distingue tra quattro classi emotive specifiche:
1. **Anger** (Rabbia) 😠
2. **Joy** (Gioia) 😂
3. **Optimism** (Ottimismo) 😀
4. **Sadness** (Tristezza) 😞

L'obiettivo è stato massimizzare l'accuratezza predittiva combinando le forze di molteplici algoritmi di Machine Learning attraverso una strategia di **Hard Voting Ensemble**.

## 🚀 Pipeline di Elaborazione

### 1. Text Preprocessing & Cleaning
Il testo dei tweet è stato pulito e normalizzato per ridurre il rumore:
* Rimozione di handle (@user), hashtag, URL e caratteri speciali.
* Rimozione delle **Stopwords** inglesi.
* **Lemmatizzazione** e Stemming per ridurre le parole alla loro radice (utilizzando NLTK).



### 2. Feature Extraction (TF-IDF)
Conversione del testo in vettori numerici utilizzando **TF-IDF (Term Frequency-Inverse Document Frequency)**.
* Sono stati considerati unigrammi e bigrammi (`ngram_range=(1, 2)`) per catturare il contesto locale delle parole.

### 3. Modellazione & Ensemble
Sono stati addestrati e validati 10 algoritmi diversi per garantire diversità nella predizione:

* **Linear Models**: Logistic Regression, Passive Aggressive Classifier.
* **Probabilistic**: Multinomial Naive Bayes.
* **Support Vector Machines**: SVM (Linear SVC).
* **Tree-Based & Boosting**: Random Forest, Extra Trees, **XGBoost**, **LightGBM**, AdaBoost.
* **Neural Networks**: MLP Classifier (Multi-Layer Perceptron).



### 4. Voting Strategy
Le predizioni di tutti i modelli sono state aggregate utilizzando un **Hard Voting System** (Majority Rule). La classe finale è determinata dalla moda delle predizioni dei singoli classificatori, riducendo la varianza e migliorando la robustezza rispetto ai singoli modelli.

## 🛠️ Tecnologie Utilizzate

* **Linguaggio**: Python
* **NLP**: NLTK (WordNetLemmatizer, Stopwords), Re (Regex)
* **ML Libraries**: Scikit-learn, XGBoost, LightGBM
* **Data Manipulation**: Pandas, NumPy

## 📊 Risultati

La strategia di Ensemble ha permesso di superare le performance dei singoli classificatori deboli.
* Il modello finale combina le "opinioni" di 10 diversi classificatori.
* La matrice di confusione mostra una buona capacità di distinzione anche tra classi semanticamente vicine (es. Joy vs Optimism).

## 💻 Come Eseguire il Codice

1.  **Clona la repository**:
    ```bash
    git clone [https://github.com/tuo-username/emotion-detection-nlp.git](https://github.com/tuo-username/emotion-detection-nlp.git)
    ```

2.  **Installa le dipendenze**:
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm nltk
    ```

3.  **Setup NLTK**:
    Assicurati di scaricare le risorse necessarie:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

4.  **Esegui il Notebook**:
    Lancia il file `Emotional_Detection_Voting.ipynb` per riprodurre il training e la generazione del file `merged_labels.csv`.

## 👨‍💻 Autore

**[Tuo Nome]**
* Master Student in Computer Science
* [LinkedIn](https://linkedin.com/in/tuo-profilo) | [GitHub](https://github.com/tuo-username)
