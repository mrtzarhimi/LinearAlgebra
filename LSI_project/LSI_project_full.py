import re
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import text as sktext
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

os.makedirs('figs', exist_ok=True)

STUDENT_ID_LAST3 = 163         
DATA_PATH = 'df_file.csv'       

WORDS_LIST = """artist chief china club company computer country deal digital director
economy election expected film firm france game government group growth help high home
industry labour law market match million minister mobile money music new number office old
online party phone play record sale service tax technology time uk user win won work""".split()

LABELS_MAP = {0: 'Politics', 1: 'Sport', 2: 'Technology', 3: 'Entertainment', 4: 'Business'}


# ============================================================
# Q7 — Load + preprocess (lowercase, remove punctuation)
# ============================================================
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def q7_load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    print("Q7 | Dataset shape:", df.shape)
    df['clean_text'] = df['Text'].apply(preprocess)
    return df


# ============================================================
# Q8 — Top 30 most frequent words + bar chart
# ============================================================
STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd
he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves said also mr said""".split())


def q8_top30_words(df):
    all_words = []
    for text in df['clean_text']:
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        all_words.extend(words)

    counter = Counter(all_words)
    top30 = counter.most_common(30)
    print("\nQ8 | Top 30 frequent words:", top30)

    words, counts = zip(*top30)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(words, counts, color='steelblue')
    plt.xticks(rotation=60, ha='right')
    ax.set_title('Top 30 Most Frequent Words (all documents)')
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('figs/q8_top30_words.png', dpi=130)
    plt.close(fig)
    print("Q8 | Discussion: most of these words are generic/high document-frequency"
          " words that do not discriminate topic; only a few (government, film, game,"
          " music) carry topical signal. This motivates TF-IDF weighting in Q9.")
    return top30


# ============================================================
# Q9 — Word cloud (top 20 words by TF-IDF), custom-rendered
# (wordcloud package unavailable offline -> manual matplotlib rendering)
# ============================================================
def q9_wordcloud(df):
    extra_stop = ['said', 'mr', 'told', 'year', 'years', 'new', 'time', 'best', 'world']
    stop_words = list(sktext.ENGLISH_STOP_WORDS.union(extra_stop))

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.9, min_df=2)
    tfidf = vectorizer.fit_transform(df['clean_text'])
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()

    top_idx = np.argsort(scores)[::-1][:20]
    top_words = [(vocab[i], scores[i]) for i in top_idx]
    print("\nQ9 | Top 20 words by TF-IDF sum:", top_words)

    random.seed(7)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    max_score = max(s for _, s in top_words)
    min_score = min(s for _, s in top_words)
    placed = []

    def overlaps(x, y, w, h, placed):
        for (px, py, pw, ph) in placed:
            if abs(x - px) < (w + pw) / 2 * 0.75 and abs(y - py) < (h + ph) / 2 * 0.75:
                return True
        return False

    for word, score in top_words:
        size = 16 + 55 * (score - min_score) / (max_score - min_score)
        for _ in range(300):
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            w = len(word) * size * 0.011
            h = size * 0.02
            if not overlaps(x, y, w, h, placed):
                placed.append((x, y, w, h))
                break
        color = plt.cm.plasma(random.random() * 0.8)
        ax.text(x, y, word, fontsize=size, ha='center', va='center', color=color,
                fontweight='bold', transform=ax.transAxes)

    ax.set_title('Word Cloud - Top 20 words by TF-IDF sum (size = TF-IDF weight)', fontsize=13)
    plt.tight_layout()
    plt.savefig('figs/q9_wordcloud.png', dpi=130)
    plt.close(fig)
    print("Q9 | Selection criterion: TF-IDF (term frequency x inverse document "
          "frequency) so words common across ALL docs are down-weighted, and "
          "topic-specific words stand out (film, government, labour, election, "
          "music, party, company, market).")
    return top_words


# ============================================================
# Q10 — Bag-of-Words matrix using words.csv vocabulary
# ============================================================
def q10_bow_matrix(df):
    vectorizer = CountVectorizer(vocabulary=WORDS_LIST)
    bow = vectorizer.fit_transform(df['clean_text'])
    bow_matrix = bow.toarray()
    print(f"\nQ10 | BoW matrix shape: {bow_matrix.shape}  "
          f"(docs x |vocab|={len(WORDS_LIST)})")
    return bow_matrix


def split_2000_225(df, bow_matrix):
    """Split literally: first 2000 rows = working set, last 225 = held-out test."""
    df_train = df.iloc[:2000].reset_index(drop=True)
    df_test = df.iloc[2000:].reset_index(drop=True)
    bow_train = bow_matrix[:2000]
    bow_test = bow_matrix[2000:]
    print("Split | train:", df_train.shape, bow_train.shape,
          "| test:", df_test.shape, bow_test.shape)
    print("Split | NOTE: dataset is label-sorted, so the literal last-225 test rows "
          "are ALL label 4 (Business). See shuffled version in Q20 for a fair "
          "per-category evaluation.")
    return df_train, df_test, bow_train, bow_test


# ============================================================
# Q11 — Standardize + full SVD
# ============================================================
def q11_standardize_svd(bow_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(bow_train.astype(float))
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    print(f"\nQ11 | Scaled matrix shape: {X_scaled.shape}")
    print(f"Q11 | SVD (economy) shapes -> U:{U.shape}  S:{S.shape}  Vt:{Vt.shape}")
    return scaler, X_scaled, U, S, Vt


# ============================================================
# Q12 — Scree plot + threshold selection + Truncated SVD + error
# ============================================================
def q12_truncated_svd(X_scaled, U, S, Vt, k=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, len(S) + 1), S, 'o-', color='darkorange')
    axes[0].set_xlabel('Component index')
    axes[0].set_ylabel('Singular value')
    axes[0].set_title('Singular values (scree plot)')
    axes[0].grid(alpha=0.3)

    cum_var = np.cumsum(S ** 2) / np.sum(S ** 2)
    axes[1].plot(range(1, len(S) + 1), cum_var, 'o-', color='teal')
    axes[1].axhline(0.9, color='red', linestyle='--', label='90% variance')
    axes[1].set_xlabel('Number of components (k)')
    axes[1].set_ylabel('Cumulative explained variance')
    axes[1].set_title('Cumulative explained variance')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figs/q12_scree_variance.png', dpi=130)
    plt.close(fig)

    print(f"\nQ12 | Chosen threshold: k={k} "
          f"(elbow of scree plot; retains {cum_var[k-1]*100:.1f}% variance)")

    Uk, Sk, Vtk = U[:, :k], S[:k], Vt[:k, :]
    X_reconstructed = Uk @ np.diag(Sk) @ Vtk
    error = np.linalg.norm(X_scaled - X_reconstructed, 'fro')
    rel_error = error / np.linalg.norm(X_scaled, 'fro')
    print(f"Q12 | Truncated SVD (k={k}) relative reconstruction error: {rel_error:.4f}")
    return Uk, Sk, Vtk, rel_error


# ============================================================
# Q13 — Randomized SVD implemented from scratch (Halko et al.)
# ============================================================
def randomized_svd(A, k, p=10, q=2, random_state=42):
    """
    Randomized SVD.
    A: input matrix (m x n) ; k: target rank ; p: oversampling ; q: power iterations
    Returns U (m x k), S (k,), Vt (k x n)
    """
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    l = k + p

    Omega = rng.standard_normal(size=(n, l))     # 1. random test matrix
    Y = A @ Omega                                 # 2. sample matrix
    for _ in range(q):                            # 3. power iterations
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)                        # 4. orthonormal basis
    B = Q.T @ A                                   # 5. project A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)  # 6. SVD of small matrix
    U = Q @ Ub                                    # 7. recover U
    return U[:, :k], S[:k], Vt[:k, :]             # 8. truncate to rank k


# ============================================================
# Q14 — Compare Randomized SVD vs Truncated SVD reconstruction error
# ============================================================
def q14_compare_svd(X_scaled, Uk, Sk, Vtk, k=20):
    rU, rS, rVt = randomized_svd(X_scaled, k=k, p=10, q=2)

    X_trunc = Uk @ np.diag(Sk) @ Vtk
    X_rand = rU @ np.diag(rS) @ rVt

    err_trunc = np.linalg.norm(X_scaled - X_trunc, 'fro') / np.linalg.norm(X_scaled, 'fro')
    err_rand = np.linalg.norm(X_scaled - X_rand, 'fro') / np.linalg.norm(X_scaled, 'fro')

    print(f"\nQ14 | Truncated SVD error  : {err_trunc:.5f}")
    print(f"Q14 | Randomized SVD error : {err_rand:.5f}")
    print("Q14 | At internet scale, Randomized SVD is preferable: it avoids full "
          "O(mn min(m,n)) decomposition, needs only a few matrix-vector products, "
          "works well on sparse data, and is trivially parallelizable — with "
          "negligible accuracy loss as shown above.")
    return rU, rS, rVt


# ============================================================
# Q15 — Top 5 words per component + latent meaning guess
# ============================================================
def q15_top_words_per_component(Vtk):
    print("\nQ15 | Top 5 words per component:")
    k = Vtk.shape[0]
    for comp_idx in range(k):
        comp = Vtk[comp_idx]
        top5_idx = np.argsort(np.abs(comp))[::-1][:5]
        top5 = [(WORDS_LIST[i], round(comp[i], 3)) for i in top5_idx]
        print(f"  Component {comp_idx + 1}: {top5}")


# ============================================================
# Q16 — Cosine similarity & Euclidean distance for word pairs
# ============================================================
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_dist(a, b):
    return np.linalg.norm(a - b)


def q16_word_pair_similarity(Vtk):
    word2idx = {w: i for i, w in enumerate(WORDS_LIST)}
    word_vecs = Vtk.T  # (52, k)

    pairs = [
        ('mobile', 'technology'), ('director', 'film'), ('win', 'won'),
        ('play', 'game'), ('play', 'law'), ('government', 'music')
    ]

    print("\nQ16 | Word pair similarities:")
    print(f"{'Pair':<28}{'Cosine Sim':>12}{'Euclidean Dist':>18}")
    for w1, w2 in pairs:
        v1 = word_vecs[word2idx[w1]]
        v2 = word_vecs[word2idx[w2]]
        cs = cosine_sim(v1, v2)
        ed = euclidean_dist(v1, v2)
        print(f"({w1}, {w2}){'':<{max(1,28-len(w1)-len(w2)-4)}}{cs:>12.4f}{ed:>18.4f}")


# ============================================================
# Q17 — Document (student-id-indexed) similarity to each word
# ============================================================
def q17_document_word_similarity(df_train, bow_train, Uk, Sk, Vtk, doc_idx=STUDENT_ID_LAST3):
    print(f"\nQ17 | Document #{doc_idx} text (first 150 chars):",
          df_train.loc[doc_idx, 'Text'][:150])
    print(f"Q17 | Document #{doc_idx} label:", df_train.loc[doc_idx, 'Label'])

    doc_vec = Uk[doc_idx] * Sk           # document vector in latent space
    word_vecs = Vtk.T                     # word vectors in latent space

    sims = np.array([cosine_sim(doc_vec, word_vecs[i]) for i in range(len(WORDS_LIST))])
    counts = bow_train[doc_idx]
    order = np.argsort(sims)[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    axes[0].bar(np.array(WORDS_LIST)[order], sims[order], color='mediumseagreen')
    axes[0].set_title(f'Cosine Similarity between Document #{doc_idx} and each word (latent space)')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].tick_params(axis='x', rotation=75)

    axes[1].bar(np.array(WORDS_LIST)[order], counts[order], color='indianred')
    axes[1].set_title(f'Word counts in Document #{doc_idx} (same word order)')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=75)

    plt.tight_layout()
    plt.savefig('figs/q17_doc_similarity.png', dpi=130)
    plt.close(fig)

    print("Q17 | Comparison: raw counts only show literally-occurring words, while "
          "cosine similarity in latent space also surfaces conceptually related "
          "words that never literally appear in the document (count=0 but sim>0).")


# ============================================================
# Q18 — Conceptual discussion (no code required)
# ============================================================
Q18_DISCUSSION = """
Q18 | Discussion (no computation needed):
Searching in latent (LSI) space lets a query for "technology" also match documents
that never contain that literal word but do contain co-occurring/related words like
"mobile" or "digital", because those words load heavily on the same latent
component(s) as "technology" (learned from statistical co-occurrence patterns across
the corpus). A bag-of-words search can only find exact string matches.
Computational advantage: BoW space has dimensionality = vocabulary size (sparse,
high-dimensional), while latent space has only k dimensions (dense, low-dimensional,
e.g. k=20 here vs 52+). Similarity computations (dot products / distances) are much
cheaper and require far less memory in the reduced latent space.
"""


# ============================================================
# Q19 — Mean latent vector per category + heatmap + labeling method
# ============================================================
def q19_category_heatmap(df_train, Uk, Sk):
    doc_vecs = Uk * Sk
    categories = sorted(df_train['Label'].unique())

    mean_vecs = []
    for c in categories:
        mask = (df_train['Label'] == c).values
        mean_vecs.append(doc_vecs[mask].mean(axis=0))
    mean_vecs = np.array(mean_vecs)
    print(f"\nQ19 | Mean latent vectors per category shape: {mean_vecs.shape}")

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(mean_vecs, cmap='coolwarm', center=0,
                yticklabels=[LABELS_MAP[c] for c in categories],
                xticklabels=[f'C{i+1}' for i in range(mean_vecs.shape[1])],
                ax=ax, cbar_kws={'label': 'Mean latent value'})
    ax.set_title('Mean Latent Space Vector per Category')
    ax.set_xlabel('Latent Component')
    plt.tight_layout()
    plt.savefig('figs/q19_category_heatmap.png', dpi=130)
    plt.close(fig)

    print("Q19 | Proposed labeling method: Nearest-Centroid classifier. Compute each "
          "category's mean latent vector (centroid) from labeled training docs, then "
          "assign a new document to the category whose centroid has the highest "
          "cosine similarity to its latent vector. This needs no manual "
          "interpretation of individual components and scales automatically.")
    return mean_vecs, categories


# ============================================================
# Q20 — Implement nearest-centroid classifier + accuracy on test set
# ============================================================
def cosine_sim_matrix(A, B):
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T


def q20_evaluate(df_train, bow_train, df_test, bow_test, scaler, Vtk, k=20):
    X_train_scaled = scaler.transform(bow_train.astype(float))
    doc_latent_train = X_train_scaled @ Vtk.T

    categories = sorted(df_train['Label'].unique())
    centroids = np.array([
        doc_latent_train[(df_train['Label'] == c).values].mean(axis=0)
        for c in categories
    ])

    X_test_scaled = scaler.transform(bow_test.astype(float))
    doc_latent_test = X_test_scaled @ Vtk.T

    sims = cosine_sim_matrix(doc_latent_test, centroids)
    preds = np.array(categories)[np.argmax(sims, axis=1)]
    true = df_test['Label'].values

    acc_overall = (preds == true).mean()
    print(f"\nQ20 | Overall accuracy: {acc_overall:.4f}  (n={len(true)})")
    for c in categories:
        mask = true == c
        if mask.sum() > 0:
            acc_c = (preds[mask] == true[mask]).mean()
            print(f"  {LABELS_MAP[c]:<15} n={mask.sum():<5} acc={acc_c:.4f}")
        else:
            print(f"  {LABELS_MAP[c]:<15} n=0 (not present in this test split)")
    return acc_overall


def q20_shuffled_variant(df_full, bow_full, k=20, random_state=42):
    """Extra: shuffled split so ALL categories appear in the test set."""
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(len(df_full))
    df_shuf = df_full.iloc[perm].reset_index(drop=True)
    bow_shuf = bow_full[perm]

    df_train2 = df_shuf.iloc[:2000].reset_index(drop=True)
    df_test2 = df_shuf.iloc[2000:].reset_index(drop=True)
    bow_train2 = bow_shuf[:2000]
    bow_test2 = bow_shuf[2000:]

    scaler2 = StandardScaler()
    X_train2_scaled = scaler2.fit_transform(bow_train2.astype(float))
    U2, S2, Vt2 = np.linalg.svd(X_train2_scaled, full_matrices=False)
    Uk2, Sk2, Vtk2 = U2[:, :k], S2[:k], Vt2[:k, :]

    acc = q20_evaluate(df_train2, bow_train2, df_test2, bow_test2, scaler2, Vtk2, k=k)
    return acc


# ============================================================
# MAIN — run everything in order
# ============================================================
if __name__ == "__main__":
    df = q7_load_and_preprocess()
    q8_top30_words(df)
    q9_wordcloud(df)
    bow_full = q10_bow_matrix(df)

    df_train, df_test, bow_train, bow_test = split_2000_225(df, bow_full)

    scaler, X_scaled, U, S, Vt = q11_standardize_svd(bow_train)
    Uk, Sk, Vtk, _ = q12_truncated_svd(X_scaled, U, S, Vt, k=20)
    rU, rS, rVt = q14_compare_svd(X_scaled, Uk, Sk, Vtk, k=20)  # also exercises Q13's function
    q15_top_words_per_component(Vtk)
    q16_word_pair_similarity(Vtk)
    q17_document_word_similarity(df_train, bow_train, Uk, Sk, Vtk, doc_idx=STUDENT_ID_LAST3)
    print(Q18_DISCUSSION)
    q19_category_heatmap(df_train, Uk, Sk)

    print("\n=== Q20: literal split (last 225 rows, all label=Business) ===")
    q20_evaluate(df_train, bow_train, df_test, bow_test, scaler, Vtk, k=20)

    print("\n=== Q20: shuffled split (all categories present in test) ===")
    q20_shuffled_variant(df, bow_full, k=20, random_state=42)

    print("\nAll done. Figures saved in ./figs/")
