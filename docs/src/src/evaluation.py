# EVALUATION METRICS FOR MOVIE RECOMMENDATION:
import numpy as np
# 1. PREDICTION ACCURACY METRICS:
def calculate_rmse(predictions, actuals):
    return np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
def calculate_mae(predictions, actuals):
    return np.mean(np.abs(np.array(predictions) - np.array(actuals)))
  
# 2. RANKING QUALITY METRICS:
def precision_at_k(recommendations, relevant_items, k):
    if k == 0:
        return 0
    top_k = recommendations[:k]
    relevant_count = sum(1 for item in top_k if item in relevant_items)
    return relevant_count / k

def recall_at_k(recommendations, relevant_items, k):
    if len(relevant_items) == 0:
        return 0
    top_k = recommendations[:k]
    relevant_count = sum(1 for item in top_k if item in relevant_items)
    return relevant_count / len(relevant_items)
  
def f1_at_k(recommendations, relevant_items, k):
    p = precision_at_k(recommendations, relevant_items, k)
    r = recall_at_k(recommendations, relevant_items, k)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def ndcg_at_k(recommendations, relevant_items, k):
    top_k = recommendations[:k]
    
    # Calculating the DCG (Discounted Cumulative Gain):
    dcg = 0
    for i, item in enumerate(top_k):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)  
          
    # Calculating the IDCG (Ideal DCG - best possible ranking)
    ideal_count = min(len(relevant_items), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_count))
    if idcg == 0:
        return 0
    return dcg / idcg

def hit_rate_at_k(recommendations, relevant_items, k):
    top_k = recommendations[:k]
    return 1 if any(item in relevant_items for item in top_k) else 0

def mrr_at_k(recommendations, relevant_items, k):
    top_k = recommendations[:k]
    for i, item in enumerate(top_k):
        if item in relevant_items:
            return 1 / (i + 1)
    return 0

# 3. COVERAGE AND DIVERSITY:
def calculate_coverage(recommendations_list, total_items):
    all_recommended = set()
    for recs in recommendations_list:
        all_recommended.update(recs)
    return len(all_recommended) / total_items

def calculate_genre_diversity(recommended_items, movies_df):
    genres = set()
    for movie_id in recommended_items:
        movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values
        if len(movie_genres) > 0:
            for genre in movie_genres[0].split('|'):
                genres.add(genre)
    
    return len(genres)

# 4. COMPLETING THE EVALUATION FUNCTION:
def evaluate_model(model, test_data, movies_df, get_recommendations_func, k_values=[5, 10, 15, 20]):
    results = {}
    # Getting relevant items for each user (movies rated 4 or higher):
    relevant_items_dict = {}
    for user_id in test_data['userId'].unique():
        user_data = test_data[test_data['userId'] == user_id]
        relevant = user_data[user_data['rating'] >= 4]['movieId'].tolist()
        relevant_items_dict[user_id] = relevant
    # For each K
    for k in k_values:
        all_precision = []
        all_recall = []
        all_f1 = []
        all_ndcg = []
        all_hit_rate = []
        all_mrr = []
        all_recommendations = []
        for user_id in test_data['userId'].unique():
            relevant = relevant_items_dict[user_id]
            if len(relevant) == 0:
                continue
            
            # Getting recommendations:
            recs = get_recommendations_func(model, user_id, k)
            rec_ids = [r[0] for r in recs]
            all_recommendations.append(rec_ids)
            
            # Calculating the metrics:
            all_precision.append(precision_at_k(rec_ids, relevant, k))
            all_recall.append(recall_at_k(rec_ids, relevant, k))
            all_f1.append(f1_at_k(rec_ids, relevant, k))
            all_ndcg.append(ndcg_at_k(rec_ids, relevant, k))
            all_hit_rate.append(hit_rate_at_k(rec_ids, relevant, k))
            all_mrr.append(mrr_at_k(rec_ids, relevant, k))
        
        results[f'precision@{k}'] = np.mean(all_precision)
        results[f'recall@{k}'] = np.mean(all_recall)
        results[f'f1@{k}'] = np.mean(all_f1)
        results[f'ndcg@{k}'] = np.mean(all_ndcg)
        results[f'hit_rate@{k}'] = np.mean(all_hit_rate)
        results[f'mrr@{k}'] = np.mean(all_mrr)
    
    # Calculating the coverage:
    coverage = calculate_coverage(all_recommendations, len(movies_df))
    results['coverage'] = coverage
    return results
print("Evaluation functions loaded successfully!")
