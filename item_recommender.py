import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from mlxtend.frequent_patterns import apriori, association_rules
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load dataset as Parquet
def load_data(parquet_file='online_retail.parquet'):
    try:
        df = pd.read_parquet(parquet_file)
        print("Loaded data from Parquet.")
    except FileNotFoundError:
        print("Parquet file not found. Preprocessing data...")
        df = preprocess_data(parquet_file)
    return df

# Preprocess and save as Parquet
def preprocess_data(parquet_file):
    df = pd.read_excel('Online Retail.xlsx', parse_dates=["InvoiceDate"])
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['StockCode'] = df['StockCode'].astype(str)
    df['Description'] = df['Description'].astype(str)
    df.to_parquet(parquet_file)
    print("Data preprocessed and saved as Parquet.")
    return df

# Create stock code to index mapping and stock description mapping
def create_stockcode_mapping(df):
    stockcode_to_index = {stock: i for i, stock in enumerate(df['StockCode'].unique())}
    stock_description_mapping = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode').to_dict()['Description']
    return stockcode_to_index, stock_description_mapping

# Create item vectors using Word2Vec
def create_item_vectors(df):
    descriptions = [desc.split() for desc in df['Description'].dropna().unique()]
    w2v_model = Word2Vec(sentences=descriptions, vector_size=75, window=5, min_count=1, workers=4)
    item_vector_dict = {}
    for stock_code, desc in df[['StockCode', 'Description']].dropna().values:
        words = desc.split()
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if vectors:
            item_vector_dict[stock_code] = np.mean(vectors, axis=0)
    df['item_vector'] = df['StockCode'].map(item_vector_dict)
    df = df[df['item_vector'].notna()]
    item_vectors_matrix = np.vstack(df['item_vector'].values)
    return item_vector_dict, item_vectors_matrix, w2v_model

# Create customer-item matrix
def create_customer_item_matrix(df):
    customer_item_matrix = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    print("Customer-item matrix created with shape:", customer_item_matrix.shape)
    return customer_item_matrix

# Get item vector by StockCode
def get_item_vector(stock_code, item_vector_dict):
    return item_vector_dict.get(stock_code, None)

# Compute cosine similarity
def compute_similarity(search_vector, item_vector):
    # Reshape the vectors to 2D arrays (each as a single row vector)
    search_vector = search_vector.reshape(1, -1)
    item_vector = item_vector.reshape(1, -1)
    return cosine_similarity(search_vector, item_vector).mean()


# 1. Apriori: Create association rules
def create_association_rules(df, min_support=0.01, min_confidence=0.1):
    basket = (df.groupby(['InvoiceNo', 'StockCode'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
    basket = basket.map(lambda x: 1 if x > 0 else 0)

    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    rules = rules[['antecedents', 'consequents', 'confidence']].sort_values(by='confidence', ascending=False)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])

    return rules

# 2. Apriori-based Recommendation
def get_apriori_recommendations(customer_id, customer_item_matrix, rules, top_n=5):
    purchased_items = customer_item_matrix.loc[customer_id]
    purchased_items = purchased_items[purchased_items > 0].index.tolist()
    
    recommendations = []
    for item in purchased_items:
        consequents = rules[rules['antecedents'] == item]['consequents'].values
        recommendations.extend(consequents)
    
    recommendations = list(set(recommendations) - set(purchased_items))
    return recommendations[:top_n]


# Hybrid recommendation based on search term and customer history
def get_hybrid_recommendations(customer_id, search_term, item_vector_dict, stock_description_mapping, customer_item_matrix,rules,top_n=15, weight_text=1.0, weight_history=1.0, w2v_model=None):
    # Convert search term to vector using Word2Vec model
    search_terms = search_term.split()
    search_vectors = [w2v_model.wv[word] for word in search_terms if word in w2v_model.wv]
    
    if not search_vectors:
        print("Search term is not found in the Word2Vec vocabulary.")
        return pd.DataFrame(columns=["StockCode", "Description"])

    # Compute the average vector for the search term
    search_vector = np.mean(search_vectors, axis=0)

    # Compute similarity between the search vector and each product's vector
    similarities = []
    for stock_code, item_vector in item_vector_dict.items():
        similarity = compute_similarity(search_vector, item_vector)
        similarities.append((stock_code, similarity))

    # Sort by similarity score (search term relevance)
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Get customer’s past purchase history
    customer_history = customer_item_matrix.loc[customer_id].values

    # Hybrid score (combining search similarity with customer history)
    hybrid_scores = []
    for stock_code, similarity in similarities:
        product_index = stockcode_to_index.get(stock_code, None)
        if product_index is not None:
            history_score = customer_history[product_index]  # History relevance (purchase frequency)
            hybrid_score = weight_text * similarity + weight_history * history_score
            hybrid_scores.append((stock_code, hybrid_score))

    # Sort by hybrid score
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    hybrid_top = hybrid_scores[:top_n]
    hybrid_top_codes = [item[0] for item in hybrid_top]

    # Apriori Recommendations
    apriori_recommendations = get_apriori_recommendations(customer_id, customer_item_matrix, rules, top_n=5)
    all_recommendations = list(set(hybrid_top_codes + apriori_recommendations))

    recommendations = [(code, stock_description_mapping.get(code, "No Description")) for code in all_recommendations]
    recommendations_df = pd.DataFrame(recommendations, columns=['StockCode', 'Description'])
    return recommendations_df


if __name__ == "__main__":
    df = load_data()  # Load or preprocess data
    stockcode_to_index, stock_description_mapping = create_stockcode_mapping(df)  # Create the stockcode to index mapping
    item_vector_dict, item_vectors_matrix, w2v_model = create_item_vectors(df)  # Create item vectors for Word2Vec
    customer_item_matrix = create_customer_item_matrix(df)  # Create customer-item matrix
    rules = create_association_rules(df)
    customer_id = 12350  # Example customer ID
    search_criteria = "T-LIGHT HOLDER"  # Example search criteria (you can input any term)
    recommendations = get_hybrid_recommendations(customer_id, search_criteria, item_vector_dict, stock_description_mapping, customer_item_matrix, rules, top_n=15, w2v_model=w2v_model)  # Get hybrid recommendations
    print(recommendations)  # Print recommendations