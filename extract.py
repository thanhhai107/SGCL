import torch
import pandas as pd
import os
import sys
import glob
from recbole_gnn.quick_start import load_data_and_model

sys.path.append(os.path.dirname(__file__))

def find_latest_model(saved_dir="saved"):
    """
    Find the latest saved model checkpoint by modification time, regardless of model name.
    """
    pattern = os.path.join(saved_dir, "*.pth")
    model_files = glob.glob(pattern)
    
    if not model_files:
        return None
    
    # Sort by modification time, the latest file will be at the end
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def extract_embeddings():
    try:
        model_path = find_latest_model()
        if model_path is None:
            print("No model checkpoint (.pth) found in saved folder")
            return None, None
        elif not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None, None
        else:
            print(f"Loading model from: {model_path}")
        
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_path)
        print(f"Number of users: {model.n_users}, Number of items: {model.n_items}")
        
        model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = model.forward()
        
        print(f"User embeddings shape: {user_embeddings.shape}")
        print(f"Item embeddings shape: {item_embeddings.shape}")
        
        return user_embeddings.cpu().numpy(), item_embeddings.cpu().numpy()
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def save_embeddings(user_embeddings, item_embeddings, output_dir="embeddings"):
    if user_embeddings is None or item_embeddings is None:
        print("Error: User or item embeddings is None")
        return
    user_df, item_df = pd.DataFrame(user_embeddings), pd.DataFrame(item_embeddings)
    user_df.index.name, item_df.index.name = 'user_id', 'item_id'
    os.makedirs(output_dir, exist_ok=True)
    user_file = os.path.join(output_dir, "user_embeddings.csv")
    item_file = os.path.join(output_dir, "item_embeddings.csv")
    user_df.to_csv(user_file)
    item_df.to_csv(item_file)

def main():
    user_embeddings, item_embeddings = extract_embeddings()
    save_embeddings(user_embeddings, item_embeddings)

if __name__ == "__main__":
    main()