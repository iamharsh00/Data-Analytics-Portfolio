import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)

def generate_restaurant_data(num_restaurants=200, num_reviews=5000):
    print("Generating Restaurant Data...")
    
    cuisines_list = ["Indian", "Chinese", "Italian", "Mexican", "Continental", "Fast Food", "Desserts", "Healthy", "South Indian", "North Indian"]
    
    # 1. Restaurants Data
    restaurant_ids = [f"R{str(i).zfill(4)}" for i in range(1, num_restaurants + 1)]
    
    prefixes = ["The Great", "Spicy", "Crispy", "Royal", "Urban", "Desi", "Golden", "Mama's", "Oven", "Chill"]
    suffixes = ["Bites", "Kitchen", "Diner", "Palace", "Wok", "Bowl", "Spoon", "Grill", "Cafe", "Eatery"]
    names = [f"{random.choice(prefixes)} {random.choice(suffixes)}" for _ in range(num_restaurants)]
    
    cuisines = [", ".join(np.random.choice(cuisines_list, size=random.randint(1, 3), replace=False)) for _ in range(num_restaurants)]
    avg_costs = np.random.choice([200, 300, 400, 500, 800, 1000, 1500], size=num_restaurants, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05])
    ratings = np.random.normal(loc=3.8, scale=0.6, size=num_restaurants).clip(1.0, 5.0).round(1)
    
    restaurants_df = pd.DataFrame({
        "Restaurant_ID": restaurant_ids,
        "Name": names,
        "Cuisines": cuisines,
        "Average_Cost_For_Two": avg_costs,
        "Overall_Rating": ratings
    })
    
    # 2. Reviews Data
    positive_phrases = ["Amazing food!", "Loved the ambiance.", "Great service, very fast.", "The taste was authentic.", "Highly recommended!", "Best in town.", "Worth every penny.", "Will definitely order again.", "Fresh and hot.", "Generous portions."]
    negative_phrases = ["Terrible taste.", "Delivery was very late.", "Food was cold.", "Overpriced.", "Not worth the hype.", "Rude staff.", "Too salty.", "Stale ingredients.", "Not enough quantity.", "Bad packaging."]
    neutral_phrases = ["It was okay.", "Nothing special.", "Standard quality.", "Average experience.", "Decent food for the price.", "Might try again.", "A bit plain.", "Hit or miss.", "Met expectations, mostly.", "Just another place."]
    
    user_ids = [f"U{str(i).zfill(5)}" for i in range(1, 1000)]
    
    review_data = []
    
    for _ in range(num_reviews):
        rest_idx = random.randint(0, num_restaurants - 1)
        r_id = restaurant_ids[rest_idx]
        base_rating = restaurants_df.loc[rest_idx, "Overall_Rating"]
        
        # Simulating a diverse set of reviews around the base rating
        review_rating = int(np.clip(np.random.normal(loc=base_rating, scale=1.0), 1, 5).round())
        
        if review_rating >= 4:
            text = random.choice(positive_phrases)
        elif review_rating <= 2:
            text = random.choice(negative_phrases)
        else:
            if random.random() > 0.5:
                text = random.choice(neutral_phrases)
            else:
                text = random.choice(positive_phrases) if random.random() > 0.5 else random.choice(negative_phrases)
        
        # Add some combined feedback for NLP length
        if random.random() < 0.3 and review_rating >= 4:
            text += f" {random.choice(['I especially liked the dessert.', 'The main course was perfectly cooked.', 'Five stars!'])}"
            
        review_data.append({
            "Review_ID": f"REV{str(_).zfill(5)}",
            "Restaurant_ID": r_id,
            "User_ID": random.choice(user_ids),
            "Rating": review_rating,
            "Review_Text": text
        })
        
    reviews_df = pd.DataFrame(review_data)
    
    # Save datasets
    base_dir = os.path.dirname(__file__)
    restaurants_df.to_csv(os.path.join(base_dir, "restaurants.csv"), index=False)
    reviews_df.to_csv(os.path.join(base_dir, "reviews.csv"), index=False)
    
    print(f"Generated {num_restaurants} restaurants and {num_reviews} reviews successfully.")

if __name__ == "__main__":
    generate_restaurant_data()
