import re
import pandas as pd


def extract_product_keyword(query):
    """
    Extract a simple product keyword from the user's sentence.
    """

    query = query.lower()

    products = [
        "laptop",
        "running shoes",
        "shoes",
        "headphones",
        "earbuds",
        "mobile",
        "phone",
        "shirt",
        "jeans",
        "watch",
        "camera",
        "bag",
        "backpack",
        "television",
        "tv"
    ]

    # Check longer phrases first
    products = sorted(products, key=len, reverse=True)

    for product in products:
        if product in query:
            return product

    # If no predefined product is found,
    # remove common words and use remaining words.
    words = re.findall(r"[a-zA-Z]+", query)

    stop_words = {
        "i", "want", "to", "buy", "a", "an", "the",
        "need", "looking", "for", "good", "best",
        "product", "please", "give", "me"
    }

    keywords = [
        word for word in words
        if word not in stop_words
    ]

    if keywords:
        return keywords[-1]

    return None


def get_recommendation(df, user_query):

    product_keyword = extract_product_keyword(user_query)

    if not product_keyword:
        return {
            "found": False,
            "message": "Sorry, I could not identify the product."
        }

    # Search product name, category and product type
    keyword = product_keyword.lower()

    mask = (
        df["product_name"].str.lower().str.contains(keyword, na=False)
        | df["category"].str.lower().str.contains(keyword, na=False)
        | df["product_type"].str.lower().str.contains(keyword, na=False)
    )

    matched_df = df[mask].copy()

    if matched_df.empty:
        return {
            "found": False,
            "product": product_keyword,
            "message": f"No products found for '{product_keyword}'."
        }

    # Create sentiment if it does not already exist
    if "sentiment" not in matched_df.columns:
        matched_df["sentiment"] = matched_df["rating"].apply(
            lambda x: "Positive" if x >= 4
            else "Neutral" if x == 3
            else "Negative"
        )

    products = []

    # Analyze each individual product
    for product_name, product_df in matched_df.groupby("product_name"):

        total = len(product_df)

        positive = (
            product_df["sentiment"] == "Positive"
        ).sum()

        neutral = (
            product_df["sentiment"] == "Neutral"
        ).sum()

        negative = (
            product_df["sentiment"] == "Negative"
        ).sum()

        average_rating = product_df["rating"].mean()

        positive_percent = (positive / total) * 100
        neutral_percent = (neutral / total) * 100
        negative_percent = (negative / total) * 100

        # Recommendation score
        score = (
            (average_rating / 5) * 50
            + (positive_percent / 100) * 50
        )

        score = round(score, 2)

        # Recommendation decision
        if average_rating >= 4 and positive_percent >= 60:
            recommendation = "BUY"
            recommendation_text = "✓ YES, YOU CAN BUY THIS PRODUCT"

        elif average_rating >= 3 and positive_percent >= 40:
            recommendation = "CONSIDER"
            recommendation_text = "⚠ CONSIDER THIS PRODUCT"

        else:
            recommendation = "AVOID"
            recommendation_text = "✗ NO, THIS PRODUCT IS NOT RECOMMENDED"

        products.append({
            "product_name": product_name,
            "average_rating": round(average_rating, 2),
            "positive_percent": round(positive_percent, 1),
            "neutral_percent": round(neutral_percent, 1),
            "negative_percent": round(negative_percent, 1),
            "score": score,
            "recommendation": recommendation,
            "recommendation_text": recommendation_text,
            "review_count": total
        })

    # Sort best products first
    products.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best_product = products[0]

    return {
        "found": True,
        "product": product_keyword,
        "best_product": best_product,
        "products": products
    }