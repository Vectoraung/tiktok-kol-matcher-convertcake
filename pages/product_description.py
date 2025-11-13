import streamlit as st
import json
import os

file_path = 'cache/product_category.json'

if not os.path.exists(file_path):
    st.warning("Product data hasn't scrapped yet. Scrape the client's website in the Graph page.")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            product = json.load(f)
            if not product:
                st.warning("Product data is empty!")
            else:
                # Streamlit app layout
                st.title(product.get("product_name", "No Name"))
                
                st.subheader("Category")
                st.write(f"Main: {product.get('main_category', 'N/A')}")
                subcategories = product.get('subcategory_list', [])
                st.write(f"Subcategories: {', '.join(subcategories) if subcategories else 'N/A'}")

                st.subheader("Description")
                st.write(product.get("description", "N/A"))

                st.subheader("Unique Selling Points")
                st.write(product.get("unique_selling_points", "N/A"))

                st.subheader("Target Audience")
                target = product.get("target_audience", {})
                st.write(f"Gender: {target.get('gender', 'N/A')}")
                st.write(f"Age Range: {target.get('age_range', 'N/A')}")
                st.write(f"Location: {target.get('location', 'N/A')}")
                interests = target.get('interest', [])
                st.write(f"Interests: {', '.join(interests) if interests else 'N/A'}")

                st.subheader("Brand Personality")
                personalities = product.get("brand_personality", [])
                st.write(", ".join(personalities) if personalities else 'N/A')
        except json.JSONDecodeError:
            st.error("Failed to read product data! Please check if the JSON file is valid.")
