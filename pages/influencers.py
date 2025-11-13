import streamlit as st
import json
import os

file_path = 'cache/influencers_scores.json'

if not os.path.exists(file_path):
    st.warning("Influencer data not found! Please make sure you ran the Graph page first.")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not data:
                st.warning("Influencer data is empty!")
            else:
                st.set_page_config(page_title="Influencer Ranking", page_icon="⭐", layout="centered")

                st.title("Weighted Influencer Scoring App")
                st.write("Adjust the sliders to change how the influencers are ranked.")

                # Sliders for weights
                w1 = st.slider("Weight for Performance Score (w₁)", 0.0, 1.0, 0.3)
                w2 = st.slider("Weight for Category Match (w₂)", 0.0, 1.0, 0.4)
                w3 = st.slider("Weight for Tone Match (w₃)", 0.0, 1.0, 0.2)

                # Normalize weights so total = 1
                total = w1 + w2 + w3
                if total != 0:
                    w1, w2, w3 = w1 / total, w2 / total, w3 / total

                # Compute overall scores
                for item in data:
                    item["overall_score"] = (
                        w1 * item.get("performance_score", 0)
                        + w2 * item.get("category_match_score", 0)
                        + w3 * item.get("tone_match_score", 0)
                    )

                # Sort data by score descending
                sorted_data = sorted(data, key=lambda x: x["overall_score"], reverse=True)

                st.subheader("Ranked Results")

                # Styled containers
                for idx, item in enumerate(sorted_data, start=1):
                    with st.container():
                        container = st.container(border=True)
                        container.subheader(f"{idx}. {item['authorMeta/name']}")
                        container.write(f"Name: {item['authorMeta/name']}")
                        container.write(f"Signature: {item['authorMeta/signature']}")
                        container.write(f"Heart: {item['authorMeta/heart']}")
                        container.write(f"Fans: {item['authorMeta/fans']}")
                        container.markdown(f"[Visit Profile]({item['authorMeta/profileUrl']})")
        except json.JSONDecodeError:
            st.error("Failed to read influencer data! Please check if the JSON file is valid.")
