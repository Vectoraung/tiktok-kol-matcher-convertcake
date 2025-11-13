import streamlit as st
import asyncio

from find_tiktok_influencers_graph import graph

# Initialize session state log
if st.session_state.get("log", None) is None:
    st.session_state["log"] = []

# Show previous logs
logs = st.session_state.get("log", [])
for log in logs:
    st.write(log)

# Input box for website URL
website_url_input = st.text_input("Enter website URL to scrape", "")

config = {"configurable": {"thread_id": "crawl-test"}}

# Async function to run the graph
async def main(website_url):
    inputs = {
        "website_url": website_url,
        "analyzed_data_use_cache": False,
        "product_category_use_cache": False,
        "search_queries_use_cache": False,
        "influencers_scores_use_cache": False
    }

    # Set stream_mode="custom" to receive the custom data in the stream
    async for chunk in graph.astream(inputs, stream_mode="custom", config=config):
        st.session_state["log"].append(chunk['message'])
        st.write(chunk['message'])

    st.write("All Done!")

# Button to start graph
if st.button("Start Graph"):
    if website_url_input.strip() == "":
        st.warning("Please enter a valid website URL before starting.")
    else:
        # Run the async function with input URL
        asyncio.run(main(website_url_input.strip()))