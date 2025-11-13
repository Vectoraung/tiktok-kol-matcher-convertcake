import streamlit as st

graph_page = st.Page("pages/graph_page.py", title="Graph", icon=":material/add_circle:")
product = st.Page("pages/product_description.py", title="Product Description", icon=":material/add_circle:")
influencers = st.Page("pages/influencers.py", title="Influencers Scores", icon=":material/add_circle:")

pg = st.navigation([graph_page, product, influencers])
st.set_page_config(page_title="Find Tiktok Influencers", page_icon=":material/edit:")
pg.run()