from settings import web_scrapper_setting

from typing import TypedDict
from dotenv import load_dotenv
from apify_client import ApifyClient
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
import copy
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from translate import translate_list_async

load_dotenv()

class ClientExtractionState(TypedDict):
    website_url: str

    client_page_scrape_run_id: str | None
    scrapped_posts_dataset_id: str | None

    analyzed_data_use_cache: bool
    product_category_use_cache: bool
    search_queries_use_cache: bool
    influencers_scores_use_cache: bool
    
    scrapped_data: list
    cleaned_data: list
    chunks: list
    
    product_category: dict
    final_analysis: dict
    search_queries: dict
    analyzed_data: list
    scrapped_posts: list
    scrapped_posts_cleaned: str

    performace_scores: list

class SummarizedChunk(BaseModel):
    summary: str = Field(..., description="Summary of the text chunk")

class TargetAudience(BaseModel):
    gender: str = Field(..., description="Target gender in one word")
    age_range: str = Field(..., description="Target age range (e.g., 18-24)")
    location: str = Field(..., description="Target location (e.g., New York)")
    interest: List[str] = Field(..., description="Target interest (e.g., fitness)")

class MarketingAnalysisFormat(BaseModel):
    product_name: str = Field(..., description="Name of product")
    main_category: str = Field(..., description="Main category that the product falls under")
    subcategory_list: List[str] = Field(..., description="List of sub-categories that the product falls under", max_length=3, min_length=1)
    description: str = Field(..., description="Product description in 1–2 sentences")
    unique_selling_points: str = Field(..., description="Unique selling points")
    target_audience: TargetAudience = Field(..., description="Demographics & interests")
    brand_personality: List[str] = Field(..., description="Main 4 tones or styles of this brand/product", max_length=4, min_length=1)

class ProductCategory(BaseModel):
    main_category: str = Field(..., description="Main category that the product falls under")
    subcategory_list: List[str] = Field(..., description="List of sub-categories that the product falls under", max_length=3, min_length=1)

class Query(BaseModel):
    search_query: str
    translation_to_eng: str

class SearchQuery(BaseModel):
    category: str = Field(..., description="One of the input product categories")
    product_placement_in_content: Query = Field(..., description="To search videos with product placement in content style")
    unboxing: Query = Field(..., description="To search videos with unboxing/ reveal style")
    reviews: Query = Field(..., description="To search videos with reviews/ recommendations style")
    lifestyle: Optional[Query] = Field(None, description="To search videos with Lifestyle/ day-in-the-life style")

class SearchQueries(BaseModel):
    search_queries: List[SearchQuery] = Field(..., description="List of search queries")    

def start_scrapping(state: ClientExtractionState):
    writer = get_stream_writer()
    run_id = state.get("client_page_scrape_run_id", None)

    if run_id is not None:
        writer(
            {
                "message": "Pulling scrapped data from DB...",
                "state": state
            }
        )
        client = ApifyClient(os.getenv("APIFY_API_TOKEN"))
        run = client.run(run_id).get()
        dataset_id = run["defaultDatasetId"]
        run_id = run['id']
        scrapped_data = client.dataset(dataset_id).list_items().items
        state["scrapped_data"] = scrapped_data
        state['client_page_scrape_run_id'] = run_id
        writer(
            {
                "message": f"Finished pulling scrapped data from DB: {len(scrapped_data)} items...",
                "state": state
            }
        )

        return state

    writer(
            {
                "message": "Start scrapping client's website data...",
                "state": state
            }
        )
    website_url = state.get("website_url", '')

    actor_setting = copy.deepcopy(web_scrapper_setting)
    actor_setting["startUrls"] = [{"url": website_url}]

    #client = ApifyClient("APIFY_API_TOKEN")
    client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

    run = client.actor("aYG0l9s7dbB7j3gbS").call(run_input=actor_setting)

    scrapped_data = client.dataset(run["defaultDatasetId"]).list_items().items
    state["scrapped_data"] = scrapped_data

    writer(
            {
                "message": "Finished scrapping client's website data...",
                "state": state
            }
        )

    return state

def clean_scrapped_data(state: ClientExtractionState):
    writer = get_stream_writer()
    writer(
            {
                "message": "Start cleaning scrapped client's website data...",
                "state": state
            }
        )
    scrapped_data = state.get("scrapped_data", [])

    cleaned_data = []

    for page in scrapped_data:
        text = page["text"].strip()
        title = page["metadata"]["title"].lower()

        if not text:
            continue

        if any(keyword in title for keyword in ["privacy", "terms", "policy", "faq"]):
            continue

        lines = text.splitlines()
        seen = set()
        deduped_lines = []
        for line in lines:
            line = line.strip()
            if line and line.lower() not in seen:
                deduped_lines.append(line)
                seen.add(line.lower())  # case-insensitive dedup
        text = "\n".join(deduped_lines)

        cleaned_data.append({"text": text, "title": title})

    writer(
            {
                "message": "Finished cleaning scrapped client's website data...",
                "state": state
            }
        )

    state["cleaned_data"] = cleaned_data

    return state

def group_cleaned_data(state: ClientExtractionState):
    writer = get_stream_writer()
    writer(
            {
                "message": "Start splitting chunks to website data...",
                "state": state
            }
        )
    cleaned_data = state.get("cleaned_data", [])

    chunks = []
    current_chunk = ""
    max_length = 3800

    for page in cleaned_data:
        title = page["title"].strip()
        text = page["text"].strip()
        position = 0

        # Keep adding text until it's fully processed
        while position < len(text):
            remaining_space = max_length - len(current_chunk)

            # If this page fits in the current chunk
            if len(title) + 2 + len(text) - position < remaining_space:
                # Add normally
                current_chunk += f"\n\n{title}\n{text[position:]}"
                position = len(text)
            else:
                # Fill remaining space with part of text
                chunk_part = text[position : position + remaining_space - len(title) - 25]
                current_chunk += f"\n\n{title}\n{chunk_part}...moved to next chunk"
                chunks.append(current_chunk.strip())
                
                # Prepare next chunk
                current_chunk = f"{title}\n...continued from previous chunk "
                position += len(chunk_part)

    # Add the final chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    state["chunks"] = chunks

    writer(
            {
                "message": f"Finished splitting chunks to website data: {len(chunks)} chunks...",
                "state": state
            }
        )
    return state

def check_end_chunk_analysis(state: ClientExtractionState):
    writer = get_stream_writer()
    chunks = state.get("chunks", [])
    analyzed_data = state.get("analyzed_data", [])

    if len(chunks) > len(analyzed_data):
        writer(
            {
                "message": "Continuing analysis to the next chunk.",
                "state": state
            }
        )
        return "Continue"
    else:
        path = "cache/analysis_chunks.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(analyzed_data, f, ensure_ascii=False, indent=4)

        writer(
            {
                "message": "Saved to cache and stopped analysis.",
                "state": state
            }
        )
        return "Stop"

def analyze_next_group(state: ClientExtractionState):
    writer = get_stream_writer()

    analyzed_data = state.get("analyzed_data", [])
    chunks = state.get("chunks", [])
    use_cache = state.get("analyzed_data_use_cache", False)

    path = "cache/analysis_chunks.json"

    if len(chunks) == 0:
        return state
    
    if use_cache:
        with open(path, "r", encoding="utf-8") as f:
            cached_chunks = json.load(f)
        state["analyzed_data"] = cached_chunks
        return state
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    llm = llm.with_structured_output(SummarizedChunk)

    current_chunk = chunks[len(analyzed_data)]

    messages = [
        (
            "system",
            """You are a marketing analyst summarizing a chunk of text scraped from a client’s product website.  
Write a 100–150 word analytical summary capturing the brand’s tone, target audience, key products or services, and marketing style or strategy.  
Use a professional yet natural tone that interprets meaning rather than copying text.

Continuity markers may appear:
- “…moved to next chunk” → the content continues in the next part.  
- “…continued from previous chunk” → this text resumes from an earlier part.  

Acknowledge these smoothly, e.g., “the section ends mid-thought, likely continuing in the next part” or “this part builds on earlier content.”  

Output strictly in a JSON object matching the SummarizedChunk schema.
""",
        ),
        ("human", current_chunk),
    ]

    output = llm.invoke(messages)

    analyzed_data.append(output.model_dump())

    state["analyzed_data"] = analyzed_data

    writer(
            {
                "message": f"Completed analyzing chunk {len(analyzed_data)}.",
                "state": state
            }
        )

    return state

def classify_product_category(state: ClientExtractionState):
    writer = get_stream_writer()
    analyzed_data_groups = state.get("analyzed_data", [])
    use_cache = state.get("product_category_use_cache", False)

    path = "cache/product_category.json"

    if use_cache:        
        with open(path, "r", encoding="utf-8") as f:
            cached_product_category = json.load(f)
        state["product_category"] = cached_product_category

        writer(
            {
                "message": f"Using cached product category from {path}.",
                "state": state
            }
        )
        return state

    input_text = "Product Description:\n\n"
    for g in analyzed_data_groups:
        input_text += g["summary"]+"\n\n"

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    llm = llm.with_structured_output(MarketingAnalysisFormat)

    messages = [
        (
            "system",
            """You are a senior marketing analyst tasked with producing a structured analysis of a client’s product website, to find tiktok influencers to collaborate with. 
You have multiple summarized paragraphs of the website text.

Your goal is to extract insights and generate the following structured information:

1. product_name: The official or short name of the product.
2. **Main category**: The broad, top-level category the product belongs to (e.g., Electronics, Beauty, Food & Beverage, Fashion).
3. **Subcategories**: The most specific categories that cannot be divided further, describing the product in 1–3 words each (e.g., Perfume, Spicy Food, Fitness Smartwatch). Include **at least 1 subcategory and no more than 3**.
4. description: A short paragraph summarizing what the product is, what it does, and why it matters.
5. unique_selling_points: Key strengths or differentiators.
6. target_audience_age: Defines the ideal customer group.
7. brand_personality: Describes the brand’s tone or image (e.g., “modern”, “friendly”, “luxury”).

Do not copy text verbatim; summarize insights intelligently and choose the most relevant words.  
Output strictly in a JSON object matching the FinaleAnalysis schema.""",
        ),
        ("human", input_text),
    ]

    output = llm.invoke(messages).model_dump()

    state["final_analysis"] = output

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    writer(
            {
                "message": f"Saved the classified categories to {path}.",
                "state": state
            }
        )

    return state

def generate_search_queries(state: ClientExtractionState):
    writer = get_stream_writer()
    analysis = state.get("final_analysis", {})
    use_cache = state.get("search_queries_use_cache", False)

    path = "cache/search_queries.json"

    if use_cache:
        with open(path, "r", encoding="utf-8") as f:
            search_queries = json.load(f)
        state["search_queries"] = search_queries

        writer(
            {
                "message": f"Using cached search queries from {path}.",
                "state": state
            }
        )
        return state

    input_text = "Product Categories: " + ", ".join([category for category in analysis['subcategory_list']])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    llm = llm.with_structured_output(SearchQueries).with_retry(stop_after_attempt=1)

    messages = [
        (
            "system",
            """You are an expert TikTok search assistant.

Your task is: Given a product categories, generate Thai search queries and hashtags to find TikTok videos of influencers featuring products in that category. Focus on videos where the influencer is presenting or using the product on-screen, not just links or external promotions.

For each category, using the following styles, provide Thai search query, along with a short translation in English for each query describing its purpose:

- product_placement_in_content: Influencer naturally uses the product in their daily life or routine, integrating it into content without a formal review.
- unboxing (Reveal): Influencer opens or shows the product for the first time, highlighting packaging and first impressions.
- reviews (Recommendations): Influencer shares their opinion or recommendation about the product’s quality, usefulness, or effectiveness.
- lifestyle (Day-in-the-Life): Influencer shows the product as part of a vlog or daily routine, integrating it naturally into their life.

Instructions:
- Generate search queries in Thai appropriate for TikTok influencer content.
- Queries should be category-focused, not tied to a specific brand or product.
- Keep translations short and concise, describing why this query helps find relevant videos.
- Return only the search queries and explanations, without any additional text or formatting.""",
        ),
        ("human", input_text),
    ]

    output = llm.invoke(messages).model_dump()

    state["search_queries"] = output

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    writer(
            {
                "message": f"Saved the search queries to {path}.",
                "state": state
            }
        )

    return state

def scrape_tiktok_posts(state: ClientExtractionState):
    writer = get_stream_writer()
    dataset_id = state.get("scrapped_posts_dataset_id", None)

    if dataset_id is None:
        search_queries = state.get("search_queries", [])['search_queries']

        max_results = 200
        search_queries_in_list = []
        for query_set in search_queries:
            search_queries_in_list.append(query_set['product_placement_in_content']['search_query'])
            search_queries_in_list.append(query_set['unboxing']['search_query'])
            search_queries_in_list.append(query_set['reviews']['search_query'])
            search_queries_in_list.append(query_set['lifestyle']['search_query'])

        # Prepare the Actor input
        run_input = {
            "excludePinnedPosts": True,
            "profileSorting": "popular",
            "resultsPerPage": int(max_results//(len(search_queries)*4)),
            "scrapeRelatedVideos": False,
            "searchQueries": search_queries_in_list,
            "searchSection": "/video",
            "shouldDownloadAvatars": False,
            "shouldDownloadCovers": False,
            "shouldDownloadMusicCovers": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadSubtitles": False,
            "shouldDownloadVideos": False
            }
        
        client = ApifyClient(os.getenv("APIFY_API_TOKEN"))
        run = client.actor("GdWCkxBtKWOsKjdch").call(run_input=run_input)
        dataset_id = run["defaultDatasetId"]
        state['scrapped_posts_dataset_id'] = dataset_id

    writer(
            {
                "message": "Finished scraping TikTok posts...",
                "state": state
            }
        )

    return state

def clean_scrapped_posts(state: ClientExtractionState):
    writer = get_stream_writer()
    dataset_id = state.get("scrapped_posts_dataset_id", None)
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?format=csv&fields=id%2CauthorMeta%2CcollectCount%2CcommentCount%2CdiggCount%2Cid%2CplayCount%2CshareCount%2Ctext%2CtextLanguage%2CvideoMeta%2CsearchQuery&clean=true&attachment=true"

    df = pd.read_csv(url)

    df = df[df['textLanguage'].str.contains('th', na=False)].reset_index(drop=True)

    subtitle_lang_cols = [col for col in df.columns if col.startswith('videoMeta/subtitleLinks/') and col.endswith('/language')]
    df = df[df[subtitle_lang_cols].apply(lambda row: row.astype(str).str.contains('tha-TH').any(), axis=1)].reset_index(drop=True)

    def clean_text(text):
        if pd.isna(text):
            return text
        # Remove URLs (http, https, www)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "\U00002B00-\U00002BFF"
            "\U0000200D"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        # Optional: remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Apply the cleaning function to the column
    df['authorMeta/signature'] = df['authorMeta/signature'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)

    # Group by 'authorMeta/id' and calculate required metrics
    agg_df = (
        df.groupby('authorMeta/id', as_index=False)
        .agg(
            videoApperanceCount=('authorMeta/id', 'count'),
            videosDiggCountAvg=('diggCount', 'mean'),
            videosShareCountAvg=('shareCount', 'mean'),
            videosPlayCountAvg=('playCount', 'mean'),
            videosCollectCountAvg=('collectCount', 'mean'),
            videosCommentCountAvg=('commentCount', 'mean'),
            videoDurationAvg=('videoMeta/duration', 'mean'),
            videosCaptionsCombinedText=('text', lambda x: ' '.join(x.dropna().astype(str))),
            combinedTextLanguage=('textLanguage', lambda x: ' '.join(sorted(set(x.dropna().astype(str)))))
        )
    )

    # Merge aggregated data back
    df = df.merge(agg_df, on='authorMeta/id', how='left')

    # Drop duplicated rows based on 'authorMeta/id', keeping the first
    df = df.drop_duplicates(subset='authorMeta/id', keep='first').reset_index(drop=True)

    df = df[[
        'videoApperanceCount',
        'videosDiggCountAvg',
        'videosShareCountAvg',
        'videosPlayCountAvg',
        'videosCollectCountAvg',
        'videosCommentCountAvg',
        'videoDurationAvg',
        'videosCaptionsCombinedText',
        'combinedTextLanguage',
        'authorMeta/digg',
        'authorMeta/fans',
        'authorMeta/following',
        'authorMeta/friends',
        'authorMeta/heart',
        'authorMeta/id',
        'authorMeta/name',
        'authorMeta/nickName',
        'authorMeta/signature',
        'authorMeta/verified',
        'authorMeta/video',
        'authorMeta/profileUrl'
    ]]

    df.to_csv("cache/scrapped_posts_cleaned.csv", index=False)

    state['scrapped_posts_cleaned'] = df.to_csv()

    writer(
            {
                "message": "Finished cleaning scrapped TikTok posts...",
                "state": state
            }
        )

    return state

async def calculate_scores(state: ClientExtractionState):
    writer = get_stream_writer()
    use_cache = state.get("influencers_scores_use_cache", False)
    if use_cache:
        with open('cache/influencers_scores.json', "r", encoding="utf-8") as f:
            data = json.load(f)

        state['influencers_scores'] = data

        writer(
            {
                "message": "Start using cached calculated influencers scores...",
                "state": state
            }
        )
        return state
        
    analysis = state.get("final_analysis", {})

    scrapped_posts_csv = "cache/scrapped_posts_cleaned.csv"
    df = pd.read_csv(scrapped_posts_csv)

    brand_tone_descriptions = analysis["brand_personality"]
    product_keywords = analysis["subcategory_list"]

    brand_tone_descriptions = await translate_list_async(brand_tone_descriptions)
    product_keywords = await translate_list_async(product_keywords)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # --- 1. Engagement Performance ---
    df['engagement_rate'] = (df['videosDiggCountAvg'] + df['videosShareCountAvg'] + df['videosCommentCountAvg']) / (df['videosPlayCountAvg'] + 1)
    df['collect_rate'] = df['videosCollectCountAvg'] / (df['videosPlayCountAvg'] + 1)
    df['engagement_score_raw'] = (df['engagement_rate'] + df['collect_rate']) / 2

    # --- 2. Content Productivity & Consistency ---
    df['videos_per_fan'] = df['authorMeta/video'] / (df['authorMeta/fans'] + 1)
    df['hearts_per_video'] = df['authorMeta/heart'] / (df['authorMeta/video'] + 1)
    df['productivity_score_raw'] = (df['videos_per_fan'] + (df['hearts_per_video'] / 1000)) / 2

    # --- 3. Audience Reach Efficiency ---
    df['avg_views_per_fan'] = df['videosPlayCountAvg'] / (df['authorMeta/fans'] + 1)
    df['digg_per_fan'] = df['videosDiggCountAvg'] / (df['authorMeta/fans'] + 1)
    df['reach_efficiency_score_raw'] = (df['avg_views_per_fan'] + df['digg_per_fan']) / 2

    # --- Normalize each score between 0–1 ---
    scaler = MinMaxScaler()
    for col in ['engagement_score_raw', 'productivity_score_raw', 'reach_efficiency_score_raw']:
        df[col.replace('_raw', '')] = scaler.fit_transform(df[[col]])

    # --- Optional: compute overall score (average of the three topics) ---
    df['performance_score'] = df[['engagement_score', 'productivity_score', 'reach_efficiency_score']].mean(axis=1)

    # --- 2. Category Match Score ---
    # Compute embeddings
    influencer_text_embeddings = model.encode(df['videosCaptionsCombinedText'].tolist(), convert_to_numpy=True)
    product_keywords_embedding = model.encode([" ".join(product_keywords)], convert_to_numpy=True)

    # Cosine similarity
    category_sim = cosine_similarity(influencer_text_embeddings, product_keywords_embedding)
    df['category_match_score'] = category_sim.flatten()  # 0-1 similarity

    # --- 3. Tone & Style Match Score ---
    # Use influencer signature + combined text as representation
    df['tone_text'] = df['authorMeta/signature'].fillna('') + " " + df['videosCaptionsCombinedText']

    influencer_tone_embeddings = model.encode(df['tone_text'].tolist(), convert_to_numpy=True)
    brand_tone_embedding = model.encode([', '.join(brand_tone_descriptions)], convert_to_numpy=True)

    tone_sim = cosine_similarity(influencer_tone_embeddings, brand_tone_embedding)
    df['tone_match_score'] = tone_sim.flatten()  # 0-1 similarity

    # --- Optional: inspect top matches ---
    df_sorted = df.sort_values(by=['category_match_score', 'tone_match_score', 'performance_score'], ascending=False)

    # --- Get top 10 influencers ---
    top_influencers_df = df_sorted[['authorMeta/id', 
                                    'authorMeta/nickName',
                                    'authorMeta/fans',
                                    'authorMeta/following',
                                    'authorMeta/friends',
                                    'authorMeta/heart',
                                    'authorMeta/id',
                                    'authorMeta/name',
                                    'authorMeta/signature',
                                    'authorMeta/verified',
                                    'authorMeta/video', 
                                    'engagement_score', 
                                    'productivity_score', 
                                    'reach_efficiency_score', 
                                    'performance_score', 
                                    'category_match_score',
                                    'tone_match_score',
                                    'authorMeta/profileUrl']]

    # --- Convert to dictionary ---
    top_influencers_dict = top_influencers_df.to_dict(orient='records')

    state['influencers_scores'] = top_influencers_dict

    with open("cache/influencers_scores.json", "w", encoding="utf-8") as f:
        json.dump(top_influencers_dict, f, ensure_ascii=False, indent=4)
    
    writer(
            {
                "message": "Finished calculating influencers scores and saved to cache...",
                "state": state
            }
        )

    return state

builder = StateGraph(ClientExtractionState)

builder.add_node("scrape", start_scrapping)
builder.add_node("clean", clean_scrapped_data)
builder.add_node("group", group_cleaned_data)
builder.add_node("analyze", analyze_next_group)
builder.add_node("classify_category", classify_product_category)
builder.add_node("generate_search_queries", generate_search_queries)
builder.add_node("scrape_tiktok_posts", scrape_tiktok_posts)
builder.add_node("clean_scrapped_posts", clean_scrapped_posts)
builder.add_node("calculate_scores", calculate_scores)
#builder.add_node("calculate_performance_score", calculate_performance_score)

builder.add_edge(START, "scrape")
builder.add_edge("scrape", "clean")
builder.add_edge("clean", "group")
builder.add_edge("group", "analyze")
builder.add_conditional_edges(
    "analyze",
    check_end_chunk_analysis,
    {
        "Continue": "analyze",
        "Stop": "classify_category"
    }
)
builder.add_edge("classify_category", "generate_search_queries")
builder.add_edge("generate_search_queries", "scrape_tiktok_posts")
builder.add_edge("scrape_tiktok_posts", "clean_scrapped_posts")
builder.add_edge("clean_scrapped_posts", "calculate_scores")
builder.add_edge("calculate_scores", END)
#builder.add_edge("calculate_performance_score", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)