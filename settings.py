web_scrapper_setting = {
    "startUrls": [],
    "useSitemaps": False,
    "respectRobotsTxtFile": True,
    "crawlerType": "playwright:adaptive",
    
    # Crawl settings
    "maxCrawlDepth": 5,             # Only go a few levels deep, adjust if needed
    "maxCrawlPages": 50,            # Limit pages to avoid unnecessary scraping
    "initialConcurrency": 0,
    "maxConcurrency": 10,           # Moderate concurrency for stability
    
    # Content extraction
    "htmlTransformer": "readableText",  # Extract readable text
    "readableTextCharThreshold": 50,    # Ignore very short texts (like buttons)
    "aggressivePrune": True,            # Remove unnecessary text fragments
    "removeElementsCssSelector": """nav, footer, script, style, noscript, svg, img,
        [role="alert"], [role="banner"], [role="dialog"], 
        [role="alertdialog"], [role="region"][aria-label*="skip" i], [aria-modal="true"]""",
    "removeCookieWarnings": True,
    "blockMedia": True,               # Ignore images/videos
    "expandIframes": False,           # Avoid iframe content for now
    "clickElementsCssSelector": "",   # Don't click dynamic elements
    
    # Saving output
    "saveMarkdown": True,             # Save readable text in markdown
    "saveHtml": False,
    "saveFiles": False,
    "saveScreenshots": False,
    
    # Requests and retries
    "maxRequestRetries": 2,
    "requestTimeoutSecs": 60,
    
    # Dataset
    "maxResults": 100,                # Limit results
}

influencers_scrapping_setting = {
    "resultsPerPage": 100,

    "searchQueries": [],
    "searchSection": "/video",

    "shouldDownloadSubtitles": True,

    "proxyCountryCode": "TH",
}