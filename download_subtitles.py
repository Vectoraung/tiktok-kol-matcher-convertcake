import re
import requests

def get_subtitles_text(url):
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        text = response.text

        # Clean WEBVTT text
        text = re.sub(r"^WEBVTT.*\n+", "", text)
        text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}", "", text)
        text = re.sub(r"\n+", "\n", text).strip()
        text = text.replace("...", "…")
        text = text.replace("\n", " ")
        
        return text

    except Exception as e:
        print(f"❌ Failed to process: {e}")
