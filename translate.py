import asyncio
from googletrans import Translator


async def translate_list_async(captions, src='th', dest='en'):
    translator = Translator()
    translated = []

    for cap in captions:
        try:
            # directly await the async translate coroutine
            result = await translator.translate(cap, src=src, dest=dest)
            translated.append(result.text)
        except Exception as e:
            print(f"⚠️ Failed to translate: {cap}, error: {e}")
            translated.append(cap)  # fallback to original
    return translated

'''# Example usage
async def main():
    captions = ["สวัสดี", "Hello", "ฉันรัก programming"]
    translated = await translate_list_async(captions)
    print(translated)

asyncio.run(main())
'''