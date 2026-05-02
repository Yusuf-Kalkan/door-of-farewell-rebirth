from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# New generation client setup
client = genai.Client(api_key=API_KEY)

def get_emotion_intensity(lyric):
    """
    Sends the lyric to Gemini and extracts emotion intensity (between 0.0 and 1.0).
    """
    prompt = f"""
    You are an AI that deeply analyzes the sociological and emotional atmosphere of 1973 America (Vietnam War fatigue, collapse of counterculture).
    Analyze the following lyric from Bob Dylan's 'Knockin' on Heaven's Door': "{lyric}"

    Numerically evaluate the intensity of 'grief, farewell, or surrender' in this lyric between 0.0 and 1.0.
    0.0 = Rebellion, freedom, dynamism
    1.0 = Deep grief, farewell, fatigue, surrender

    Output ONLY in the following JSON format:
    {{
        "lyric": "{lyric}",
        "emotion": "grief", 
        "intensity": 0.92,
        "historical_note": "A short explanation in the context of 1973"
    }}
    """

    try:
        # JSON output format suitable for the new package
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        # Return neutral (0.5) value to prevent crash on connection loss or error
        return {"lyric": lyric, "emotion": "neutral", "intensity": 0.5, "historical_note": "Error: Connection failed"}

# Small block to test the code
if __name__ == "__main__":
    test_lyric = "Mama, take this badge off of me"
    print("Connecting to Gemini API and analyzing lyric...\n")
    result = get_emotion_intensity(test_lyric)
    print(json.dumps(result, indent=4, ensure_ascii=False))