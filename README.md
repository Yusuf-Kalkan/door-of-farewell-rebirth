# Door of Farewell - Rebirth (Knock! Design Your Door)

**CSE 358 Introduction to Artificial Intelligence - Creative Project Assignment**

This project reimagines Bob Dylan's 1973 song *Knockin' on Heaven's Door* through generative AI techniques. The historical and emotional weight of the song is directly integrated into the parameters of the music generation algorithm, dictating how safely or chaotically the machine interprets the original piece.

---

## 1. System Architecture

The system operates through four integrated stages:

1. **Data Preparation:** The original MIDI file is parsed and converted into a machine-learning-friendly DPT (Duration, Pitch, Technique) format.
2. **Contextual Emotion Analysis (AI Technique 1):** A line from the song is analyzed via an LLM (Google Gemini 2.5 Flash API) within the sociological context of 1973 America. It outputs an intensity score between `0.0` (Rebellion/Freedom) and `1.0` (Grief/Surrender).
3. **Parametric Generation (AI Technique 2):** This emotion intensity is inversely mapped to the $\alpha$ (Laplace Smoothing) parameter of our Generative N-gram model. High grief results in strict adherence to the original melody, while low grief (rebellion) introduces musical chaos.
4. **Output Synthesis:** The newly generated DPT sequence is synthesized back into a playable MIDI file.

---

## 2. File Structure

- **`pipeline.py`:** The main orchestrator. It manages the API calls, calculates the $\alpha$ value, trains the N-gram model, and generates the final sequence.
- **`emotion.py`:** A module utilizing the Google Gemini API with strict prompt engineering to analyze lyrics and output contextual emotion data in JSON format.
- **`midi_to_dpt.py`:** Extracts notes from the raw MIDI file and converts them into the DPT format (`.dpt.json`).
- **`build_n_grams.py`:** A 2nd-order Markov Chain (N-gram) model that maps the probability distribution of notes.
- **`dpt_to_midi.py`:** Synthesizes the generated DPT lists back into an industry-standard `.mid` file.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.8 or higher.
- A valid [Google Gemini API Key](https://aistudio.google.com/app/apikey).

### Step 1: Install Dependencies
Install the required Python libraries using `pip`:

```bash
pip install mido python-dotenv google-genai
```

### Step 2: Environment Variables (Crucial)
To securely use the Gemini API without hardcoding the key, create a file named `.env` in the root directory of the project and add your API key:

```env
GEMINI_API_KEY="your_api_key_here"
```

---

## 4. Execution

To run the project from scratch, follow these steps in your terminal:

### Step 1: Process the Original MIDI
Convert the raw MIDI file into the training format. Ensure your original MIDI is inside the `midi_input` folder.

```bash
python midi_to_dpt.py ./midi_input ./json_output
```

### Step 2: Run the Generation Pipeline
Execute the main script. This will analyze the lyrics, set the chaos parameters, and generate the new song:

```bash
python pipeline.py
```

The final generated track will be saved in the root directory as `Door_of_Farewell.mid`.

---

## 5. Example Output

When running `pipeline.py`, you should expect a terminal output similar to this:

```text
==================================================
        DOOR OF FAREWELL - REBIRTH (PIPELINE)     
==================================================

[SYSTEM] N-Gram Model is learning the song's DNA...
 -> Learning completed.

--- SECTION 1 / 6 ---
Lyric: 'Mama, take this badge off of me'
Emotion: GRIEF | Intensity: 0.92
Section Alpha Coefficient: 0.49

--- SECTION 2 / 6 ---
Lyric: 'I can't use it anymore'
Emotion: SURRENDER | Intensity: 0.85
Section Alpha Coefficient: 0.83

...

[SYSTEM] Combining all segments and saving MIDI...
DONE! New music file created as 'Door_of_Farewell.mid'.
```

## 6. License

This project is submitted as academic coursework for CSE 358. The code is available under the MIT License. You are free to use, modify, and distribute it, provided that proper attribution is given.