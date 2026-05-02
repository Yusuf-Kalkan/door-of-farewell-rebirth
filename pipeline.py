import os
from dotenv import load_dotenv
from emotion import get_emotion_intensity
import build_n_grams
import dpt_to_midi

# Load API key from .env file
load_dotenv()

def main():
    print("==================================================")
    print("        DOOR OF FAREWELL - REBIRTH (PIPELINE)     ")
    print("==================================================\n")

    # Lyrics of the song (Emotion map will be extracted from these lines)
    lyrics = [
        "Mama, take this badge off of me",
        "I can't use it anymore",
        "It's gettin' dark, too dark to see",
        "I feel like I'm knockin' on heaven's door",
        "Mama, put my guns in the ground",
        "I can't shoot them anymore"
    ]

    # Train the N-Gram Model (We only need to train it once)
    print("[SYSTEM] N-Gram Model is learning the song's DNA...")
    seqs = build_n_grams.load_dpt_sequences("./json_output")
    
    if not seqs:
        print("ERROR: .dpt.json file not found in ./json_output directory!")
        return
        
    models = build_n_grams.build_counts(seqs)
    D_vocab, P_vocab, T_vocab = models["vocab"]["D"], models["vocab"]["P"], models["vocab"]["T"]
    print(" -> Learning completed.\n")

    # Main lists to store notes generated from all lyrics
    final_D, final_P, final_T = [], [], []
    
    # Initial history for N-gram
    D_prev2, D_prev1 = "D_START", "D_START"
    P_prev2, P_prev1 = "P_START", "P_START"
    T_prev1 = "T_START"

    # Start loop for each lyric
    for index, lyric in enumerate(lyrics):
        print(f"--- SECTION {index + 1} / {len(lyrics)} ---")
        print(f"Lyric: '{lyric}'")
        
        emotion_data = get_emotion_intensity(lyric)
        intensity = emotion_data['intensity']
        print(f"Emotion: {emotion_data['emotion'].upper()} | Intensity: {intensity}")

        # Calculate Alpha (Chaos) Coefficient
        alpha_val = 0.1 + ((1.0 - intensity) * 4.9)
        print(f"Section Alpha Coefficient: {alpha_val:.2f}\n")

        # Generate 16 notes (segment) for this lyric
        segment_note_count = 16 
        
        for i in range(segment_note_count):
            d_dist = build_n_grams.get_D_distribution(models["d_counts"], models["d_unigram"], D_vocab, D_prev2, D_prev1, alpha=alpha_val)
            d_t = build_n_grams.sample_from_dist(d_dist)

            p_dist = build_n_grams.get_P_distribution(models["p_counts"], models["p_unigram"], P_vocab, P_prev2, P_prev1, d_t, alpha=alpha_val)
            p_t = build_n_grams.sample_from_dist(p_dist)

            t_dist = build_n_grams.get_T_distribution(models["t_counts"], models["t_unigram"], T_vocab, T_prev1, p_t, d_t, alpha=alpha_val)
            t_t = build_n_grams.sample_from_dist(t_dist)

            final_D.append(d_t)
            final_P.append(p_t)
            final_T.append(t_t)

            # Update history
            D_prev2, D_prev1 = D_prev1, d_t
            P_prev2, P_prev1 = P_prev1, p_t
            T_prev1 = t_t

    # Combine all generated segments into a single MIDI file
    print("\n[SYSTEM] Combining all segments and saving MIDI...")
    seq_dict = {
        "tpq": seqs[0]["tpq"],
        "tempo_bpm": 120, # Naming secured
        "bpm": 120,       # Both added to prevent conflicts
        "D": final_D,
        "P": final_P,
        "T": final_T
    }

    output_file = "Door_of_Farewell.mid"
    dpt_to_midi.build_midi_from_sequence(seq_dict, output_file)
    print(f"DONE! New music file created as '{output_file}'.")

if __name__ == "__main__":
    main()