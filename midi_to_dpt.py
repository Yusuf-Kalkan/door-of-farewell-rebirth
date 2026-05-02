"""
midi_to_dpt.py

Convert .mid files (one phrase/sequence per file) to D -> P -> T sequences
and write one JSON per input file.

Dependencies:
    pip install mido

Usage:
    python midi_to_dpt.py /path/to/mid/folder /path/to/output/folder
"""

import os
import json
import math
import sys
from collections import defaultdict, namedtuple
import mido

# ---------------------
# USER / PROJECT CONFIG
# ---------------------
# Minimum pitch considered a real note (C#2 = 37). Anything < 37 is an articulation / marker.
MIN_PITCH_NOTE = 37 - 12

# Technique MIDI-note mapping (articulations)
TECH_NOTE_TO_NAME = {
    24: "normal",       # C1
    25: "harmonic",     # C#1
    26: "palm_mute",    # D1
    28: "slide",        # E1
    29: "hammer",       # F1
    30: "tap"           # F#1
}

# Priority order when multiple techniques overlap on an event
TECH_PRIORITY = ["palm_mute", "harmonic", "slide", "hammer", "tap", "normal"]

# ticks-per-beat expected (but we'll use midi.ticks_per_beat from file)
# Your files use 96 ticks-per-beat but we'll read it directly to be robust.

# Duration tokens we support (expressed as multiplier of quarter-note = TPB)
DURATION_TOKENS = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "eighth_triplet": 1.0/3.0,     # one third of a quarter
    "dotted_eighth": 0.75,
    "sixteenth": 0.25,
    "thirtysecond": 0.125
}

# Acceptable tolerance when matching duration to nearest token (fraction of quarter)
DURATION_MATCH_TOLERANCE = 0.15  # 15% of quarter duration

# Default velocity if not present or static
DEFAULT_VELOCITY = 90

# ---------------------
# Helper functions
# ---------------------
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_name(n):
    """Return scientific name for midi note number n (e.g. 60 -> 'C4')."""
    octave = (n // 12) - 1
    name = NOTE_NAMES[n % 12]
    return f"{name}{octave}"

def nearest_duration_token(duration_ticks, tpq):
    """Map absolute ticks to a duration token string by converting ticks -> quarter multiples."""
    quarter_ticks = float(tpq)
    frac_quarter = duration_ticks / quarter_ticks
    best_token = None
    best_diff = float('inf')
    for token, qmult in DURATION_TOKENS.items():
        diff = abs(qmult - frac_quarter)
        if diff < best_diff:
            best_diff = diff
            best_token = token
    # enforce tolerance: if too far, fallback to closest but mark as "other_<ticks>"
    if best_diff > DURATION_MATCH_TOLERANCE:
        return f"other_{int(round(duration_ticks))}"
    return best_token

def pick_technique(tech_list):
    """Pick a single technique from a list according to priority. tech_list may be empty."""
    if not tech_list:
        return "normal"
    for t in TECH_PRIORITY:
        if t in tech_list:
            return t
    # fallback
    return tech_list[0]

# ---------------------
# Core conversion logic
# ---------------------
NoteEvent = namedtuple("NoteEvent", ["note", "start_tick", "duration_ticks", "velocity", "techniques"])



def raw_midi_summary(mid: mido.MidiFile):
    event_list = []
    for i, track in enumerate(mid.tracks):
        t = 0
        for msg in track:
            t += msg.time
            event_list.append((t, msg, i))
    event_list.sort(key=lambda x: x[0])

    note_on_msgs = [(t, msg.channel if hasattr(msg, 'channel') else None, msg.note, msg.velocity)
                    for (t,msg,_) in event_list if msg.type == 'note_on']
    note_off_msgs = [(t, msg.channel if hasattr(msg, 'channel') else None, msg.note, getattr(msg, 'velocity', None))
                     for (t,msg,_) in event_list if msg.type == 'note_off' or (msg.type=='note_on' and msg.velocity==0)]
    print("Total messages:", len(event_list))
    if note_on_msgs:
        notes = [n for (_,_,n,_) in note_on_msgs]
        print("note_on count:", len(note_on_msgs), "min_note_on:", min(notes), "max_note_on:", max(notes))
    else:
        print("No note_on messages at all.")
    print("note_off count:", len(note_off_msgs))

    # show pitchwheel presence
    pw = [(t, msg) for (t,msg,_) in event_list if msg.type.startswith('pitch')]
    print("pitchwheel/control messages present:", len(pw))
    if pw:
        print("Sample pitch/meta msgs (first 20):")
        for t,msg in pw[:20]:
            print(t, msg)




def parse_midi_to_events(mid: mido.MidiFile):
    """
    Parse mido MidiFile and return:
      - tpq (ticks per beat)
      - tempo in microseconds per quarter (or None)
      - events_by_onset: dict mapping onset_tick -> list of NoteEvent (only for pitch >= MIN_PITCH_NOTE)
    Articulation notes (< MIN_PITCH_NOTE) are interpreted as technique markers at their onset tick.
    """

    tpq = mid.ticks_per_beat
    # tempo: if there's a set_tempo message, take the first one; else default 500000 (120bpm)
    tempo = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo is not None:
            break
    if tempo is None:
        tempo = 500000  # default 120 BPM

    # We'll collect start times for notes keyed by (note, channel)
    open_notes = defaultdict(list)  # key: (note, channel) -> list of (start_tick, velocity)
    tech_markers = defaultdict(list)  # onset_tick -> list of technique names

    # Build global event list with absolute ticks (per-track accumulation then global sort)
    event_list = []
    for i, track in enumerate(mid.tracks):
        t = 0
        for msg in track:
            t += msg.time
            # store channel if present (some meta messages lack channel)
            event_list.append((t, msg))

    # sort globally by absolute tick
    event_list.sort(key=lambda x: x[0])

    last_tick = event_list[-1][0] if event_list else 0
    yielded_count = 0

    for abs_tick, msg in event_list:
        # normalize channel presence
        channel = getattr(msg, "channel", None)

        if msg.type == 'note_on' and msg.velocity > 0:
            n = msg.note
            if n < MIN_PITCH_NOTE:
                # technique marker (only if mapped)
                if n in TECH_NOTE_TO_NAME:
                    tech_markers[abs_tick].append(TECH_NOTE_TO_NAME[n])
                else:
                    # otherwise ignore as technique
                    pass
            else:
                # store start keyed by (note, channel) so identical notes on different channels don't clash
                key = (n, channel)
                open_notes[key].append((abs_tick, msg.velocity or DEFAULT_VELOCITY))
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            n = msg.note
            key = (n, channel)
            if key in open_notes and open_notes[key]:
                # use LIFO matching (last-on matches this off)
                start_tick, velocity = open_notes[key].pop()
                duration_ticks = max(0, abs_tick - start_tick)
                if n >= MIN_PITCH_NOTE:
                    techniques = tech_markers.get(start_tick, [])
                    yield NoteEvent(
                        note=n,
                        start_tick=start_tick,
                        duration_ticks=duration_ticks,
                        velocity=velocity or DEFAULT_VELOCITY,
                        techniques=techniques
                    )
                    yielded_count += 1
            else:
                # unmatched note_off (could be due to missing on or channel mismatch) -> ignore or optionally warn
                # example: print("Unmatched note_off", n, "ch", channel, "at", abs_tick)
                pass
        else:
            # other message: ignore
            pass

    # Flush any remaining open notes (notes without an explicit note_off)
    # Use last_tick as the end time if possible; otherwise give a small default duration.
    for key, starts in list(open_notes.items()):
        n, ch = key
        while starts:
            start_tick, velocity = starts.pop()
            # assume the note lasted until end of file
            duration_ticks = max(1, last_tick - start_tick)
            if n >= MIN_PITCH_NOTE:
                techniques = tech_markers.get(start_tick, [])
                yield NoteEvent(
                    note=n,
                    start_tick=start_tick,
                    duration_ticks=duration_ticks,
                    velocity=velocity or DEFAULT_VELOCITY,
                    techniques=techniques
                )
                yielded_count += 1

    # (optional) debugging summary - uncomment when debugging
    # print(f"parse_midi_to_events: total events in file={len(event_list)}, yielded_notes={yielded_count}, last_tick={last_tick}")

    # function is a generator; it yields NoteEvent objects


"""
def parse_midi_to_events(mid: mido.MidiFile):

    tpq = mid.ticks_per_beat
    # tempo: if there's a set_tempo message, take the first one; else default 500000 (120bpm)
    tempo = None
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo is not None:
            break
    if tempo is None:
        tempo = 500000  # default 120 BPM

    # We'll collect start times for notes (note number -> list of start ticks,
    # in case overlapping same note numbers occur)
    open_notes = defaultdict(list)  # key: note number -> list of start_tick entries
    # technique markers: onset_tick -> list of technique names (strings)
    tech_markers = defaultdict(list)
    # (optional) string_markers = dict but we'll ignore them per your instruction

    # First pass: collect start/end ticks for every note (including articulation notes)
    absolute_tick = 0
    # We'll iterate through all tracks merging events by absolute time:
    # mido does not automatically merge tracks' times when iterating separately, so we iterate track by track but we can
    # combine by adding running ticks per track. Simpler: use mid.tracks combined by iterating track messages with their deltas,
    # pushing into a global event list, then sort by absolute tick.
    event_list = []  # each: (abs_tick, msg)
    for i, track in enumerate(mid.tracks):
        t = 0
        for msg in track:
            t += msg.time
            event_list.append((t, msg))
    # sort globally
    event_list.sort(key=lambda x: x[0])

    # Now iterate global events
    for abs_tick, msg in event_list:
        if msg.type == 'note_on' and msg.velocity > 0:
            n = msg.note
            # Is articulation / technique?
            if n < MIN_PITCH_NOTE:
                # If it's recognized technique note, add to markers
                if n in TECH_NOTE_TO_NAME:
                    tech_markers[abs_tick].append(TECH_NOTE_TO_NAME[n])
                else:
                    # treat as a string marker or other articulation -> ignore as technique
                    pass
            else:
                # Real note: push its start time (store velocity too)
                open_notes[n].append((abs_tick, msg.velocity))
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            n = msg.note
            # find the most recent start for this note (LIFO)
            if n in open_notes and open_notes[n]:
                start_tick, velocity = open_notes[n].pop(0)  # use FIFO matching (first-on first-off)
                duration_ticks = abs_tick - start_tick
                # store as event in a structure keyed by onset tick (we'll only retain pitch >= MIN_PITCH_NOTE)
                # for articulation notes, we don't store as musical events here
                if n >= MIN_PITCH_NOTE:
                    # attach techniques that exist at the same onset tick if any
                    techniques = tech_markers.get(start_tick, [])
                    yield NoteEvent(note=n, start_tick=start_tick, duration_ticks=duration_ticks, velocity=velocity or DEFAULT_VELOCITY, techniques=techniques)
            else:
                # unmatched note_off (ignore or warn)
                pass
"""
def build_dpt_sequence_from_events(mid, events_iter, quantize_grid=None):
    """
    Build aligned D, P, T lists from NoteEvent iterator for a single midi file.
    Each input .mid file is treated as one independent sequence (one phrase).
    quantize_grid: if provided, quantize onset ticks to nearest multiple of quantize_grid (default: tpq/4 -> 16th).
    Returns a dict with fields: tpq, tempo, onsets (ticks), D, P, T
    """
    tpq = mid.ticks_per_beat
    # default quantize to sixteenth grid
    if quantize_grid is None:
        quantize_grid = tpq // 4

    # collect events grouped by (quantized) onset
    grouped = defaultdict(list)  # onset_tick -> list of NoteEvent
    # events_iter yields NoteEvent namedtuples
    for ev in events_iter:
        q_on = int(round(ev.start_tick / quantize_grid)) * quantize_grid
        grouped[q_on].append(ev)

    if not grouped:
        return None  # empty file / nothing to convert

    ticks_sorted = sorted(grouped.keys())
    onsets = []
    D = []
    P = []
    T = []

    for onset in ticks_sorted:
        evs = grouped[onset]
        # Filter out any events erroneously below MIN_PITCH (shouldn't happen)
        evs = [e for e in evs if e.note >= MIN_PITCH_NOTE]
        if len(evs) == 0:
            # no pitch at this onset (unlikely) -> produce a REST event?
            # We'll skip, or create an explicit REST if the gap from previous onset is > 0
            # To preserve rhythm, create a REST with default 1 sixteenth
            onsets.append(onset)
            D.append(nearest_duration_token(quantize_grid, tpq))
            P.append("REST")
            T.append("_")
            continue

        # chord if multiple notes, single note otherwise
        if len(evs) == 1:
            # single note
            e = evs[0]
            # duration choice: use the note's duration rounded to quantize grid
            dur_ticks = max(1, int(round(e.duration_ticks / quantize_grid)) * quantize_grid)
            token_d = nearest_duration_token(dur_ticks, tpq)
            token_p = midi_to_name(e.note)
            token_t = pick_technique(e.techniques)
        else:
            # chord — compute a representative duration and technique
            dur_ticks = max(1, max(int(round(e.duration_ticks / quantize_grid)) * quantize_grid for e in evs))
            token_d = nearest_duration_token(dur_ticks, tpq)
            # create canonical sorted chord pitch list (by pitch ascending)
            pitches = sorted([e.note for e in evs])
            pitch_names = [midi_to_name(p) for p in pitches]
            token_p = "CHORD[" + ",".join(pitch_names) + "]"
            # decide technique if any note had one
            all_techs = []
            for e in evs:
                all_techs.extend(e.techniques or [])
            token_t = pick_technique(all_techs)

        onsets.append(onset)
        D.append(token_d)
        P.append(token_p)
        T.append(token_t)

    return {
        "tpq": tpq,
        "tempo_us_per_q": getattr(mid, "tempo", None),  # not directly stored here; extracted elsewhere if needed
        "onsets": onsets,
        "D": D,
        "P": P,
        "T": T
    }

# ---------------------
# Main runner
# ---------------------
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mid', '.midi'))]
    if not files:
        print("No MIDI files (.mid/.midi) found in", input_folder)
        return

    for fn in files:
        path = os.path.join(input_folder, fn)
        print("Processing", path)
        try:
            mid = mido.MidiFile(path)
        except Exception as e:
            print("  Failed to open MIDI:", e)
            continue

        # Parse events (generator)
        events_gen = parse_midi_to_events(mid)
        seq = build_dpt_sequence_from_events(mid, events_gen, quantize_grid=(mid.ticks_per_beat // 4))
        if seq is None:
            print("  No notes >= MIN_PITCH_NOTE found. Skipping.")
            continue

        # Try to read tempo: first set_tempo meta message
        tempo = None
        for track in mid.tracks:
            t = 0
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
            if tempo:
                break
        if tempo is None:
            tempo = 500000  # default 120bpm
        seq["tempo_us_per_q"] = tempo

        # Write JSON
        outname = input_folder.split("\\")[-1] + "_" + os.path.splitext(fn)[0] + ".dpt.json"
        outpath = os.path.join(output_folder, outname)
        with open(outpath, "w", encoding="utf8") as fh:
            json.dump({
                "file": fn,
                "tpq": seq["tpq"],
                "tempo_us_per_q": seq["tempo_us_per_q"],
                "onsets": seq["onsets"],
                "D": seq["D"],
                "P": seq["P"],
                "T": seq["T"]
            }, fh, indent=2)
        print("  Wrote", outpath)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python midi_to_dpt.py /path/to/mid_folder /path/to/out_folder")
        sys.exit(1)
    in_folder = sys.argv[1]
    out_folder = sys.argv[2]
    process_folder(in_folder, out_folder)
