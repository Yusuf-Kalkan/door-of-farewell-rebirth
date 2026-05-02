#!/usr/bin/env python3
"""
dpt_to_midi.py

Convert D->P->T sequences to a MIDI file.

Supports:
 - input .dpt.json files with keys: "tpq" (optional), "tempo_bpm" or "tempo_us_per_q" (optional),
   "onsets" (optional), "D", "P", "T" (lists).
 - input plain text file where each non-empty line is: D=<dur>  P=<pitch_or_CHORD>  T=<tech>
   e.g. D=eighth P=F2 T=normal
         D=sixteenth P=CHORD[C#4,G#4] T=normal

Defaults:
 - tpq = 96 (ticks per quarter)
 - tempo = 120 BPM
 - default velocities & technique effects are applied (see code)

Requires: mido
    pip install mido
"""
import argparse
import json
import re
import math
from collections import defaultdict
import mido

NOTE_NAMES = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}

# duration mapping in quarters (fractions of quarter note)
DURATION_QUARTER_MULT = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "eighth_triplet": 1.0/3.0,
    "dotted_eighth": 0.75,
    "sixteenth": 0.25,
    "thirtysecond": 0.125
}

# technique rendering config
TECH_CONFIG = {
    "normal": {"vel": 90, "len_mult": 1.0},
    "palm_mute": {"vel": 70, "len_mult": 0.5},
    "harmonic": {"vel": 110, "len_mult": 0.75}
}

DEFAULT_TPQ = 96
DEFAULT_BPM = 120

def name_to_midi(note_name):
    """Convert 'C#4' style name to MIDI number (C4 = 60)."""
    m = re.match(r'^([A-G]#?)(-?\d+)$', note_name)
    if not m:
        raise ValueError(f"Bad pitch name: {note_name}")
    base, octs = m.group(1), int(m.group(2))
    return (octs + 1) * 12 + NOTE_NAMES[base]

def parse_chord_token(token):
    """Parse token like CHORD[C#4,G#4] into list of MIDI numbers."""
    m = re.match(r'^CHORD\[(.*)\]$', token)
    if not m:
        raise ValueError("Not a chord token: " + token)
    inner = m.group(1).strip()
    if inner == "":
        return []
    parts = [p.strip() for p in inner.split(',') if p.strip()]
    return [name_to_midi(p) for p in parts]

def duration_token_to_ticks(d_token, tpq):
    """Map duration token name to integer ticks (round). Unknown tokens are interpreted as
    'other_<n>' where n is ticks (int)."""
    if d_token.startswith("other_"):
        try:
            return int(d_token.split("_",1)[1])
        except:
            return tpq // 4  # fallback to sixteenth
    if d_token in DURATION_QUARTER_MULT:
        frac = DURATION_QUARTER_MULT[d_token]
        return int(round(frac * tpq))
    # last attempt: numeric string
    try:
        return int(d_token)
    except:
        # fallback to sixteenth
        return tpq // 4

def load_dpt_json(path):
    with open(path, "r", encoding="utf8") as fh:
        data = json.load(fh)
    # validate
    if "D" not in data or "P" not in data or "T" not in data:
        raise ValueError("DPT JSON must contain D,P,T arrays")
    tpq = data.get("tpq", DEFAULT_TPQ)
    # tempo: prefer BPM if present; else try tempo_us_per_q -> compute bpm
    tempo_bpm = data.get("tempo_bpm", None)
    tempo_us = data.get("tempo_us_per_q", None)
    if tempo_bpm is None and tempo_us is not None:
        tempo_bpm = mido.tempo2bpm(tempo_us)
    if tempo_bpm is None:
        tempo_bpm = DEFAULT_BPM
    return {
        "tpq": tpq,
        "bpm": tempo_bpm,
        "D": data["D"],
        "P": data["P"],
        "T": data["T"]
    }

def load_txt_sequence(path):
    D, P, T = [], [], []
    with open(path, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # allow flexible separators
            # look for D=..., P=..., T=...
            d_match = re.search(r'D=([^\s]+)', line)
            p_match = re.search(r'P=([^\s]+)', line)
            t_match = re.search(r'T=([^\s]+)', line)
            if not (d_match and p_match and t_match):
                raise ValueError("Line not parseable (expect D=... P=... T=...): " + line)
            D.append(d_match.group(1))
            P.append(p_match.group(1))
            T.append(t_match.group(1))
    return {
        "tpq": DEFAULT_TPQ,
        "bpm": DEFAULT_BPM,
        "D": D,
        "P": P,
        "T": T
    }

def build_midi_from_sequence(seq, output_path, channel=0):
    """
    seq: dict with keys 'tpq','bpm','D','P','T' (lists same length)
    Writes MIDI file to output_path.
    """
    tpq = int(seq.get("tpq", DEFAULT_TPQ))
    bpm = float(seq.get("bpm", DEFAULT_BPM))
    D = seq["D"]
    P = seq["P"]
    T = seq["T"]
    if not (len(D) == len(P) == len(T)):
        raise ValueError("D,P,T lengths mismatch")

    # Collect absolute-tick events: each event -> add note-ons at onset, note-offs at onset+length
    events = []  # list of tuples (abs_tick, 'on'/'off', midi_note, velocity)
    cur_tick = 0
    for i, (d_tok, p_tok, t_tok) in enumerate(zip(D,P,T)):
        dur_ticks = duration_token_to_ticks(d_tok, tpq)
        if p_tok.upper() == "REST":
            # just advance time by dur_ticks (no events)
            cur_tick += dur_ticks
            continue

        # determine notes
        if p_tok.startswith("CHORD["):
            notes = parse_chord_token(p_tok)
        else:
            notes = [name_to_midi(p_tok)]

        # technique mapping
        tech = t_tok if t_tok in TECH_CONFIG else "normal"
        cfg = TECH_CONFIG.get(tech, TECH_CONFIG["normal"])
        velocity = cfg["vel"]
        length_multiplier = cfg["len_mult"]
        note_length = max(1, int(round(dur_ticks * length_multiplier)))

        # schedule note-ons at cur_tick, note-offs at cur_tick + note_length
        for n in notes:
            events.append((cur_tick, "on", n, velocity, channel))
            events.append((cur_tick + note_length, "off", n, 0, channel))

        cur_tick += dur_ticks

    # sort events by absolute tick, and ensure off events come before on events at same tick to avoid tiny overlaps
    def evt_sort_key(e):
        abs_tick, kind, note, vel, ch = e
        return (abs_tick, 0 if kind == "off" else 1, note)
    events.sort(key=evt_sort_key)

    # convert absolute ticks -> delta times and create mido messages
    mid = mido.MidiFile(ticks_per_beat=tpq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    # set tempo
    tempo_us = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    # optionally program change or name
    track.append(mido.MetaMessage('track_name', name='DPT_generated', time=0))

    last_tick = 0
    for abs_tick, kind, note, vel, ch in events:
        delta = abs_tick - last_tick
        last_tick = abs_tick
        if kind == "on":
            msg = mido.Message('note_on', note=note, velocity=vel, time=delta, channel=ch)
        else:
            msg = mido.Message('note_off', note=note, velocity=vel, time=delta, channel=ch)
        track.append(msg)
    # ensure file ends with some trailing delta 0 meta if needed
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(output_path)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert DPT sequences to a MIDI file.")
    parser.add_argument("--input", "-i", required=True, help="Input .dpt.json or plain text sequence file")
    parser.add_argument("--output", "-o", required=True, help="Output .mid path")
    parser.add_argument("--format", "-f", choices=["dpt_json", "txt", "auto"], default="auto",
                        help="Input format (auto detects by extension).")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        if args.input.lower().endswith(".dpt.json") or args.input.lower().endswith(".json"):
            fmt = "dpt_json"
        else:
            fmt = "txt"

    if fmt == "dpt_json":
        seq = load_dpt_json(args.input)
    else:
        seq = load_txt_sequence(args.input)

    out = build_midi_from_sequence(seq, args.output)
    print("Wrote MIDI to", out)

if __name__ == "__main__":
    main()
