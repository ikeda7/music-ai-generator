"""
Baseline trivial pra comparação acadêmica contra o Transformer (ep74).

Cadeia de Markov de ordem 1 sobre pares (pitch, duração) extraídos do MAESTRO.
Não há rede neural, não há atenção, não há embeddings — apenas frequências de
transição entre notas. Gera arquivos .mid no mesmo formato do generate.py
(piano solo, canal 0) pra avaliação cega comparativa.

Uso:
    python markov_baseline.py --dataset ./datasets/maestro --output markov.mid
    python markov_baseline.py --dataset ./datasets/maestro --output markov.mid \\
        --duration 60 --num_files 100 --seed 42

Resultado esperado: música "soa caótica/aleatória", contraste claro com o
Transformer. Critério MOS: avaliadores cegos pontuam Markov < Transformer.
"""

import argparse
import glob
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import mido
import pretty_midi


# Tipo do estado da cadeia: (pitch, duração quantizada em 1/16 de segundo)
State = Tuple[int, int]


def _quantize_duration(seconds: float) -> int:
    """Quantiza duração em bins de 0.0625s (resolução do projeto)."""
    return max(1, min(32, round(seconds / 0.0625)))


def extract_states(midi_path: str) -> List[State]:
    """Extrai sequência de (pitch, duração quantizada) de um arquivo MIDI."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []
    states = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        # Ordena por start time pra preservar a sequência temporal real
        for note in sorted(instrument.notes, key=lambda n: n.start):
            if 21 <= note.pitch <= 108:
                states.append((note.pitch, _quantize_duration(note.end - note.start)))
    return states


def build_transition_table(dataset_dir: str, num_files: int = 100,
                           seed: int = 42) -> Tuple[Dict[State, List[Tuple[State, float]]], List[State]]:
    """
    Constrói tabela de transições de Markov ordem 1.

    Retorna:
        transitions: state → lista [(next_state, prob)] ordenada
        initial_states: pool de estados iniciais (primeiros de cada arquivo)
    """
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.mid*'), recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum .mid encontrado em {dataset_dir}")

    rng.shuffle(files)
    files = files[:num_files]
    print(f"Processando {len(files)} arquivos do MAESTRO...")

    bigram_counts: Dict[State, Counter] = defaultdict(Counter)
    initial_states: List[State] = []
    total_states = 0

    for i, path in enumerate(files, 1):
        if i % 20 == 0:
            print(f"  {i}/{len(files)}")
        states = extract_states(path)
        if not states:
            continue
        initial_states.append(states[0])
        for prev, curr in zip(states, states[1:]):
            bigram_counts[prev][curr] += 1
        total_states += len(states)

    # Converte counts em probabilidades cumulativas (pra amostragem rápida)
    transitions: Dict[State, List[Tuple[State, float]]] = {}
    for state, counter in bigram_counts.items():
        total = sum(counter.values())
        cumulative = []
        running = 0.0
        for next_state, count in counter.most_common():
            running += count / total
            cumulative.append((next_state, running))
        transitions[state] = cumulative

    print(f"Tabela construída: {len(transitions)} estados distintos, "
          f"{total_states} notas processadas")
    return transitions, initial_states


def sample_next(transitions: Dict[State, List[Tuple[State, float]]],
                state: State, rng: random.Random,
                initial_states: List[State]) -> State:
    """Amostra próximo estado da cadeia. Se state desconhecido, reinicia."""
    if state not in transitions:
        return rng.choice(initial_states)
    r = rng.random()
    for next_state, cum_prob in transitions[state]:
        if r <= cum_prob:
            return next_state
    return transitions[state][-1][0]


def generate_midi(transitions: Dict[State, List[Tuple[State, float]]],
                  initial_states: List[State], output_path: str,
                  duration_seconds: float = 60.0, tempo_bpm: int = 100,
                  seed: int = 42) -> None:
    """Gera arquivo MIDI por random walk na cadeia."""
    rng = random.Random(seed)

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm)))
    track.append(mido.Message('program_change', channel=0, program=0))

    ticks_per_beat = mid.ticks_per_beat
    current_state = rng.choice(initial_states)
    elapsed = 0.0
    notes_generated = 0

    while elapsed < duration_seconds:
        pitch, dur_quant = current_state
        duration_s = dur_quant * 0.0625
        duration_ticks = max(1, int(duration_s * ticks_per_beat * tempo_bpm / 60))

        track.append(mido.Message('note_on', channel=0, note=pitch,
                                   velocity=80, time=0))
        track.append(mido.Message('note_off', channel=0, note=pitch,
                                   velocity=0, time=duration_ticks))

        elapsed += duration_s
        notes_generated += 1
        current_state = sample_next(transitions, current_state, rng, initial_states)

    mid.save(output_path)
    print(f"MIDI salvo em: {output_path}")
    print(f"  Duração: {elapsed:.1f}s | Notas: {notes_generated}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Markov baseline para comparação acadêmica contra o Transformer'
    )
    parser.add_argument('--dataset', default='./datasets/maestro',
                        help='Diretório com arquivos MIDI do MAESTRO')
    parser.add_argument('--output', default='markov.mid',
                        help='Arquivo MIDI de saída')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duração da peça gerada em segundos')
    parser.add_argument('--tempo', type=int, default=100,
                        help='BPM da peça gerada')
    parser.add_argument('--num_files', type=int, default=100,
                        help='Quantos arquivos do dataset usar pra construir a tabela')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed pra reprodutibilidade')
    args = parser.parse_args()

    transitions, initial_states = build_transition_table(
        args.dataset, num_files=args.num_files, seed=args.seed
    )
    generate_midi(
        transitions, initial_states, args.output,
        duration_seconds=args.duration, tempo_bpm=args.tempo, seed=args.seed
    )