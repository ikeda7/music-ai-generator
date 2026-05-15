"""
Métricas quantitativas pra comparação Transformer vs Markov vs Dataset.

Calcula sobre arquivos .mid:
- pitch_diversity: # pitches únicos / total de notas (0-1)
- note_density: notas por segundo
- pitch_range: maior - menor pitch (semitons)
- ioi_std: desvio-padrão dos inter-onset intervals (regularidade rítmica;
  valores baixos = mais metronômico, altos = mais variação humana)
- kl_vs_dataset: divergência KL do histograma de pitch classes (pc 0-11)
  contra a referência do MAESTRO; menor = mais "musical" (próximo do dataset)

Uso:
    # Métricas de um arquivo
    python metrics.py --input minha_musica.mid --reference ./datasets/maestro

    # Comparação batch (gera CSV)
    python metrics.py --input ./samples/ --reference ./datasets/maestro \\
        --output metricas.csv

A referência (MAESTRO) é processada uma vez e cacheada em memória.
"""

import argparse
import csv
import glob
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional

import pretty_midi


def _pitch_class_histogram(midi_path: str) -> Optional[Counter]:
    """Retorna Counter de pitch classes (0-11) normalizado por # notas."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None
    counter = Counter()
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            counter[note.pitch % 12] += 1
    return counter if sum(counter.values()) > 0 else None


def _build_reference_pc_distribution(dataset_dir: str, num_files: int = 50,
                                     seed: int = 42) -> Dict[int, float]:
    """Constrói distribuição de pitch classes de referência (do MAESTRO)."""
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.mid*'), recursive=True))
    if not files:
        raise FileNotFoundError(f"Sem .mid em {dataset_dir}")
    rng.shuffle(files)
    files = files[:num_files]

    total = Counter()
    for path in files:
        h = _pitch_class_histogram(path)
        if h:
            total.update(h)

    grand = sum(total.values())
    return {pc: total.get(pc, 0) / grand for pc in range(12)}


def _kl_divergence(p: Dict[int, float], q: Dict[int, float],
                   epsilon: float = 1e-9) -> float:
    """KL(P || Q) em pitch classes. Smoothing pra evitar log(0)."""
    kl = 0.0
    for pc in range(12):
        p_pc = p.get(pc, 0.0) + epsilon
        q_pc = q.get(pc, 0.0) + epsilon
        kl += p_pc * math.log(p_pc / q_pc)
    return kl


def compute_metrics(midi_path: str,
                    reference_pc: Optional[Dict[int, float]] = None) -> Dict[str, float]:
    """Computa todas as métricas pra um arquivo MIDI."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        all_notes.extend(inst.notes)

    if not all_notes:
        return {'pitch_diversity': 0, 'note_density': 0, 'pitch_range': 0,
                'ioi_std': 0, 'kl_vs_dataset': float('inf'), 'total_notes': 0,
                'duration_s': 0}

    total_notes = len(all_notes)
    duration_s = max(n.end for n in all_notes)
    pitches = [n.pitch for n in all_notes]

    pitch_diversity = len(set(pitches)) / total_notes
    note_density = total_notes / max(duration_s, 0.1)
    pitch_range = max(pitches) - min(pitches)

    # Inter-onset intervals: variação de tempo entre ataques consecutivos
    onsets = sorted(n.start for n in all_notes)
    iois = [b - a for a, b in zip(onsets, onsets[1:])]
    if iois:
        mean_ioi = sum(iois) / len(iois)
        ioi_std = math.sqrt(sum((x - mean_ioi) ** 2 for x in iois) / len(iois))
    else:
        ioi_std = 0.0

    kl_vs_dataset = float('nan')
    if reference_pc is not None:
        hist = _pitch_class_histogram(midi_path)
        if hist:
            grand = sum(hist.values())
            pc_dist = {pc: hist.get(pc, 0) / grand for pc in range(12)}
            kl_vs_dataset = _kl_divergence(pc_dist, reference_pc)

    return {
        'pitch_diversity': round(pitch_diversity, 3),
        'note_density': round(note_density, 2),
        'pitch_range': pitch_range,
        'ioi_std': round(ioi_std, 3),
        'kl_vs_dataset': round(kl_vs_dataset, 3) if not math.isnan(kl_vs_dataset) else None,
        'total_notes': total_notes,
        'duration_s': round(duration_s, 1),
    }


def _format_row(name: str, m: Dict[str, float]) -> str:
    return (f"{name:30s}  "
            f"diversity={m['pitch_diversity']:.3f}  "
            f"density={m['note_density']:5.2f} n/s  "
            f"range={m['pitch_range']:3d}  "
            f"ioi_std={m['ioi_std']:.3f}  "
            f"kl={m['kl_vs_dataset'] if m['kl_vs_dataset'] is not None else 'N/A'}  "
            f"notes={m['total_notes']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Métricas quantitativas pra MIDIs')
    parser.add_argument('--input', required=True,
                        help='Arquivo .mid ou diretório com .mids')
    parser.add_argument('--reference', default='./datasets/maestro',
                        help='Dataset de referência pra KL (default: MAESTRO)')
    parser.add_argument('--output', default=None,
                        help='Saída CSV (opcional; default só imprime)')
    parser.add_argument('--num_ref_files', type=int, default=50,
                        help='# arquivos do dataset pra calcular referência')
    args = parser.parse_args()

    print(f"Construindo distribuição de referência do MAESTRO...")
    ref_pc = _build_reference_pc_distribution(args.reference,
                                              num_files=args.num_ref_files)
    print(f"Referência pronta: {len(ref_pc)} pitch classes\n")

    # Coleta lista de arquivos
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, '*.mid*')))
    else:
        files = [args.input]

    rows = []
    for path in files:
        try:
            m = compute_metrics(path, reference_pc=ref_pc)
            rows.append((os.path.basename(path), m))
            print(_format_row(os.path.basename(path), m))
        except Exception as e:
            print(f"Erro em {path}: {e}")

    if args.output and rows:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'pitch_diversity', 'note_density',
                             'pitch_range', 'ioi_std', 'kl_vs_dataset',
                             'total_notes', 'duration_s'])
            for name, m in rows:
                writer.writerow([name, m['pitch_diversity'], m['note_density'],
                                 m['pitch_range'], m['ioi_std'],
                                 m['kl_vs_dataset'], m['total_notes'],
                                 m['duration_s']])
        print(f"\nCSV salvo em: {args.output}")
