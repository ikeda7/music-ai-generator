"""
Gera figura comparativa lado a lado de piano rolls — Transformer vs Markov.

Cria uma figura única com 2 (ou mais) subplots horizontais, cada um
mostrando o piano roll de um arquivo .mid. Uso pretendido: figura
qualitativa pro artigo mostrando a diferença visual entre os modelos.

Uso:
    # Comparação de 2 amostras
    python plot_comparison_rolls.py --inputs eval_samples/sample_A.mid eval_samples/sample_B.mid \\
        --labels "Transformer (Em)" "Markov" --output figura_comparacao.png

    # Comparação de 4 amostras (grid 2x2)
    python plot_comparison_rolls.py --inputs sample_A.mid sample_B.mid sample_E.mid sample_F.mid \\
        --labels Transformer Markov Transformer Markov --output grid.png
"""

import argparse
import os

import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


_ROLE_COLORS = {
    'Solo':    '#4CAF50',
    'Base':    '#2196F3',
    'Baixo':   '#9C27B0',
    'Bateria': '#E91E63',
    'Outro':   '#FF9800',
}

_REGISTER_LIMITS = [
    (66, 108, 'Solo'),
    (48,  65, 'Base'),
    (21,  47, 'Baixo'),
]


def _infer_role(instrument) -> str:
    if instrument.is_drum:
        return 'Bateria'
    notes = instrument.notes
    if not notes:
        return 'Outro'
    pitches = sorted(n.pitch for n in notes)
    median = pitches[len(pitches) // 2]
    for pmin, pmax, name in _REGISTER_LIMITS:
        if pmin <= median <= pmax:
            return name
    return 'Outro'


def _draw_roll(ax, midi_path: str, label: str):
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_pitches = []
    total_duration = 0.0
    n_notes = 0

    for instrument in pm.instruments:
        if not instrument.notes:
            continue
        role = _infer_role(instrument)
        color = _ROLE_COLORS.get(role, _ROLE_COLORS['Outro'])
        for note in instrument.notes:
            duration = max(note.end - note.start, 0.05)
            rect = patches.Rectangle(
                (note.start, note.pitch - 0.4), duration, 0.8,
                linewidth=0.2, edgecolor='black', facecolor=color, alpha=0.85,
            )
            ax.add_patch(rect)
            all_pitches.append(note.pitch)
            total_duration = max(total_duration, note.end)
            n_notes += 1

    if not all_pitches:
        ax.text(0.5, 0.5, '(vazio)', transform=ax.transAxes,
                ha='center', va='center')
        return

    min_pitch = min(all_pitches)
    max_pitch = max(all_pitches)
    ax.set_xlim(0, total_duration)
    ax.set_ylim(min_pitch - 2, max_pitch + 2)
    ax.set_xlabel('Tempo (s)', fontsize=10)
    ax.set_ylabel('Pitch MIDI', fontsize=10)
    ax.set_title(f'{label}  ({n_notes} notas, {total_duration:.1f}s)', fontsize=12)
    ax.grid(True, alpha=0.25, linestyle='--')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='Arquivos .mid')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels (um por input). Se ausente usa basename.')
    parser.add_argument('--output', default='figura_comparacao.png')
    parser.add_argument('--cols', type=int, default=2,
                        help='Quantidade de colunas no grid (default 2)')
    args = parser.parse_args()

    n = len(args.inputs)
    if args.labels and len(args.labels) != n:
        raise ValueError("--labels deve ter o mesmo tamanho de --inputs")
    labels = args.labels or [os.path.basename(p) for p in args.inputs]

    cols = min(args.cols, n)
    rows = (n + cols - 1) // cols
    fig_w = 8 * cols
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, (path, label) in enumerate(zip(args.inputs, labels)):
        r, c = divmod(idx, cols)
        _draw_roll(axes[r][c], path, label)

    # Limpa subplots vazios se houver
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis('off')

    # Legenda compartilhada
    legend_handles = [
        patches.Patch(facecolor=color, edgecolor='black', label=role)
        for role, color in _ROLE_COLORS.items() if role != 'Outro'
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"Figura salva em: {args.output}")


if __name__ == '__main__':
    main()
