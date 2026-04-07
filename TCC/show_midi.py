"""
Visualiza um arquivo MIDI como piano roll e salva como PNG.
Uso: python show_midi.py arquivo.mid [saida.png]
"""

import sys
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def piano_roll_png(midi_path: str, output_path: str):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Coleta todas as notas de todos os instrumentos
    all_notes = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, note.end, note.pitch, instrument.program))

    if not all_notes:
        print("Nenhuma nota encontrada no arquivo MIDI.")
        return

    total_duration = max(end for _, end, _, _ in all_notes)
    print(f"Arquivo: {midi_path}")
    print(f"  Duração total:  {total_duration:.1f}s")
    print(f"  Total de notas: {len(all_notes)}")
    print(f"  Instrumentos:   {len(pm.instruments)}")

    min_pitch = min(p for _, _, p, _ in all_notes)
    max_pitch = max(p for _, _, p, _ in all_notes)
    pitch_range = max(max_pitch - min_pitch + 1, 12)

    fig_width = max(16, total_duration * 0.5)
    fig_height = max(6, pitch_range * 0.15)
    fig, ax = plt.subplots(figsize=(min(fig_width, 40), fig_height))

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
              '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63']

    for start, end, pitch, program in all_notes:
        duration = max(end - start, 0.05)  # mínimo visual
        color = colors[program % len(colors)]
        rect = patches.Rectangle(
            (start, pitch - 0.4), duration, 0.8,
            linewidth=0.3, edgecolor='black', facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(min_pitch - 2, max_pitch + 2)
    ax.set_xlabel('Tempo (s)', fontsize=11)
    ax.set_ylabel('Nota MIDI (pitch)', fontsize=11)
    ax.set_title(f'Piano Roll — {midi_path}', fontsize=13)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    # Marca oitavas (C de cada oitava = pitches 24, 36, 48, 60, 72, 84, 96, 108)
    for c_pitch in range(24, 109, 12):
        if min_pitch - 2 <= c_pitch <= max_pitch + 2:
            ax.axhline(y=c_pitch, color='gray', linewidth=0.5, alpha=0.5)
            note_name = f'C{(c_pitch // 12) - 1}'
            ax.text(-0.3, c_pitch, note_name, ha='right', va='center',
                    fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Piano roll salvo em: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python show_midi.py arquivo.mid [saida.png]")
        sys.exit(1)

    midi_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else midi_file.replace('.mid', '.png')
    piano_roll_png(midi_file, out_file)
