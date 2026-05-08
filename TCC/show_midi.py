"""
Visualiza um arquivo MIDI como piano roll colorido por track e salva como PNG.
Uso: python show_midi.py arquivo.mid [saida.png]

Cores por papel funcional (auto-detectado pelo registro de pitch da track):
- Solo  (66-108): verde
- Base  (48-65):  azul
- Baixo (21-47):  roxo
"""

import sys
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paleta por papel funcional. Track sem papel claro cai pro fallback.
_ROLE_COLORS = {
    'Solo':   '#4CAF50',   # verde
    'Base':   '#2196F3',   # azul
    'Baixo':  '#9C27B0',   # roxo
    'Outro':  '#FF9800',   # laranja (fallback)
}

# Limites de registro pro auto-label (mesmos do music_utils._BAND_REGISTERS)
_REGISTER_LIMITS = [
    (66, 108, 'Solo'),
    (48,  65, 'Base'),
    (21,  47, 'Baixo'),
]


def _infer_role(notes) -> str:
    """Classifica a track pelo registro mediano dos pitches."""
    if not notes:
        return 'Outro'
    pitches = sorted(n.pitch for n in notes)
    median = pitches[len(pitches) // 2]
    for pmin, pmax, name in _REGISTER_LIMITS:
        if pmin <= median <= pmax:
            return name
    return 'Outro'


def piano_roll_png(midi_path: str, output_path: str):
    pm = pretty_midi.PrettyMIDI(midi_path)

    # Coleta notas e infere papel de cada track
    tracks = []
    for instrument in pm.instruments:
        if not instrument.notes:
            continue
        role = _infer_role(instrument.notes)
        tracks.append((role, instrument))

    if not tracks:
        print("Nenhuma nota encontrada no arquivo MIDI.")
        return

    all_notes_flat = [n for _, inst in tracks for n in inst.notes]
    total_duration = max(n.end for n in all_notes_flat)
    min_pitch = min(n.pitch for n in all_notes_flat)
    max_pitch = max(n.pitch for n in all_notes_flat)

    print(f"Arquivo: {midi_path}")
    print(f"  Duração total:  {total_duration:.1f}s")
    print(f"  Total de notas: {len(all_notes_flat)}")
    print(f"  Tracks:         {len(tracks)}")
    for role, inst in tracks:
        print(f"    - {role:6s}: {len(inst.notes)} notas (program={inst.program})")

    pitch_range = max(max_pitch - min_pitch + 1, 12)
    fig_width = max(16, total_duration * 0.5)
    fig_height = max(6, pitch_range * 0.15)
    fig, ax = plt.subplots(figsize=(min(fig_width, 40), fig_height))

    # Renderiza notas de cada track na sua cor
    for role, inst in tracks:
        color = _ROLE_COLORS.get(role, _ROLE_COLORS['Outro'])
        for note in inst.notes:
            duration = max(note.end - note.start, 0.05)
            rect = patches.Rectangle(
                (note.start, note.pitch - 0.4), duration, 0.8,
                linewidth=0.3, edgecolor='black', facecolor=color, alpha=0.85,
            )
            ax.add_patch(rect)

    # Legenda — uma entrada por papel presente
    legend_handles = []
    seen_roles = set()
    for role, _ in tracks:
        if role in seen_roles:
            continue
        seen_roles.add(role)
        legend_handles.append(
            patches.Patch(facecolor=_ROLE_COLORS.get(role, _ROLE_COLORS['Outro']),
                          edgecolor='black', label=role)
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10)

    ax.set_xlim(0, total_duration)
    ax.set_ylim(min_pitch - 2, max_pitch + 2)
    ax.set_xlabel('Tempo (s)', fontsize=11)
    ax.set_ylabel('Nota MIDI (pitch)', fontsize=11)
    ax.set_title(f'Piano Roll — {midi_path}', fontsize=13)

    # Grid de beats (assume 120 BPM padrão pra grid visual)
    beat_duration = 0.5
    bar_duration = beat_duration * 4
    beat = 0.0
    while beat <= total_duration:
        is_bar = abs(beat % bar_duration) < 0.01
        ax.axvline(x=beat, color='red' if is_bar else 'orange',
                   linewidth=0.8 if is_bar else 0.3,
                   alpha=0.6 if is_bar else 0.35,
                   linestyle='-' if is_bar else '--')
        beat += beat_duration

    # Marca oitavas (C de cada oitava)
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