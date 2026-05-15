"""
Trunca .mid do eval_samples pra duração uniforme (~60s) e re-gera MP3.

Necessário pra MOS justo: amostras Transformer estavam saindo 70-155s
enquanto Markov saía sempre 60s. Avaliador identificaria modelo pela
duração e quebraria o "blind".

Uso:
    python trim_eval_samples.py --input_dir ./eval_samples --duration 60
"""

import argparse
import glob
import os
import mido


def trim_midi(input_path: str, output_path: str, max_seconds: float,
              tempo_bpm_fallback: int = 100) -> tuple:
    """
    Trunca um arquivo MIDI no max_seconds. Preserva todas as tracks,
    corta eventos cujo tempo acumulado excede o limite e adiciona NOTE_OFFs
    sintéticos pras notas que ficaram pendentes.

    Retorna: (duração_original, duração_truncada) em segundos.
    """
    mid = mido.MidiFile(input_path)
    ticks_per_beat = mid.ticks_per_beat

    # Detecta tempo (último set_tempo encontrado)
    tempo = mido.bpm2tempo(tempo_bpm_fallback)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    seconds_per_tick = mido.tick2second(1, ticks_per_beat, tempo)
    max_ticks = max_seconds / seconds_per_tick

    new_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    original_duration = 0.0

    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)

        elapsed_ticks = 0
        open_notes = {}  # (channel, note) → tick em que abriu

        for msg in track:
            elapsed_ticks += msg.time
            if elapsed_ticks > max_ticks:
                break

            new_track.append(msg.copy())

            # Rastreia notas abertas pra fechar no truncamento
            if msg.type == 'note_on' and msg.velocity > 0:
                open_notes[(msg.channel, msg.note)] = elapsed_ticks
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                open_notes.pop((msg.channel, msg.note), None)

        # Fecha notas que ficaram pendentes — delta_time = 0 pra fechar imediatamente
        for (channel, note), open_tick in open_notes.items():
            new_track.append(mido.Message(
                'note_off', channel=channel, note=note, velocity=0, time=0
            ))

        # Duração original (último evento da track mais longa)
        original_duration = max(original_duration,
                                elapsed_ticks * seconds_per_tick)

    new_mid.save(output_path)
    return original_duration, max_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./eval_samples')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duração máxima em segundos')
    parser.add_argument('--pattern', default='sample_*.mid')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"Nenhum arquivo casando com {args.pattern} em {args.input_dir}")
        return

    print(f"Truncando {len(files)} arquivos pra {args.duration}s...")
    print()
    for path in files:
        # Sobrescreve in-place (backup em /work já preserva originais)
        original_dur, target_dur = trim_midi(path, path, args.duration)
        was_longer = original_dur > args.duration + 0.5
        marker = '[cut] ' if was_longer else '      '
        print(f"  {marker}{os.path.basename(path):20s}  {original_dur:6.1f}s -> {target_dur:.1f}s")

    print()
    print("Pronto. Próximo: re-converter os .mid em .mp3 (MuseScore ou midi2audio).")


if __name__ == '__main__':
    main()
