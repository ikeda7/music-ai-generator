"""
Utilitários musicais para conversão entre tokens e MIDI.
"""

import mido
from typing import List
from data_processor import MIDITokenizer

# Programa General MIDI por slot de instrumento (modo padrão — usa o instrumento do token)
_INSTRUMENT_PROGRAM = {
    0: 0,   # Piano (Acoustic Grand Piano)
    1: 24,  # Melody (Nylon Guitar)
    2: 32,  # Bass (Acoustic Bass)
    3: 0,   # Bateria — canal 9 não usa program_change
    4: 48,  # Harmony (String Ensemble)
}

# Canal MIDI por slot. Bateria deve estar no canal 9 (General MIDI).
_INSTRUMENT_CHANNEL = {0: 0, 1: 1, 2: 2, 3: 9, 4: 3}

# --- Modo "banda": remapeia pitch → instrumento GM por registro ---
# Reaproveita o modelo piano-solo (MAESTRO) separando as 3 vozes que já emergem
# naturalmente do piano (baixo / harmonia / solo) em timbres distintos.
# Bateria é excluída — tem pipeline próprio.
_BAND_REGISTERS = [
    # (pitch_min, pitch_max, slot_virtual, channel, program, nome)
    (21,  47, 100, 2, 33, 'Bass'),      # Finger Bass
    (48,  65, 101, 1, 24, 'Guitar'),    # Nylon Guitar (base/harmonia)
    (66, 108, 102, 0, 30, 'Lead'),      # Distortion Guitar (solo)
]

# Mapas derivados para lookup O(1) a partir do slot virtual
_BAND_CHANNEL = {slot: ch for _, _, slot, ch, _, _ in _BAND_REGISTERS}
_BAND_PROGRAM = {slot: prog for _, _, slot, _, prog, _ in _BAND_REGISTERS}


def _band_slot_for_pitch(pitch: int) -> int:
    """Retorna o slot virtual (100/101/102) correspondente ao registro do pitch."""
    for pmin, pmax, slot, _, _, _ in _BAND_REGISTERS:
        if pmin <= pitch <= pmax:
            return slot
    # Fallback: pitch extremo fora das faixas definidas → solo
    return 102


def tokens_to_midi(tokens: List[int], tokenizer: MIDITokenizer,
                   output_path: str, tempo: int = 120,
                   max_note_duration: float = 1.5,
                   render_as_band: bool = False) -> bool:
    """
    Converte uma sequência de tokens em um arquivo MIDI.

    Args:
        tokens: Lista de ids de tokens
        tokenizer: Instância do MIDITokenizer
        output_path: Caminho para salvar o arquivo MIDI
        tempo: Tempo em BPM (batidas por minuto)
        max_note_duration: Duração máxima de uma nota em segundos (cap anti-hanging).
        render_as_band: Se True, ignora o INSTRUMENT do token e roteia cada NOTE_ON
            pra canal GM diferente conforme o registro de pitch (baixo/base/solo).
            Bateria não é renderizada neste modo.

    Retorna:
        True se bem-sucedido, False caso contrário
    """
    try:
        events = tokenizer.decode_tokens(tokens)

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        ticks_per_beat = mid.ticks_per_beat
        microseconds_per_beat = mido.bpm2tempo(tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))

        # Chave de ordenação: NOTE_OFF antes de NOTE_ON no mesmo instante
        def _sort_key(x):
            t, e = x
            return (t, 0 if e.event_type == 'NOTE_OFF' else 1)

        # Agrupa NOTE_ON por "slot". No modo band, slot vem do registro de pitch;
        # caso contrário, vem do event.instrument (slot original do token).
        instrument_tracks = {}
        current_time = 0.0

        for event in events:
            if event.event_type == 'TIME_SHIFT':
                current_time = event.time
            elif event.event_type == 'NOTE_ON':
                if render_as_band:
                    slot = _band_slot_for_pitch(event.pitch)
                else:
                    slot = event.instrument
                if slot not in instrument_tracks:
                    instrument_tracks[slot] = []
                instrument_tracks[slot].append((current_time, event))

        # Fecha notas abertas sem NOTE_OFF correspondente (cap em max_note_duration)
        for instr_idx, instr_events in instrument_tracks.items():
            open_at = {}
            extras = []
            for event_time, event in sorted(instr_events, key=lambda x: x[0]):
                if event.event_type == 'NOTE_ON':
                    if event.pitch in open_at:
                        forced_off = min(open_at[event.pitch] + max_note_duration, event_time)
                        extras.append((forced_off, type('E', (), {
                            'event_type': 'NOTE_OFF', 'pitch': event.pitch,
                            'velocity': 0, 'instrument': instr_idx,
                        })()))
                    open_at[event.pitch] = event_time
                elif event.event_type == 'NOTE_OFF' and event.pitch in open_at:
                    del open_at[event.pitch]
            for pitch, on_time in open_at.items():
                extras.append((on_time + max_note_duration, type('E', (), {
                    'event_type': 'NOTE_OFF', 'pitch': pitch,
                    'velocity': 0, 'instrument': instr_idx,
                })()))
            instrument_tracks[instr_idx] = sorted(instr_events + extras, key=_sort_key)

        # Cria tracks com canal/program conforme modo
        for instr_idx, instr_events in instrument_tracks.items():
            instr_track = mido.MidiTrack()
            mid.tracks.append(instr_track)

            if render_as_band:
                channel = _BAND_CHANNEL.get(instr_idx, 0)
                program = _BAND_PROGRAM.get(instr_idx, 0)
            else:
                channel = _INSTRUMENT_CHANNEL.get(instr_idx, instr_idx % 9)
                program = _INSTRUMENT_PROGRAM.get(instr_idx, 0)

            # Canal 9 é bateria — não envia program_change
            if channel != 9:
                instr_track.append(mido.Message('program_change', channel=channel, program=program))

            instr_events.sort(key=lambda x: (x[0], 0 if x[1].event_type == 'NOTE_OFF' else 1))

            last_time = 0.0
            for event_time, event in instr_events:
                delta_ticks = max(0, int((event_time - last_time) * ticks_per_beat * tempo / 60))

                if event.event_type == 'NOTE_ON':
                    instr_track.append(mido.Message(
                        'note_on', channel=channel,
                        note=event.pitch, velocity=event.velocity,
                        time=delta_ticks
                    ))
                elif event.event_type == 'NOTE_OFF':
                    instr_track.append(mido.Message(
                        'note_off', channel=channel,
                        note=event.pitch, velocity=0,
                        time=delta_ticks
                    ))

                last_time = event_time

        mid.save(output_path)
        print(f"MIDI salvo em: {output_path}")
        return True

    except Exception as e:
        print(f"Erro ao converter tokens para MIDI: {e}")
        return False
