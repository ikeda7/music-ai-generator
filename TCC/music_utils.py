"""
Utilitários musicais para conversão entre tokens e MIDI.
"""

import random
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

# --- Modo "trio": mesma separação funcional, TUDO em piano ---
# Mantém o timbre de Grand Piano (alta qualidade GM) em vez dos sintetizadores
# de guitarra/baixo. Três tracks no MIDI (uma por registro) mas todas no
# mesmo instrumento. O usuário enxerga solo/base/baixo estruturados no DAW.
_TRIO_REGISTERS = [
    # (pitch_min, pitch_max, slot_virtual, channel, program, nome)
    (21,  47, 100, 2, 0, 'Baixo'),     # Grand Piano
    (48,  65, 101, 1, 0, 'Base'),      # Grand Piano
    (66, 108, 102, 0, 0, 'Solo'),      # Grand Piano
]

# Mapas derivados para lookup O(1) a partir do slot virtual
_BAND_CHANNEL = {slot: ch for _, _, slot, ch, _, _ in _BAND_REGISTERS}
_BAND_PROGRAM = {slot: prog for _, _, slot, _, prog, _ in _BAND_REGISTERS}
_TRIO_CHANNEL = {slot: ch for _, _, slot, ch, _, _ in _TRIO_REGISTERS}
_TRIO_PROGRAM = {slot: prog for _, _, slot, _, prog, _ in _TRIO_REGISTERS}


def _band_slot_for_pitch(pitch: int) -> int:
    """Retorna o slot virtual (100/101/102) correspondente ao registro do pitch."""
    for pmin, pmax, slot, _, _, _ in _BAND_REGISTERS:
        if pmin <= pitch <= pmax:
            return slot
    # Fallback: pitch extremo fora das faixas definidas → solo
    return 102


def _apply_band_filters(instrument_tracks: dict, tempo: int, seed: int = 42) -> dict:
    """
    Filtros funcionais aplicados no modo render_as_band para reduzir congestionamento
    sonoro e atribuir papéis musicais claros a cada registro:

    - Bass (slot 100):   mantém NOTE_ON próximos a BAR boundaries (±0.15s);
                         notas fora do tempo forte passam com 15% de chance (groove).
    - Base (slot 101):   mantém apenas NOTE_ON em clusters (≥2 notas em 0.2s).
                         Notas isoladas do registro médio são descartadas — o acorde
                         só soa quando tem pelo menos dois pitches simultâneos.
    - Solo (slot 102):   passa livre, sem filtro.

    Determinístico: usa random.Random(seed) pra reprodutibilidade (regra do projeto).
    """
    rng = random.Random(seed)
    beat_duration = 60.0 / tempo       # segundos por beat
    bar_duration = beat_duration * 4   # 4/4 tempo
    bar_tolerance = 0.15               # janela do tempo forte
    cluster_window = 0.2               # janela pra detectar acorde
    groove_probability = 0.15          # chance de bass tocar fora do tempo

    # Bass: quantiza NOTE_ON ao BAR; descarta fora com prob 1 - groove
    if 100 in instrument_tracks:
        filtered = []
        for item in instrument_tracks[100]:
            t, event = item
            if event.event_type != 'NOTE_ON':
                filtered.append(item)
                continue
            nearest_bar = round(t / bar_duration) * bar_duration
            if abs(t - nearest_bar) <= bar_tolerance:
                filtered.append(item)
            elif rng.random() < groove_probability:
                filtered.append(item)
        instrument_tracks[100] = filtered

    # Base: detecta clusters (janela deslizante de 0.2s com ≥2 notas)
    if 101 in instrument_tracks:
        notes = [it for it in instrument_tracks[101] if it[1].event_type == 'NOTE_ON']
        notes.sort(key=lambda x: x[0])
        n = len(notes)
        keep_idx = set()
        for i in range(n):
            t_i = notes[i][0]
            count = 1
            j = i - 1
            while j >= 0 and t_i - notes[j][0] <= cluster_window:
                count += 1
                j -= 1
            j = i + 1
            while j < n and notes[j][0] - t_i <= cluster_window:
                count += 1
                j += 1
            if count >= 2:
                keep_idx.add(i)
        instrument_tracks[101] = [notes[i] for i in sorted(keep_idx)]

    # Solo: sem modificação
    return instrument_tracks


def tokens_to_midi(tokens: List[int], tokenizer: MIDITokenizer,
                   output_path: str, tempo: int = 120,
                   max_note_duration: float = 1.5,
                   render_as_band: bool = False,
                   render_as_trio: bool = False) -> bool:
    """
    Converte uma sequência de tokens em um arquivo MIDI.

    Args:
        tokens: Lista de ids de tokens
        tokenizer: Instância do MIDITokenizer
        output_path: Caminho para salvar o arquivo MIDI
        tempo: Tempo em BPM (batidas por minuto)
        max_note_duration: Duração máxima de uma nota em segundos (cap anti-hanging).
        render_as_band: Se True, ignora o INSTRUMENT do token e roteia cada NOTE_ON
            pra canal GM diferente conforme o registro de pitch (baixo/base/solo),
            aplicando timbres distintos (Bass + Nylon Guitar + Lead Guitar).
            Bateria não é renderizada neste modo.
        render_as_trio: Como render_as_band, mas mantém TODAS as vozes em piano
            (Grand Piano em 3 tracks separados). Recomendado pro TCC — preserva
            a qualidade de timbre do MAESTRO sem introduzir sintetizadores de
            baixa fidelidade. Tem precedência sobre render_as_band se ambos True.

    Retorna:
        True se bem-sucedido, False caso contrário
    """
    # Trio tem precedência; remapear por registro funciona igual pros dois modos
    remap_by_register = render_as_trio or render_as_band
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
                if remap_by_register:
                    slot = _band_slot_for_pitch(event.pitch)
                else:
                    slot = event.instrument
                if slot not in instrument_tracks:
                    instrument_tracks[slot] = []
                instrument_tracks[slot].append((current_time, event))

        # Nos modos banda/trio, aplica filtros funcionais (bass quantizado ao BAR,
        # base só em clusters de 2+ notas, solo livre). Isso reduz o congestionamento
        # e atribui papéis musicais claros sem mexer no modelo — tudo no render.
        if remap_by_register:
            instrument_tracks = _apply_band_filters(instrument_tracks, tempo=tempo)

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

            if render_as_trio:
                channel = _TRIO_CHANNEL.get(instr_idx, 0)
                program = _TRIO_PROGRAM.get(instr_idx, 0)
            elif render_as_band:
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
