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
    cluster_window = 0.5               # janela pra detectar acorde (meio beat a 95 BPM)
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

    # Bass e Solo: monofônicos — cada NOTE_ON fecha a nota anterior do mesmo
    # registro. Piano real toca melodia e linha de baixo em uma nota por vez;
    # polifonia no solo/baixo cria sensação de "notas comendo umas às outras".
    # Base permanece polifônico (acordes precisam de múltiplas notas).
    for slot in (100, 102):
        if slot in instrument_tracks:
            instrument_tracks[slot] = _enforce_monophony(instrument_tracks[slot])

    return instrument_tracks


def _inject_solid_foundation(instrument_tracks: dict, tempo: int,
                             key_root: int, key_mode: str,
                             total_duration: float) -> dict:
    """
    Substitui o conteúdo dos slots Bass (100) e Base (101) por fundação rítmica
    sintética baseada na progressão diatônica do tom. O slot Solo (102) é
    preservado e enriquecido com notas promovidas do registro Base original.

    Progressões usadas:
      - Maior: I-V-vi-IV (clássica pop)
      - Menor: i-VI-iv-V (V harmônico — clássica pop minor)

    Bass dinâmico: alterna tônica (tempo 1) e quinta (tempo 3) dentro do BAR.
    Base dinâmica: stamp em tempo 1 + tempo 3, mas a cada 4º compasso pula o
                   tempo 3 pra criar respiração ("breath bar").
    Velocity: variação sutil entre stamps fortes (90) e médios (70).

    Solo: notas em 48-65 sobem +12 (preservadas como melodia); aplica gap
    mínimo de 0.25s entre NOTE_ONs pra evitar densidade > base.

    Args:
        instrument_tracks: dict com slots 100/101/102 já populados.
        tempo: BPM da peça.
        key_root: root do tom em semitons (0=C, 2=D, ...).
        key_mode: 'major' ou 'minor'.
        total_duration: duração total em segundos.
    """
    beat_duration = 60.0 / tempo
    bar_duration = beat_duration * 4

    # Promove notas do registro Base pro Solo (+12) ANTES de substituir
    promoted = []
    for item in instrument_tracks.get(101, []):
        t, event = item
        if event.event_type == 'NOTE_ON':
            promoted.append((t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': event.pitch + 12,
                'velocity': event.velocity, 'instrument': 102,
            })()))
    if 102 not in instrument_tracks:
        instrument_tracks[102] = []
    instrument_tracks[102].extend(promoted)

    # Gap mínimo entre NOTE_ONs do solo — 0.15s permite densidade melódica
    # razoável (~semicolcheia a 100 BPM) sem virar metralhadora de notas.
    instrument_tracks[102] = _enforce_min_gap(
        instrument_tracks[102], min_gap=0.15
    )

    # Progressão por modo. Cada entrada: (bass_root_pc, [chord_pcs])
    if key_mode == 'minor':
        # i - VI - iv - V (V maior com 3ª maior — leading tone harmônico)
        progression = [
            (key_root,     [key_root,      key_root + 3,  key_root + 7]),   # i  (menor)
            (key_root + 8, [key_root + 8,  key_root + 12, key_root + 15]),  # VI (relativa maior)
            (key_root + 5, [key_root + 5,  key_root + 8,  key_root + 12]),  # iv (menor)
            (key_root + 7, [key_root + 7,  key_root + 11, key_root + 14]),  # V  (maior)
        ]
    else:
        # I - V - vi - IV (clássica pop maior)
        progression = [
            (key_root,     [key_root,      key_root + 4,  key_root + 7]),   # I
            (key_root + 7, [key_root + 7,  key_root + 11, key_root + 14]),  # V
            (key_root + 9, [key_root + 9,  key_root + 12, key_root + 16]),  # vi
            (key_root + 5, [key_root + 5,  key_root + 9,  key_root + 12]),  # IV
        ]

    bass_octave_base = 36   # C2
    chord_octave_base = 48  # C3
    vel_strong = 95
    vel_medium = 75
    vel_soft   = 60

    bass_events = []
    base_events = []
    bar_count = 0
    t = 0.0

    while t < total_duration:
        bass_pc, chord_pcs = progression[bar_count % 4]
        is_breath_bar = (bar_count % 4 == 3)  # último bar do ciclo respira

        # Bass: tônica no tempo 1, quinta no tempo 3 (movimento root-fifth)
        # Velocity decai sutilmente em breath bar pra dar sensação de cadência
        bass_root_pitch = bass_octave_base + (bass_pc % 12)
        bass_fifth_pitch = bass_octave_base + ((bass_pc + 7) % 12)
        bass_vel = vel_medium if is_breath_bar else vel_strong

        bass_events.append((t, type('E', (), {
            'event_type': 'NOTE_ON', 'pitch': bass_root_pitch,
            'velocity': bass_vel, 'instrument': 100,
        })()))
        # Quinta no tempo 3 (meio do BAR)
        t_mid = t + bar_duration / 2
        if t_mid < total_duration:
            bass_events.append((t_mid, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': bass_fifth_pitch,
                'velocity': vel_soft, 'instrument': 100,
            })()))

        # Base: chord stamp no tempo 1 sempre (forte)
        chord_vel_strong = vel_medium if is_breath_bar else vel_strong
        for pc in chord_pcs:
            chord_pitch = chord_octave_base + (pc % 12)
            base_events.append((t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': chord_pitch,
                'velocity': chord_vel_strong, 'instrument': 101,
            })()))

        # Base: stamp no tempo 3 — EXCETO em breath bar (cria respiração)
        if not is_breath_bar and t_mid < total_duration:
            for pc in chord_pcs:
                chord_pitch = chord_octave_base + (pc % 12)
                base_events.append((t_mid, type('E', (), {
                    'event_type': 'NOTE_ON', 'pitch': chord_pitch,
                    'velocity': vel_medium, 'instrument': 101,
                })()))

        t += bar_duration
        bar_count += 1

    instrument_tracks[100] = bass_events
    instrument_tracks[101] = base_events
    return instrument_tracks


def _enforce_min_gap(events: list, min_gap: float = 0.25) -> list:
    """
    Garante intervalo mínimo entre NOTE_ONs sucessivos. Notas que chegam
    em < min_gap do NOTE_ON anterior são descartadas. Suaviza densidade
    melódica sem alterar o material restante.
    """
    sorted_events = sorted(events, key=lambda x: x[0])
    result = []
    last_note_on_t = -float('inf')
    for item in sorted_events:
        t, event = item
        if event.event_type == 'NOTE_ON':
            if t - last_note_on_t < min_gap:
                continue  # descarta nota muito próxima da anterior
            last_note_on_t = t
            result.append(item)
        else:
            result.append(item)
    return result


def _enforce_monophony(events: list) -> list:
    """
    Força monofonia no registro: cada NOTE_ON fecha a nota anterior ainda ativa.
    Insere NOTE_OFF sintético no timestamp do novo NOTE_ON pra encerrar a anterior.
    """
    result = []
    active_pitch = None
    active_onset = None

    for item in sorted(events, key=lambda x: x[0]):
        t, event = item
        if event.event_type == 'NOTE_ON':
            if active_pitch is not None and active_pitch != event.pitch:
                result.append((t, type('E', (), {
                    'event_type': 'NOTE_OFF', 'pitch': active_pitch,
                    'velocity': 0, 'instrument': event.instrument,
                })()))
            active_pitch = event.pitch
            active_onset = t
            result.append(item)
        elif event.event_type == 'NOTE_OFF':
            result.append(item)
            if event.pitch == active_pitch:
                active_pitch = None

    return result


def tokens_to_midi(tokens: List[int], tokenizer: MIDITokenizer,
                   output_path: str, tempo: int = 120,
                   max_note_duration: float = 1.0,
                   render_as_band: bool = False,
                   render_as_trio: bool = False,
                   solid_base: bool = False,
                   key_root: int = None,
                   key_mode: str = 'major') -> bool:
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

        # Modo --solid_base: substitui bass/base do modelo por fundação sintética
        # alinhada à progressão I-V-vi-IV. O solo (registro alto) vem do modelo.
        # Resultado: arranjo híbrido ML+algorítmico com base e baixo "marcando".
        if solid_base and key_root is not None:
            instrument_tracks = _inject_solid_foundation(
                instrument_tracks, tempo=tempo, key_root=key_root,
                key_mode=key_mode, total_duration=current_time,
            )

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
                    # Velocity boost: garante mínimo 80 (audível) e amplifica ~1.6x.
                    # MAESTRO foi gravado com dynamics suaves; o GM Piano renderizado
                    # ficava abafado. Cap em 120 evita clipping.
                    boosted_vel = max(80, min(120, int(event.velocity * 1.6)))
                    instr_track.append(mido.Message(
                        'note_on', channel=channel,
                        note=event.pitch, velocity=boosted_vel,
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
