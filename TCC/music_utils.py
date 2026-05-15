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


# Slot virtual da bateria. Canal 9 no GM = percussão; pitches são mapeamentos GM:
# 36=kick, 38=snare, 42=closed hi-hat, 46=open hi-hat
_DRUM_SLOT = 200


def _generate_drum_pattern(total_duration: float, tempo: int,
                           seed: int = 42) -> list:
    """
    Padrão rock/pop básico 4/4 — algoritmo determinístico, sem ML:
    - Kick (36):       beats 1 e 3 — fundação
    - Snare (38):      beats 2 e 4 — backbeat
    - Closed hihat (42): a cada colcheia (8 por bar) — tempo
    - Open hihat (46): no último beat de cada 4º bar — crash de transição

    Velocity tem variação aleatória ±5-8 por nota pra evitar som mecânico.
    Determinístico (seed=42 por padrão) pra reprodutibilidade do TCC.
    """
    rng = random.Random(seed)
    beat_duration = 60.0 / tempo
    drum_events = []

    beat = 0
    t = 0.0
    while t < total_duration:
        beat_in_bar = beat % 4
        bar_count = beat // 4
        is_transition_bar = (bar_count % 4 == 3)

        # Kick nos beats 1 e 3
        if beat_in_bar in (0, 2):
            drum_events.append((t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': 36,
                'velocity': max(85, min(110, 100 + rng.randint(-5, 5))),
                'instrument': _DRUM_SLOT,
            })()))

        # Snare nos beats 2 e 4
        if beat_in_bar in (1, 3):
            drum_events.append((t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': 38,
                'velocity': max(85, min(105, 95 + rng.randint(-5, 5))),
                'instrument': _DRUM_SLOT,
            })()))

        # Hi-hat fechado nas colcheias (beat e half-beat)
        for sub in (0, 0.5):
            t_hh = t + sub * beat_duration
            if t_hh >= total_duration:
                continue
            # Open hi-hat no final do 4º bar pra marcar transição
            use_open = (is_transition_bar and beat_in_bar == 3 and sub == 0.5)
            drum_events.append((t_hh, type('E', (), {
                'event_type': 'NOTE_ON',
                'pitch': 46 if use_open else 42,
                'velocity': max(55, min(85, 70 + rng.randint(-8, 8))),
                'instrument': _DRUM_SLOT,
            })()))

        beat += 1
        t += beat_duration

    # NOTE_OFFs sintéticos — percussão soa curta (0.1s)
    closed = []
    for evt_t, evt in drum_events:
        closed.append((evt_t, evt))
        closed.append((evt_t + 0.1, type('E', (), {
            'event_type': 'NOTE_OFF', 'pitch': evt.pitch,
            'velocity': 0, 'instrument': _DRUM_SLOT,
        })()))
    return closed


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

    # Solo: quantiza pra grade rítmica de 1/8 de tempo ANTES da monofonia.
    # Sem isso o solo flutua entre beats; com snap, cada nota cai em posição
    # métrica clara — sente "tocando no compasso" e não correndo solto.
    # Após quantizar, monofonia descarta colisões (múltiplas notas no mesmo slot).
    if 102 in instrument_tracks:
        instrument_tracks[102] = _quantize_to_grid(
            instrument_tracks[102], beat_duration=beat_duration, subdivisions=2
        )

    # Bass e Solo: monofônicos — cada NOTE_ON fecha a nota anterior do mesmo
    # registro. Piano real toca melodia e linha de baixo em uma nota por vez;
    # polifonia no solo/baixo cria sensação de "notas comendo umas às outras".
    # Base permanece polifônico (acordes precisam de múltiplas notas).
    # Cap agressivo de 1.2s evita drones sustentados no solo.
    for slot in (100, 102):
        if slot in instrument_tracks:
            instrument_tracks[slot] = _enforce_monophony(
                instrument_tracks[slot], max_duration=1.2
            )

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

    # Quantiza solo (incluindo notas promovidas da base) ao grid de 1/8 de tempo
    # e re-aplica monofonia: notas promovidas vinham sem NOTE_OFF, podiam
    # sobrepor com notas do modelo. Cap em 1.0s aqui pra punch melódico.
    instrument_tracks[102] = _quantize_to_grid(
        instrument_tracks[102], beat_duration=beat_duration, subdivisions=2
    )
    instrument_tracks[102] = _enforce_monophony(
        instrument_tracks[102], max_duration=1.0
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

        # ---------- WALKING BASS ----------
        # Pattern: root → 3ª (do acorde) → 5ª → oitava (4 notas, 1 por beat).
        # 3ª usa o intervalo real do acorde (chord_pcs[1] - bass_pc) — fica
        # +4 em maior, +3 em menor — preserva a tonalidade.
        third_interval = (chord_pcs[1] - bass_pc) % 12 or 4
        # Intervalos do walking + velocidade por beat (cria groove sem groove_prob)
        walking = [
            (0,                    vel_strong if not is_breath_bar else vel_medium),
            (third_interval,       vel_soft),
            (7,                    vel_medium if not is_breath_bar else vel_soft),
            (12,                   vel_soft),
        ]

        for beat_idx, (interval, vel) in enumerate(walking):
            note_t = t + beat_idx * beat_duration
            if note_t >= total_duration:
                break
            # Mod 12 mantém estritamente em uma oitava (36-47, C2-B2).
            # Walking sai do "root-fifth" puro mas permanece no registro do baixo.
            walking_pitch = bass_octave_base + ((bass_pc + interval) % 12)
            bass_events.append((note_t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': walking_pitch,
                'velocity': vel, 'instrument': 100,
            })()))

        # ---------- BASE COM INVERSÕES ROTATIVAS ----------
        # Alterna posição fundamental (root) e 1ª inversão a cada compasso.
        # Cria movimento ascendente sutil que evita "pianola" repetitiva.
        # Normaliza chord_pcs em pcs 0-11, ordena ascendente, depois rotaciona.
        normalized = sorted(set(pc % 12 for pc in chord_pcs))
        if bar_count % 2 == 0:
            # Posição fundamental: [a, b, c]
            voicing = normalized
        else:
            # 1ª inversão: [b, c, a+12] — sobe a fundamental uma oitava
            voicing = normalized[1:] + [normalized[0] + 12]

        chord_vel_strong = vel_medium if is_breath_bar else vel_strong
        t_mid = t + bar_duration / 2

        for pc in voicing:
            chord_pitch = chord_octave_base + (pc % 24)  # %24 acomoda +12
            base_events.append((t, type('E', (), {
                'event_type': 'NOTE_ON', 'pitch': chord_pitch,
                'velocity': chord_vel_strong, 'instrument': 101,
            })()))

        # Base: stamp no tempo 3 — EXCETO em breath bar (cria respiração)
        if not is_breath_bar and t_mid < total_duration:
            for pc in voicing:
                chord_pitch = chord_octave_base + (pc % 24)
                base_events.append((t_mid, type('E', (), {
                    'event_type': 'NOTE_ON', 'pitch': chord_pitch,
                    'velocity': vel_medium, 'instrument': 101,
                })()))

        t += bar_duration
        bar_count += 1

    instrument_tracks[100] = bass_events
    instrument_tracks[101] = base_events
    return instrument_tracks


def _quantize_to_grid(events: list, beat_duration: float,
                      subdivisions: int = 2) -> list:
    """
    Snapeia cada NOTE_ON pra posição mais próxima da grade rítmica
    (default: 1/8 de tempo — 2 subdivisões por beat = 8 slots por bar 4/4).

    Resultado: melodia "respira no compasso". Notas que caíam entre beats
    passam a tocar em ponto métrico claro, alinhando o solo com bass/base.
    NOTE_OFFs ficam intactos — a duração das notas é normalizada depois pelo
    _enforce_monophony.
    """
    grid = beat_duration / subdivisions
    result = []
    for item in events:
        t, event = item
        if event.event_type == 'NOTE_ON':
            snapped = round(t / grid) * grid
            result.append((snapped, event))
        else:
            result.append(item)
    return result


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


def _enforce_monophony(events: list, max_duration: float = 1.2) -> list:
    """
    Força monofonia no registro: cada NOTE_ON fecha a nota anterior ainda ativa.
    Cap agressivo de duração: nenhuma nota dura mais que max_duration, mesmo
    sem retrigger ou NOTE_OFF natural — fundamental pra evitar sustains de
    3-5s que aparecem como "drone" no piano roll do solo.
    """
    result = []
    active_pitch = None
    active_onset = None
    last_instr = 0

    for item in sorted(events, key=lambda x: x[0]):
        t, event = item
        if event.event_type == 'NOTE_ON':
            last_instr = event.instrument
            if active_pitch is not None and active_onset is not None:
                # Sempre fecha a nota anterior antes da nova (mesmo se mesmo pitch).
                # Isso garante visual limpo no piano roll e evita merging contínuo.
                close_at = min(t, active_onset + max_duration)
                result.append((close_at, type('E', (), {
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
                active_onset = None

    # Cap a última nota se ficou pendente (garante que nenhum NOTE_ON fica órfão)
    if active_pitch is not None and active_onset is not None:
        result.append((active_onset + max_duration, type('E', (), {
            'event_type': 'NOTE_OFF', 'pitch': active_pitch,
            'velocity': 0, 'instrument': last_instr,
        })()))

    return result


def tokens_to_midi(tokens: List[int], tokenizer: MIDITokenizer,
                   output_path: str, tempo: int = 120,
                   max_note_duration: float = 1.0,
                   render_as_band: bool = False,
                   render_as_trio: bool = False,
                   solid_base: bool = False,
                   add_drums: bool = False,
                   drum_seed: int = 42,
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

        # Modo --add_drums: injeta padrão rock/pop algorítmico em canal 9 GM.
        # Compatível com qualquer modo de render — só adiciona um slot novo (200).
        # drum_seed permite que cada peça tenha variações de velocity únicas
        # (humanização) mesmo mantendo o padrão estrutural igual.
        if add_drums:
            instrument_tracks[_DRUM_SLOT] = _generate_drum_pattern(
                total_duration=current_time, tempo=tempo, seed=drum_seed,
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

            if instr_idx == _DRUM_SLOT:
                channel = 9
                program = 0  # ignorado em canal 9
            elif render_as_trio:
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
                    # Bateria (slot 200) já tem velocidades calibradas — passa direto.
                    if instr_idx == _DRUM_SLOT:
                        boosted_vel = max(1, min(127, event.velocity))
                    else:
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
