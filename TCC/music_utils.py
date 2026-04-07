"""
Utilitários musicais para conversão entre tokens e MIDI.
"""

import mido
from typing import List
from data_processor import MIDITokenizer

# Programa General MIDI por slot de instrumento
_INSTRUMENT_PROGRAM = {
    0: 0,   # Piano (Acoustic Grand Piano)
    1: 24,  # Melody (Nylon Guitar)
    2: 32,  # Bass (Acoustic Bass)
    3: 0,   # Bateria — canal 9 não usa program_change
    4: 48,  # Harmony (String Ensemble)
}

# Canal MIDI por slot. Bateria deve estar no canal 9 (General MIDI).
_INSTRUMENT_CHANNEL = {0: 0, 1: 1, 2: 2, 3: 9, 4: 3}


def tokens_to_midi(tokens: List[int], tokenizer: MIDITokenizer,
                   output_path: str, tempo: int = 120,
                   max_note_duration: float = 1.5) -> bool:
    """
    Converte uma sequência de tokens em um arquivo MIDI.
    
    Args:
        tokens: Lista de ids de tokens
        tokenizer: Instância do MIDITokenizer
        output_path: Caminho para salvar o arquivo MIDI
        tempo: Tempo em BPM (batidas por minuto)
        
    Retorna:
        True se bem-sucedido, False caso contrário
    """
    try:
        # Decodifica tokens para eventos
        events = tokenizer.decode_tokens(tokens)
        
        # Cria arquivo MIDI
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # Configura tempo
        ticks_per_beat = mid.ticks_per_beat
        microseconds_per_beat = mido.bpm2tempo(tempo)
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        
        # Agrupa apenas NOTE_ON por instrumento.
        # Todos os NOTE_OFF são gerados pelo hanging-note fix abaixo,
        # garantindo que nenhuma nota ultrapasse max_note_duration.
        instrument_tracks = {}
        current_time = 0.0

        for event in events:
            if event.event_type == 'TIME_SHIFT':
                current_time = event.time
            elif event.event_type == 'NOTE_ON':
                instr = event.instrument
                if instr not in instrument_tracks:
                    instrument_tracks[instr] = []
                instrument_tracks[instr].append((current_time, event))
        
        # Fecha notas abertas sem NOTE_OFF correspondente (capping em max_note_duration)
        for instr_idx, instr_events in instrument_tracks.items():
            open_at = {}  # pitch → on_time
            extras = []
            for event_time, event in sorted(instr_events, key=lambda x: x[0]):
                if event.event_type == 'NOTE_ON':
                    # Se já estava aberta, fecha antes de reabrir
                    if event.pitch in open_at:
                        forced_off = min(open_at[event.pitch] + max_note_duration, event_time)
                        extras.append((forced_off, type('E', (), {
                            'event_type': 'NOTE_OFF', 'pitch': event.pitch,
                            'velocity': 0, 'instrument': instr_idx,
                        })()))
                    open_at[event.pitch] = event_time
                elif event.event_type == 'NOTE_OFF' and event.pitch in open_at:
                    del open_at[event.pitch]
            # Fecha notas ainda abertas ao final
            for pitch, on_time in open_at.items():
                extras.append((on_time + max_note_duration, type('E', (), {
                    'event_type': 'NOTE_OFF', 'pitch': pitch,
                    'velocity': 0, 'instrument': instr_idx,
                })()))
            # NOTE_OFF deve vir antes de NOTE_ON no mesmo instante (evita notas contínuas longas)
            def _sort_key(x):
                t, e = x
                return (t, 0 if e.event_type == 'NOTE_OFF' else 1)
            instrument_tracks[instr_idx] = sorted(instr_events + extras, key=_sort_key)

        # Cria tracks separados para cada instrumento
        for instr_idx, instr_events in instrument_tracks.items():
            instr_track = mido.MidiTrack()
            mid.tracks.append(instr_track)

            channel = _INSTRUMENT_CHANNEL.get(instr_idx, instr_idx % 9)
            # Canal 9 é bateria — não envia program_change
            if channel != 9:
                program = _INSTRUMENT_PROGRAM.get(instr_idx, 0)
                instr_track.append(mido.Message('program_change', channel=channel, program=program))
            
            # Ordena eventos por tempo; NOTE_OFF antes de NOTE_ON no mesmo instante
            instr_events.sort(key=lambda x: (x[0], 0 if x[1].event_type == 'NOTE_OFF' else 1))
            
            last_time = 0.0
            for event_time, event in instr_events:
                delta_ticks = int((event_time - last_time) * ticks_per_beat * tempo / 60)

                if event.event_type == 'NOTE_ON':
                    instr_track.append(mido.Message(
                        'note_on',
                        channel=channel,
                        note=event.pitch,
                        velocity=event.velocity,
                        time=delta_ticks
                    ))
                elif event.event_type == 'NOTE_OFF':
                    instr_track.append(mido.Message(
                        'note_off',
                        channel=channel,
                        note=event.pitch,
                        velocity=0,
                        time=delta_ticks
                    ))
                
                last_time = event_time
        
        # Salva arquivo
        mid.save(output_path)
        print(f"MIDI salvo em: {output_path}")
        return True
        
    except Exception as e:
        print(f"Erro ao converter tokens para MIDI: {e}")
        return False



