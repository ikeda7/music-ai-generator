"""
Módulo de processamento de dados MIDI.
Responsável por carregar, processar e tokenizar arquivos MIDI para treinamento.
"""

import os
import json
import mido
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class MusicalEvent:
    """Representa um evento musical."""
    event_type: str  # 'NOTE_ON', 'NOTE_OFF', 'TIME_SHIFT'
    time: float
    pitch: Optional[int] = None
    velocity: Optional[int] = None
    instrument: Optional[int] = None


class MIDIProcessor:
    """
    Processa arquivos MIDI e extrai eventos musicais multi-instrumental.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o processador MIDI.
        
        Args:
            config: Dicionário com configurações (quantization_resolution, num_instruments, etc.)
        """
        self.quantization_resolution = config['data']['quantization_resolution']
        self.num_instruments = config['data']['num_instruments']
        self.max_time_shift = config['vocab']['num_time_shifts']  # 128 steps por token TIME_SHIFT
        self.min_pitch = config['data']['min_pitch']
        self.max_pitch = config['data']['max_pitch']
        self.min_velocity = config['data']['min_velocity']
        self.max_velocity = config['data']['max_velocity']
        
        # Mapeamento de instrumentos MIDI para nossos slots
        self.instrument_map = self._create_instrument_map()
    
    def _create_instrument_map(self) -> Dict[int, int]:
        """
        Cria mapeamento de programas MIDI para slots de instrumentos.
        
        Retorna:
            Dicionário mapeando programa MIDI -> slot de instrumento (0-4)
        """
        # Piano: 0-7, Melodic: 24-31, Bass: 32-39, Drums: 128, Harmony: 48-55
        mapping = {}
        # Piano (slot 0)
        for prog in range(0, 8):
            mapping[prog] = 0
        # Melodia (slot 1)
        for prog in range(24, 32):
            mapping[prog] = 1
        # Baixo (slot 2)
        for prog in range(32, 40):
            mapping[prog] = 2
        # Bateria (slot 3)
        mapping[128] = 3
        # Harmonia (slot 4)
        for prog in range(48, 56):
            mapping[prog] = 4
        
        # Fallback: distribuir outros instrumentos
        for prog in range(128):
            if prog not in mapping:
                mapping[prog] = prog % self.num_instruments
        
        return mapping
    
    def load_midi(self, file_path: str) -> List[MusicalEvent]:
        """
        Carrega um arquivo MIDI e extrai eventos musicais.

        Args:
            file_path: Caminho para o arquivo MIDI

        Retorna:
            Lista de eventos musicais ordenados por tempo
        """
        try:
            mid = mido.MidiFile(file_path)
            ticks_per_beat = mid.ticks_per_beat
            events = []

            # Constrói mapa de tempo: lista de (tick_absoluto, microsegundos_por_beat)
            # Necessário para converter ticks -> segundos corretamente com mudanças de BPM
            tempo_map = [(0, 500000)]  # padrão: 120 BPM
            for track in mid.tracks:
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    if msg.type == 'set_tempo':
                        tempo_map.append((abs_tick, msg.tempo))
            tempo_map.sort(key=lambda x: x[0])

            def ticks_to_seconds(abs_tick: int) -> float:
                """Converte tick absoluto para segundos usando o mapa de tempo."""
                seconds = 0.0
                prev_tick, current_tempo = tempo_map[0]
                for i in range(1, len(tempo_map)):
                    seg_tick, next_tempo = tempo_map[i]
                    if abs_tick <= seg_tick:
                        break
                    seconds += mido.tick2second(seg_tick - prev_tick, ticks_per_beat, current_tempo)
                    prev_tick, current_tempo = seg_tick, next_tempo
                seconds += mido.tick2second(abs_tick - prev_tick, ticks_per_beat, current_tempo)
                return seconds

            # Processa cada track convertendo ticks para segundos
            for track in mid.tracks:
                abs_tick = 0
                current_instrument = 0

                for msg in track:
                    abs_tick += msg.time

                    if msg.type == 'program_change':
                        current_instrument = self.instrument_map.get(msg.program, 0)

                    elif msg.type == 'note_on' and msg.velocity > 0:
                        events.append(MusicalEvent(
                            event_type='NOTE_ON',
                            time=ticks_to_seconds(abs_tick),
                            pitch=msg.note,
                            velocity=msg.velocity,
                            instrument=current_instrument
                        ))

                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        events.append(MusicalEvent(
                            event_type='NOTE_OFF',
                            time=ticks_to_seconds(abs_tick),
                            pitch=msg.note,
                            velocity=0,
                            instrument=current_instrument
                        ))

            # Ordena eventos por tempo
            events.sort(key=lambda x: x.time)

            # Quantiza e adiciona TIME_SHIFT
            quantized_events = self._quantize_and_add_time_shifts(events)

            return quantized_events

        except Exception as e:
            print(f"Erro ao carregar MIDI {file_path}: {e}")
            return []
    
    def _quantize_and_add_time_shifts(self, events: List[MusicalEvent]) -> List[MusicalEvent]:
        """
        Quantiza eventos temporais e adiciona tokens TIME_SHIFT.

        O campo `time` de cada evento TIME_SHIFT armazena o número de steps (1..128),
        não o tempo absoluto. Isso permite encode/decode correto e tokens mais compactos.

        Args:
            events: Lista de eventos musicais com tempo em segundos

        Retorna:
            Lista de eventos quantizados com TIME_SHIFT intercalados
        """
        if not events:
            return []

        quantized = []
        last_time = 0.0
        step_duration = 1.0 / self.quantization_resolution  # segundos por step
        for event in events:
            # Quantiza o tempo para o step mais próximo
            quantized_time = round(event.time * self.quantization_resolution) / self.quantization_resolution
            time_diff = quantized_time - last_time

            # Emite TIME_SHIFT(s) cobrindo o intervalo em chunks de até max_time_shift steps
            if time_diff > step_duration * 0.5:  # tolerância de meio step
                num_steps = max(1, round(time_diff / step_duration))
                remaining = num_steps
                while remaining > 0:
                    chunk = min(remaining, self.max_time_shift)  # máximo de steps por token
                    quantized.append(MusicalEvent(
                        event_type='TIME_SHIFT',
                        time=float(chunk),  # armazena a contagem de steps, não tempo absoluto
                        instrument=0
                    ))
                    remaining -= chunk
                last_time = quantized_time

            quantized.append(event)

        return quantized
    
    def load_midi_dataset(self, dataset_path: str) -> List[List[MusicalEvent]]:
        """
        Carrega múltiplos arquivos MIDI de um diretório.
        
        Args:
            dataset_path: Caminho para diretório com arquivos MIDI
            
        Retorna:
            Lista de listas de eventos (uma lista por arquivo)
        """
        midi_files = []
        
        if os.path.isfile(dataset_path):
            # Se for um arquivo único
            midi_files = [dataset_path]
        elif os.path.isdir(dataset_path):
            # Se for um diretório, busca todos os .mid e .midi
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.mid', '.midi')):
                        midi_files.append(os.path.join(root, file))
        
        all_events = []
        for midi_file in midi_files:
            events = self.load_midi(midi_file)
            if events:
                all_events.append(events)
        
        print(f"Carregados {len(all_events)} arquivos MIDI")
        return all_events


class MIDITokenizer:
    """
    Tokeniza eventos musicais em sequências de tokens para o Transformer.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o tokenizador.
        
        Args:
            config: Dicionário com configurações (vocab, data, etc.)
        """
        self.config = config
        self.special_tokens = config['vocab']['special_tokens']
        self.num_time_shifts = config['vocab']['num_time_shifts']
        self.num_velocities = config['vocab']['num_velocities']
        self.num_pitches = config['vocab']['num_pitches']
        self.num_instruments = config['data']['num_instruments']
        
        # Atributos de pitch e velocity do config
        self.min_pitch = config['data']['min_pitch']
        self.max_pitch = config['data']['max_pitch']
        self.min_velocity = config['data']['min_velocity']
        self.max_velocity = config['data']['max_velocity']
        
        # Cria vocabulário
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def _build_vocab(self) -> Dict[str, int]:
        """
        Constrói o vocabulário de tokens.
        
        Retorna:
            Dicionário mapeando token -> id
        """
        vocab = {}
        idx = 0
        
        # Tokens especiais
        for token_name, token_id in self.special_tokens.items():
            vocab[token_name] = token_id
            idx = max(idx, token_id)
        
        idx = max(idx, max(self.special_tokens.values())) + 1
        
        # Tokens de instrumento
        for i in range(self.num_instruments):
            vocab[f'INSTRUMENT_{i}'] = idx
            idx += 1
        
        # Tokens de pitch (NOTE_ON e NOTE_OFF)
        for pitch in range(self.min_pitch, self.max_pitch + 1):
            vocab[f'NOTE_ON_{pitch}'] = idx
            idx += 1
            vocab[f'NOTE_OFF_{pitch}'] = idx
            idx += 1
        
        # Tokens de velocity (quantizados em num_velocities bins uniformes)
        for i in range(self.num_velocities):
            vocab[f'VELOCITY_{i}'] = idx
            idx += 1
        
        # Tokens de TIME_SHIFT
        for i in range(1, self.num_time_shifts + 1):
            vocab[f'TIME_SHIFT_{i}'] = idx
            idx += 1

        # Tokens de estrutura rítmica (Bar-Relative Encoding)
        # BAR = início de compasso; BEAT_2/3/4 = tempos 2, 3, 4 do compasso (4/4)
        for token in ('BAR', 'BEAT_2', 'BEAT_3', 'BEAT_4'):
            vocab[token] = idx
            idx += 1

        return vocab
    
    def encode_events(self, events: List[MusicalEvent]) -> List[int]:
        """
        Converte eventos musicais em sequência de tokens.
        
        Args:
            events: Lista de eventos musicais
            
        Retorna:
            Lista de ids de tokens
        """
        tokens = [self.vocab['BOS']]  # Início da sequência

        # Bar-Relative Encoding: a cada 32 steps (1 compasso em 120 BPM, resolução 16)
        # insere BAR/BEAT_X para o modelo aprender estrutura métrica.
        # 1 beat = resolution/2 = 8 steps; 1 compasso (4/4) = 32 steps.
        _beat_steps = self.config['data']['quantization_resolution'] // 2  # 8 steps por beat
        _bar_steps  = _beat_steps * 4                    # 32 steps por compasso
        _beat_tokens = ('BAR', 'BEAT_2', 'BEAT_3', 'BEAT_4')
        _abs_step = 0
        _last_boundary = 0

        # Marca início do primeiro compasso
        if 'BAR' in self.vocab:
            tokens.append(self.vocab['BAR'])

        current_instrument = None

        for event in events:
            if event.event_type == 'TIME_SHIFT':
                # event.time armazena o número de steps (não tempo absoluto)
                steps = int(event.time)
                if steps > 0 and steps <= self.num_time_shifts:
                    tokens.append(self.vocab.get(f'TIME_SHIFT_{steps}', self.vocab['PAD']))

                _abs_step += steps

                # Emite BAR/BEAT para cada fronteira de tempo cruzada
                _next_b = (_last_boundary // _beat_steps + 1) * _beat_steps
                while _abs_step >= _next_b and 'BAR' in self.vocab:
                    beat_idx = (_next_b // _beat_steps) % 4  # 0=BAR, 1=BEAT_2, 2=BEAT_3, 3=BEAT_4
                    tok = _beat_tokens[beat_idx]
                    if tok in self.vocab:
                        tokens.append(self.vocab[tok])
                    _last_boundary = _next_b
                    _next_b += _beat_steps
            
            elif event.event_type == 'NOTE_ON':
                # Adiciona token de instrumento se mudou
                if current_instrument != event.instrument:
                    instr_token = f'INSTRUMENT_{event.instrument}'
                    if instr_token in self.vocab:
                        tokens.append(self.vocab[instr_token])
                    current_instrument = event.instrument
                
                # Adiciona token de pitch
                pitch_token = f'NOTE_ON_{event.pitch}'
                if pitch_token in self.vocab:
                    tokens.append(self.vocab[pitch_token])
                
                # Adiciona token de velocity (quantizada)
                velocity_bin = int((event.velocity - self.config['data']['min_velocity']) / 
                                  (self.config['data']['max_velocity'] - self.config['data']['min_velocity']) * 
                                  (self.num_velocities - 1))
                velocity_bin = max(0, min(self.num_velocities - 1, velocity_bin))
                tokens.append(self.vocab[f'VELOCITY_{velocity_bin}'])
            
            elif event.event_type == 'NOTE_OFF':
                # Similar ao NOTE_ON, mas com NOTE_OFF
                if current_instrument != event.instrument:
                    instr_token = f'INSTRUMENT_{event.instrument}'
                    if instr_token in self.vocab:
                        tokens.append(self.vocab[instr_token])
                    current_instrument = event.instrument
                
                pitch_token = f'NOTE_OFF_{event.pitch}'
                if pitch_token in self.vocab:
                    tokens.append(self.vocab[pitch_token])
        
        tokens.append(self.vocab['EOS'])  # Fim da sequência
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> List[MusicalEvent]:
        """
        Converte sequência de tokens de volta para eventos musicais.
        
        Args:
            tokens: Lista de ids de tokens
            
        Retorna:
            Lista de eventos musicais
        """
        events = []
        current_time = 0.0
        current_instrument = 0
        ticks_per_step = 1.0 / self.config['data']['quantization_resolution']
        
        i = 0
        while i < len(tokens):
            token_id = tokens[i]
            token = self.id_to_token.get(token_id, 'UNK')
            
            if token in ('BOS', 'PAD', 'BAR', 'BEAT_2', 'BEAT_3', 'BEAT_4'):
                i += 1
                continue
            elif token == 'EOS':
                break
            elif token.startswith('INSTRUMENT_'):
                current_instrument = int(token.split('_')[1])
                i += 1
            elif token.startswith('TIME_SHIFT_'):
                steps = int(token.split('_')[2])
                current_time += steps * ticks_per_step
                events.append(MusicalEvent(
                    event_type='TIME_SHIFT',
                    time=current_time,
                    instrument=0
                ))
                i += 1
            elif token.startswith('NOTE_ON_'):
                pitch = int(token.split('_')[2])
                # Tenta ler VELOCITY do próximo token; usa 64 (mf) se ausente
                velocity = 64
                advance = 1
                if i + 1 < len(tokens):
                    velocity_token = self.id_to_token.get(tokens[i + 1], '')
                    if velocity_token.startswith('VELOCITY_'):
                        velocity_bin = int(velocity_token.split('_')[1])
                        velocity = int(self.config['data']['min_velocity'] +
                                     velocity_bin * (self.config['data']['max_velocity'] -
                                                    self.config['data']['min_velocity']) /
                                     (self.num_velocities - 1))
                        advance = 2
                events.append(MusicalEvent(
                    event_type='NOTE_ON',
                    time=current_time,
                    pitch=pitch,
                    velocity=velocity,
                    instrument=current_instrument
                ))
                i += advance
            elif token.startswith('NOTE_OFF_'):
                pitch = int(token.split('_')[2])
                events.append(MusicalEvent(
                    event_type='NOTE_OFF',
                    time=current_time,
                    pitch=pitch,
                    velocity=0,
                    instrument=current_instrument
                ))
                i += 1
            else:
                i += 1
        
        return events


def prepare_sequences(tokenized_data: List[List[int]], seq_length: int, 
                     vocab_size: int, pad_token: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara sequências tokenizadas para treinamento.
    
    Args:
        tokenized_data: Lista de sequências tokenizadas
        seq_length: Comprimento máximo das sequências
        vocab_size: Tamanho do vocabulário
        pad_token: Token usado para padding
        
    Retorna:
        Tupla (input_sequences, target_sequences) como arrays numpy
    """
    input_seqs = []
    target_seqs = []
    
    for seq in tokenized_data:
        # Divide sequência em chunks de tamanho seq_length
        for i in range(0, len(seq) - 1, seq_length // 2):  # Overlap de 50%
            chunk = seq[i:i + seq_length + 1]
            
            if len(chunk) < 2:
                continue
            
            # Input é tudo exceto o último token
            # Target é tudo exceto o primeiro token (shifted)
            input_seq = chunk[:-1]
            target_seq = chunk[1:]
            
            # Padding
            input_seq = input_seq + [pad_token] * (seq_length - len(input_seq))
            target_seq = target_seq + [pad_token] * (seq_length - len(target_seq))
            
            input_seqs.append(input_seq[:seq_length])
            target_seqs.append(target_seq[:seq_length])
    
    return np.array(input_seqs, dtype=np.int64), np.array(target_seqs, dtype=np.int64)

