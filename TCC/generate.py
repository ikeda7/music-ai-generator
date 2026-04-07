"""
Script de geração de música usando modelo Transformer treinado.
Suporta filtro de escala musical via --key para garantir coerência harmônica.
"""

import os
import json
import torch
import argparse
import numpy as np

from data_processor import MIDITokenizer
from model import MultiInstrumentTransformer
from music_utils import tokens_to_midi


# ---------------------------------------------------------------------------
# Teoria musical: escalas e detecção de tonalidade
# ---------------------------------------------------------------------------

_NOTE_MAP = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
}
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],  # natural minor
}

# Perfis de Krumhansl-Schmuckler para detecção automática de tonalidade
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _parse_key(key_str: str):
    """
    Interpreta string de tonalidade (ex: 'C', 'Am', 'F#m', 'Bb').
    Retorna (root_semitone: int, scale_type: str).
    """
    key_str = key_str.strip()
    minor = key_str.endswith('m') and len(key_str) > 1
    root_str = key_str[:-1] if minor else key_str
    if root_str not in _NOTE_MAP:
        raise ValueError(
            f"Tonalidade inválida: '{key_str}'. "
            f"Use ex: C, D, F#, Bb (maior) ou Am, Em, C#m (menor)."
        )
    return _NOTE_MAP[root_str], 'minor' if minor else 'major'


def build_note_mask(key_str: str, tokenizer: MIDITokenizer, device: torch.device) -> torch.Tensor:
    """
    Retorna tensor de shape (vocab_size,) com -inf para tokens NOTE_ON
    cujo pitch está fora da escala especificada. Demais tokens recebem 0.
    """
    root, scale_type = _parse_key(key_str)
    scale_pcs = {(root + i) % 12 for i in _SCALES[scale_type]}

    mask = torch.zeros(tokenizer.vocab_size)
    for token, token_id in tokenizer.vocab.items():
        if token.startswith('NOTE_ON_'):
            pitch = int(token.split('_')[2])
            if pitch % 12 not in scale_pcs:
                mask[token_id] = float('-inf')
    return mask.to(device)


def detect_key(model, tokenizer: MIDITokenizer, config: dict,
               device: torch.device) -> str:
    """
    Gera ~150 tokens sem restrição e detecta a tonalidade predominante
    usando correlação com perfis de Krumhansl-Schmuckler.
    """
    bos = config['vocab']['special_tokens']['BOS']
    eos = config['vocab']['special_tokens']['EOS']
    inp = torch.LongTensor([[bos]]).to(device)

    with torch.no_grad():
        out = model.generate(
            inp, max_length=150, temperature=0.9, top_k=30,
            top_p=0.95, eos_token_id=eos,
            context_size=config['data']['seq_length'],
        )

    tokens = out[0].cpu().tolist()
    pitches = [
        int(tokenizer.id_to_token[t].split('_')[2]) % 12
        for t in tokens
        if tokenizer.id_to_token.get(t, '').startswith('NOTE_ON_')
    ]

    if not pitches:
        print("  Detecção de tonalidade: sem notas — usando C maior como padrão.")
        return 'C'

    pc_hist = [pitches.count(i) for i in range(12)]

    best_score, best_key = -float('inf'), 'C'
    for root in range(12):
        for label, profile in [('', _KS_MAJOR), ('m', _KS_MINOR)]:
            rotated = profile[root:] + profile[:root]
            score = sum(pc_hist[i] * rotated[i] for i in range(12))
            if score > best_score:
                best_score = score
                best_key = _NOTE_NAMES[root] + label

    return best_key


def load_config(config_path: str = 'config.json') -> dict:
    """
    Carrega configurações do arquivo JSON.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Retorna:
        Dicionário com configurações
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """
    Carrega modelo a partir de um checkpoint.
    
    Args:
        checkpoint_path: Caminho para o checkpoint
        config: Configurações
        device: Device (CPU/GPU)
        
    Retorna:
        Tupla (modelo, tokenizer, configurações do checkpoint)
    """
    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recria tokenizer com vocabulário salvo
    tokenizer = MIDITokenizer(config)
    if 'vocab' in checkpoint:
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
    
    # Recria modelo
    model_config = config['model']
    vocab_size = checkpoint.get('vocab_size', tokenizer.vocab_size)
    
    model = MultiInstrumentTransformer(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        dropout=model_config['dropout'],
        max_seq_length=model_config['max_seq_length'],
        num_instruments=config['data']['num_instruments']
    ).to(device)
    
    # Carrega pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Modelo carregado com sucesso!")
    return model, tokenizer


def build_vocab_constraint(tokenizer: MIDITokenizer, device: torch.device,
                           notes_per_instrument: int = 16):
    """
    Retorna função de restrição de vocabulário com duas regras:
    1. VELOCITY só é permitido imediatamente após NOTE_ON.
    2. A cada `notes_per_instrument` notas geradas, força troca de instrumento
       no próximo ponto de quebra (TIME_SHIFT ou NOTE_OFF), ciclando pelos 5 slots.
    """
    velocity_ids  = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('VELOCITY_'))
    note_on_ids   = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('NOTE_ON_'))
    time_shift_ids = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('TIME_SHIFT_'))
    note_off_ids  = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('NOTE_OFF_'))

    # Instrumentos disponíveis em ordem (Piano, Melodia, Baixo, Bateria, Harmonia)
    instr_ids = [tokenizer.vocab[f'INSTRUMENT_{i}']
                 for i in range(5) if f'INSTRUMENT_{i}' in tokenizer.vocab]

    # Pré-computa máscaras
    velocity_block = torch.zeros(tokenizer.vocab_size, device=device)
    for tid in velocity_ids:
        velocity_block[tid] = float('-inf')

    zero = torch.zeros(tokenizer.vocab_size, device=device)

    # Máscara por instrumento: força um instrumento específico
    instr_masks = []
    for iid in instr_ids:
        m = torch.full((tokenizer.vocab_size,), float('-inf'), device=device)
        m[iid] = 0.0
        instr_masks.append(m)

    # Estado mutável
    state = {'note_count': 0, 'next_instr': 1, 'force': False}

    def constraint_fn(last_token_id: int) -> torch.Tensor:
        token = tokenizer.id_to_token.get(last_token_id, '')

        # Após INSTRUMENT: reseta contador, limpa flag
        if token.startswith('INSTRUMENT_'):
            state['note_count'] = 0
            state['force'] = False
            return velocity_block  # VELOCITY não faz sentido após INSTRUMENT

        # Se flag de troca ativa e chegamos num ponto de quebra: força INSTRUMENT
        if state['force'] and (last_token_id in time_shift_ids or last_token_id in note_off_ids):
            idx = state['next_instr'] % len(instr_ids)
            state['next_instr'] += 1
            state['force'] = False
            return instr_masks[idx]

        # Após NOTE_ON: permite VELOCITY e incrementa contador
        if last_token_id in note_on_ids:
            state['note_count'] += 1
            if state['note_count'] >= notes_per_instrument and instr_ids:
                state['force'] = True  # troca no próximo ponto de quebra
                state['note_count'] = 0
            return zero  # permite VELOCITY

        return velocity_block

    return constraint_fn


def generate_music(model: MultiInstrumentTransformer, tokenizer: MIDITokenizer,
                   config: dict, device: torch.device,
                   priming_sequence: list = None,
                   max_length: int = None,
                   temperature: float = None,
                   top_k: int = None,
                   top_p: float = None,
                   note_mask: torch.Tensor = None) -> list:
    """
    Gera uma sequência musical.

    Args:
        model: Modelo Transformer
        tokenizer: Tokenizador
        config: Configurações
        device: Device (CPU/GPU)
        priming_sequence: Sequência inicial (opcional)
        max_length: Comprimento máximo da geração
        temperature: Temperatura para sampling
        top_k: Top-k sampling
        top_p: Nucleus sampling

    Retorna:
        Lista de tokens gerados
    """
    gen_config = config['generation']

    # Usa valores padrão se não especificados
    max_length = max_length or gen_config['max_length']
    temperature = temperature if temperature is not None else gen_config['temperature']
    top_k = top_k if top_k is not None else gen_config['top_k']
    top_p = top_p if top_p is not None else gen_config['top_p']

    eos_token_id = config['vocab']['special_tokens']['EOS']
    bos_token_id = config['vocab']['special_tokens']['BOS']

    # Prepara sequência inicial
    if priming_sequence:
        input_ids = torch.LongTensor([priming_sequence]).to(device)
    else:
        # Começa com BOS
        input_ids = torch.LongTensor([[bos_token_id]]).to(device)

    print(f"Gerando música (max_length={max_length}, temperature={temperature})...")

    # Restrição: VELOCITY só pode aparecer após NOTE_ON (evita loop de velocity)
    vocab_constraint_fn = build_vocab_constraint(tokenizer, device)

    # Gera sequência
    context_size = config['data']['seq_length']
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            context_size=context_size,
            note_mask=note_mask,
            vocab_constraint_fn=vocab_constraint_fn,
        )
    
    # Converte para lista
    generated_tokens = generated[0].cpu().tolist()
    
    # Remove tokens de padding
    pad_token_id = config['vocab']['special_tokens']['PAD']
    generated_tokens = [t for t in generated_tokens if t != pad_token_id]
    
    print(f"Sequência gerada com {len(generated_tokens)} tokens")
    return generated_tokens


def main():
    parser = argparse.ArgumentParser(description='Gera música usando modelo Transformer treinado')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Caminho para checkpoint do modelo')
    parser.add_argument('--output', type=str, default='generated_music.mid',
                       help='Caminho para salvar o arquivo MIDI gerado')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--length', type=int, default=None,
                       help='Comprimento máximo da geração (padrão: do config)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperatura para sampling (padrão: do config)')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling (padrão: do config)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p (nucleus) sampling (padrão: do config)')
    parser.add_argument('--priming', type=str, default=None,
                       help='Arquivo MIDI para usar como priming (opcional)')
    parser.add_argument('--num_generations', type=int, default=1,
                       help='Número de músicas para gerar')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device para geração')
    parser.add_argument('--key', type=str, default=None,
                       help='Tonalidade para filtro de escala (ex: C, Am, F#, Bbm). '
                            'Garante coerência harmônica restringindo notas à escala.')
    parser.add_argument('--auto_key', action='store_true',
                       help='Detecta automaticamente a tonalidade antes de gerar.')

    args = parser.parse_args()
    
    # Carrega configurações
    config = load_config(args.config)
    
    # Determina device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Usando device: {device}")
    
    # Carrega modelo
    model, tokenizer = load_model(args.checkpoint, config, device)

    # Filtro de escala musical
    note_mask = None
    key_label = None
    if args.key:
        key_label = args.key
        note_mask = build_note_mask(key_label, tokenizer, device)
        print(f"Filtro de escala: {key_label}")
    elif args.auto_key:
        print("Detectando tonalidade automaticamente...")
        key_label = detect_key(model, tokenizer, config, device)
        note_mask = build_note_mask(key_label, tokenizer, device)
        print(f"Tonalidade detectada: {key_label}")

    # Processa priming se especificado
    priming_sequence = None
    if args.priming:
        from data_processor import MIDIProcessor
        processor = MIDIProcessor(config)
        events = processor.load_midi(args.priming)
        if events:
            priming_sequence = tokenizer.encode_events(events)
            print(f"Priming carregado com {len(priming_sequence)} tokens")
            # Limita tamanho do priming
            max_priming_len = config['data']['seq_length'] // 4
            if len(priming_sequence) > max_priming_len:
                priming_sequence = priming_sequence[:max_priming_len]
    
    # Gera músicas
    for i in range(args.num_generations):
        print(f"\n=== Geração {i + 1}/{args.num_generations} ===")
        
        # Gera sequência
        generated_tokens = generate_music(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            priming_sequence=priming_sequence,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            note_mask=note_mask,
        )
        
        # Salva MIDI
        if args.num_generations > 1:
            base_name = os.path.splitext(args.output)[0]
            ext = os.path.splitext(args.output)[1]
            output_path = f"{base_name}_{i+1}{ext}"
        else:
            output_path = args.output
        
        success = tokens_to_midi(
            tokens=generated_tokens,
            tokenizer=tokenizer,
            output_path=output_path,
            tempo=120
        )
        
        if success:
            print(f"Música gerada salva em: {output_path}")
        else:
            print(f"Erro ao salvar música em: {output_path}")
    
    print("\nGeração concluída!")


if __name__ == '__main__':
    main()

