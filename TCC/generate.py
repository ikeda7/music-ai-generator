"""
Script de geração de música usando modelo Transformer treinado.
Suporta filtro de escala musical via --key para garantir coerência harmônica.
"""

import os
import json
import torch
import argparse

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
                           key_str: str = None):
    """
    Restrições aplicadas aos logits durante a geração — modo PIANO SOLO com 3 vozes.

    Estratégia: todo output vai pro INSTRUMENT_0 (Piano GM). A separação em vozes
    (solo/base/baixo) emerge via registros de pitch, não via tokens de instrumento.
    Isso alinha com MAESTRO (74h de piano solo com mão direita + mão esquerda).

    Registros:
      - Baixo:    MIDI 21–47  (A0–B2)    — mão esquerda, notas-raiz e oitavas
      - Base:     MIDI 48–65  (C3–F4)    — acordes/acompanhamento
      - Solo:     MIDI 66–108 (F#4–C8)   — melodia

    Constraints ativas:
      1. VELOCITY só após NOTE_ON                    (gramatical)
      2. Força INSTRUMENT_0; bloqueia INSTRUMENT_1..4 (piano único)
      3. Voice leading                               (musical, soft)
      4. Repetition penalty (-1.0)                   (comportamental)
      5. Bloqueio de EOS nos primeiros 300 tokens    (técnico)
      6. Chord backbone I-V-vi-IV no registro Base   (musical, opt-in via --key)
      7. Re-entry bias por REGISTRO silente          (polifônico, bônus positivo)
    """
    from collections import deque

    velocity_ids   = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('VELOCITY_'))
    note_on_ids    = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('NOTE_ON_'))
    time_shift_ids = frozenset(v for k, v in tokenizer.vocab.items() if k.startswith('TIME_SHIFT_'))
    eos_id         = tokenizer.vocab.get('EOS', 2)
    bar_id         = tokenizer.vocab.get('BAR')

    vocab_size = tokenizer.vocab_size
    zero = torch.zeros(vocab_size, device=device)

    # #1 Bloqueio de VELOCITY (liberado apenas logo após NOTE_ON)
    velocity_block = torch.zeros(vocab_size, device=device)
    for tid in velocity_ids:
        velocity_block[tid] = float('-inf')

    # #2 Força INSTRUMENT_0 (Piano): bloqueia todos os outros tokens INSTRUMENT_*.
    # O canal 0 é GM Piano e suporta polifonia nativa — as 3 vozes (solo/base/baixo)
    # saem naturalmente dos registros de pitch aprendidos do MAESTRO.
    instr_block = torch.zeros(vocab_size, device=device)
    for i in range(5):
        key = f'INSTRUMENT_{i}'
        if key in tokenizer.vocab and i != 0:
            instr_block[tokenizer.vocab[key]] = float('-inf')

    # Mapeamento pitch → registro (0=baixo, 1=base, 2=solo)
    def _pitch_register(pitch: int) -> int:
        if pitch <= 47:  return 0   # Baixo
        if pitch <= 65:  return 1   # Base/Harmonia
        return 2                    # Solo/Melodia

    # Tokens NOTE_ON agrupados por registro (pra re-entry bias)
    register_tokens = {0: [], 1: [], 2: []}
    note_on_by_pitch = {}
    for k, v in tokenizer.vocab.items():
        if k.startswith('NOTE_ON_'):
            pitch = int(k.split('_')[2])
            note_on_by_pitch[pitch] = v
            register_tokens[_pitch_register(pitch)].append(v)

    # Parâmetros do re-entry bias por registro
    silence_threshold = 60    # ~3.75s a 16 steps/s — registros reagem mais rápido que slots
    silence_max_bonus = 1.5   # teto; sobe de 1.2 pra chamar vozes ausentes com mais força
    silence_slope    = 0.03   # rampa linear: satura em ~110 tokens de silêncio

    # Máscaras de re-entry: bônus em todos os NOTE_ON do registro silente
    register_masks = {}
    for reg, tids in register_tokens.items():
        m = torch.zeros(vocab_size, device=device)
        for tid in tids:
            m[tid] = 1.0  # unitário; multiplicado pelo bônus calculado em runtime
        register_masks[reg] = m

    # #5 Bloqueio de EOS nos primeiros tokens
    eos_block = torch.zeros(vocab_size, device=device)
    eos_block[eos_id] = float('-inf')

    # #3 Voice leading soft: penaliza saltos melódicos grandes.
    # Aplicado com threshold dinâmico conforme o registro do último pitch:
    # solo usa 7 semitons (5ª justa); baixo/base usam 12 semitons (oitava).
    interval_masks_tight = {}   # solo (passos conjuntos)
    interval_masks_wide  = {}   # baixo/base (saltos ok, oitava penalizada)
    for src_pitch in range(21, 109):
        m_tight = torch.zeros(vocab_size, device=device)
        m_wide  = torch.zeros(vocab_size, device=device)
        for dst_pitch, tid in note_on_by_pitch.items():
            jump = abs(dst_pitch - src_pitch)
            if jump > 7:
                m_tight[tid] = -0.6 - (jump - 7) * 0.1
            if jump > 12:
                m_wide[tid]  = -0.8 - (jump - 12) * 0.1
        interval_masks_tight[src_pitch] = m_tight
        interval_masks_wide[src_pitch]  = m_wide

    # #6 Chord backbone I-V-vi-IV aplicado ao REGISTRO Base (48–65).
    # Penaliza NOTE_ON do registro Base fora do acorde do compasso atual.
    chord_masks_base = None
    if key_str and bar_id is not None:
        root, _ = _parse_key(key_str)
        _progression = [
            {root % 12,      (root+4) % 12, (root+7) % 12},   # I
            {(root+7) % 12, (root+11) % 12, (root+2) % 12},   # V
            {(root+9) % 12,  (root+0) % 12, (root+4) % 12},   # vi
            {(root+5) % 12,  (root+9) % 12, (root+0) % 12},   # IV
        ]
        chord_masks_base = []
        for chord_pcs in _progression:
            m = torch.zeros(vocab_size, device=device)
            for pitch, tid in note_on_by_pitch.items():
                if _pitch_register(pitch) == 1 and (pitch % 12) not in chord_pcs:
                    m[tid] = -1.0
            chord_masks_base.append(m)

    # #4 Repetition penalty soft
    repeat_window  = 16
    repeat_penalty = 1.0
    recent_tokens  = deque(maxlen=repeat_window)

    # Estado mutável — rastreia último step em que cada registro teve NOTE_ON
    min_tokens = 300
    state = {
        'last_pitch': None,
        'last_register': None,
        'steps': 0,
        'bar_just_seen': False,
        'bar_count': 0,
        'last_note_step': {0: 0, 1: 0, 2: 0},  # por registro
    }

    def constraint_fn(last_token_id: int) -> torch.Tensor:
        state['steps'] += 1
        recent_tokens.append(last_token_id)
        token = tokenizer.id_to_token.get(last_token_id, '')

        # Rastreia pitch e registro ativos
        if token.startswith('NOTE_ON_'):
            pitch = int(token.split('_')[2])
            state['last_pitch'] = pitch
            reg = _pitch_register(pitch)
            state['last_register'] = reg
            state['last_note_step'][reg] = state['steps']
            state['bar_just_seen'] = False

        if last_token_id == bar_id:
            state['bar_just_seen'] = True
            state['bar_count'] += 1

        # Token INSTRUMENT_0 emitido: libera VELOCITY (caso venha em sequência)
        if token == 'INSTRUMENT_0':
            return velocity_block + instr_block

        # #1 VELOCITY liberado imediatamente após NOTE_ON
        if last_token_id in note_on_ids:
            return instr_block  # mantém bloqueio de outros INSTRUMENT

        # Contexto geral: VELOCITY bloqueado + outros INSTRUMENT bloqueados + EOS
        eos_penalty = eos_block if state['steps'] < min_tokens else zero
        mask = velocity_block + instr_block + eos_penalty

        # #3 Voice leading: threshold conforme o registro do último pitch
        if state['last_pitch'] is not None:
            if state['last_register'] == 2:  # solo — graus conjuntos
                mask = mask + interval_masks_tight[state['last_pitch']]
            else:  # baixo ou base — saltos ok
                mask = mask + interval_masks_wide[state['last_pitch']]

        # #4 Repetition penalty soft em NOTE_ON e TIME_SHIFT recentes
        for tid in recent_tokens:
            if tid in note_on_ids or tid in time_shift_ids:
                mask[tid] -= repeat_penalty

        # #6 Chord backbone no registro Base (opt-in via --key)
        if chord_masks_base is not None:
            chord_idx = max(0, state['bar_count'] - 1) % 4
            mask = mask + chord_masks_base[chord_idx]

        # #7 Re-entry bias por REGISTRO silente
        for reg in (0, 1, 2):
            if reg == state['last_register']:
                continue
            silence = state['steps'] - state['last_note_step'][reg]
            if silence > silence_threshold:
                bonus = min(silence_max_bonus, (silence - silence_threshold) * silence_slope)
                mask = mask + register_masks[reg] * bonus

        return mask

    return constraint_fn


def generate_music(model: MultiInstrumentTransformer, tokenizer: MIDITokenizer,
                   config: dict, device: torch.device,
                   priming_sequence: list = None,
                   max_length: int = None,
                   temperature: float = None,
                   top_k: int = None,
                   top_p: float = None,
                   note_mask: torch.Tensor = None,
                   key_str: str = None,
                   temp_start: float = None,
                   temp_end: float = None,
                   temp_warmup: int = 150) -> list:
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
    vocab_constraint_fn = build_vocab_constraint(tokenizer, device, key_str=key_str)

    # Temperatura dinâmica: começa baixa (estabelece tema) e sobe até temp_end (variação)
    # Se temp_start/temp_end não fornecidos, usa temperatura fixa do parâmetro temperature
    temperature_fn = None
    if temp_start is not None and temp_end is not None:
        _ts, _te, _tw = temp_start, temp_end, temp_warmup
        def temperature_fn(step: int) -> float:
            if step >= _tw:
                return _te
            return _ts + (_te - _ts) * (step / _tw)

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
            temperature_fn=temperature_fn,
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
    parser.add_argument('--temp_start', type=float, default=None,
                       help='Temperatura inicial (baixa) para estabelecer tema. '
                            'Se fornecido junto com --temp_end, ativa temperatura dinâmica.')
    parser.add_argument('--temp_end', type=float, default=None,
                       help='Temperatura final (alta) para variação criativa.')
    parser.add_argument('--temp_warmup', type=int, default=150,
                       help='Número de tokens para rampa de temperatura (padrão: 150)')
    parser.add_argument('--tempo', type=int, default=100,
                       help='BPM de renderização do MIDI (padrão: 100). '
                            'Valores menores = peça mais relaxada.')
    parser.add_argument('--render_as_band', action='store_true',
                       help='Remapeia as 3 vozes do piano (baixo/base/solo) pra '
                            'timbres GM distintos (Bass + Nylon Guitar + Lead Guitar). '
                            'Requer modelo em modo piano solo (INSTRUMENT_0 forçado).')

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
            key_str=key_label,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            temp_warmup=args.temp_warmup,
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
            tempo=args.tempo,
            render_as_band=args.render_as_band,
        )
        
        if success:
            print(f"Música gerada salva em: {output_path}")
        else:
            print(f"Erro ao salvar música em: {output_path}")
    
    print("\nGeração concluída!")


if __name__ == '__main__':
    main()

