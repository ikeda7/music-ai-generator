"""
Diagnóstico completo da geração de música.
Analisa tokens gerados, eventos musicais e o arquivo MIDI resultante.
"""

import json
import sys
import torch
import mido
import numpy as np
from collections import Counter

from data_processor import MIDITokenizer
from model import MultiInstrumentTransformer
from music_utils import tokens_to_midi


def load_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model(checkpoint_path, config, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = MIDITokenizer(config)
    if 'vocab' in checkpoint:
        tokenizer.vocab = checkpoint['vocab']
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)

    mc = config['model']
    model = MultiInstrumentTransformer(
        vocab_size=checkpoint.get('vocab_size', tokenizer.vocab_size),
        d_model=mc['d_model'],
        nhead=mc['nhead'],
        num_layers=mc['num_layers'],
        dim_feedforward=mc['dim_feedforward'],
        dropout=mc['dropout'],
        max_seq_length=mc['max_seq_length'],
        num_instruments=config['data']['num_instruments'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, checkpoint


def analisar_tokens(tokens, tokenizer):
    print("\n" + "="*60)
    print("ANÁLISE DOS TOKENS GERADOS")
    print("="*60)

    id_to_token = tokenizer.id_to_token
    nomes = [id_to_token.get(t, f'UNK({t})') for t in tokens]

    # Categorias
    categorias = Counter()
    for nome in nomes:
        if nome in ('BOS', 'EOS', 'PAD', 'MASK'):
            categorias['especiais'] += 1
        elif nome.startswith('TIME_SHIFT_'):
            categorias['TIME_SHIFT'] += 1
        elif nome.startswith('NOTE_ON_'):
            categorias['NOTE_ON'] += 1
        elif nome.startswith('NOTE_OFF_'):
            categorias['NOTE_OFF'] += 1
        elif nome.startswith('VELOCITY_'):
            categorias['VELOCITY'] += 1
        elif nome.startswith('INSTRUMENT_'):
            categorias['INSTRUMENT'] += 1
        else:
            categorias['outros'] += 1

    total = len(tokens)
    print(f"\nTotal de tokens: {total}")
    print("\nDistribuição:")
    for cat, count in categorias.most_common():
        pct = count / total * 100
        barra = '█' * int(pct / 2)
        print(f"  {cat:<15} {count:>5}  ({pct:5.1f}%)  {barra}")

    # Notas geradas
    pitches = [int(n.split('_')[2]) for n in nomes if n.startswith('NOTE_ON_')]
    if pitches:
        print(f"\nNotas ON geradas: {len(pitches)}")
        print(f"  Pitch mínimo : {min(pitches)} (nota MIDI)")
        print(f"  Pitch máximo : {max(pitches)} (nota MIDI)")
        print(f"  Pitch médio  : {np.mean(pitches):.1f}")
        # 10 notas mais frequentes
        mais_freq = Counter(pitches).most_common(10)
        print(f"  Pitches mais frequentes: {mais_freq}")
    else:
        print("\n⚠️  NENHUMA nota NOTE_ON gerada! O modelo está gerando sequências sem notas.")

    # Primeiros 30 tokens
    print(f"\nPrimeiros 30 tokens:")
    print("  " + " | ".join(nomes[:30]))

    # Verificar repetição excessiva
    if len(nomes) > 10:
        repeticao = sum(1 for i in range(1, len(nomes)) if nomes[i] == nomes[i-1])
        pct_rep = repeticao / len(nomes) * 100
        if pct_rep > 20:
            print(f"\n⚠️  Alta repetição consecutiva: {pct_rep:.1f}% dos tokens são iguais ao anterior")

    return categorias, pitches


def analisar_midi(caminho):
    print("\n" + "="*60)
    print(f"ANÁLISE DO ARQUIVO MIDI: {caminho}")
    print("="*60)
    try:
        mid = mido.MidiFile(caminho)
        print(f"\nTipo MIDI     : {mid.type}")
        print(f"Ticks/beat    : {mid.ticks_per_beat}")
        print(f"Num. tracks   : {len(mid.tracks)}")
        print(f"Duração aprox : {mid.length:.1f} segundos")

        total_notes = 0
        for i, track in enumerate(mid.tracks):
            msgs = [m for m in track if hasattr(m, 'type')]
            note_ons = [m for m in msgs if m.type == 'note_on' and m.velocity > 0]
            note_offs = [m for m in msgs if m.type == 'note_off' or
                         (m.type == 'note_on' and m.velocity == 0)]
            total_notes += len(note_ons)
            print(f"\n  Track {i}: {len(msgs)} mensagens | "
                  f"{len(note_ons)} note_on | {len(note_offs)} note_off")

        print(f"\nTotal de notas no MIDI: {total_notes}")
        if total_notes == 0:
            print("⚠️  MIDI está VAZIO — nenhuma nota foi inserida!")
        elif total_notes < 20:
            print("⚠️  MIDI tem muito poucas notas para uma música coerente.")
        else:
            print("✓  MIDI tem notas suficientes.")

    except Exception as e:
        print(f"Erro ao abrir MIDI: {e}")


def diagnostico_completo(checkpoint_path, midi_path=None, temperature=0.7, top_k=10):
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"\nCarregando checkpoint: {checkpoint_path}")
    model, tokenizer, checkpoint = load_model(checkpoint_path, config, device)
    print(f"Época do checkpoint: {checkpoint.get('epoch', '?')}")
    print(f"Loss salva: {checkpoint.get('loss', '?'):.4f}" if isinstance(checkpoint.get('loss'), float) else "")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Gera com temperatura baixa para diagnóstico
    bos = config['vocab']['special_tokens']['BOS']
    eos = config['vocab']['special_tokens']['EOS']
    input_ids = torch.LongTensor([[bos]]).to(device)

    print(f"\nGerando 512 tokens (temperature={temperature}, top_k={top_k})...")
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=512,
            temperature=temperature,
            top_k=top_k,
            top_p=0.95,
            eos_token_id=eos,
            context_size=config['data']['seq_length'],
        )
    tokens = generated[0].cpu().tolist()

    categorias, pitches = analisar_tokens(tokens, tokenizer)

    # Salva MIDI de diagnóstico
    output_diag = 'diagnostico_saida.mid'
    print(f"\nConvertendo para MIDI: {output_diag}")
    tokens_to_midi(tokens, tokenizer, output_diag, tempo=120)
    analisar_midi(output_diag)

    # Analisa MIDI existente se fornecido
    if midi_path:
        analisar_midi(midi_path)

    # Diagnóstico final
    print("\n" + "="*60)
    print("DIAGNÓSTICO FINAL")
    print("="*60)
    note_ons = categorias.get('NOTE_ON', 0)
    time_shifts = categorias.get('TIME_SHIFT', 0)
    total = sum(categorias.values())

    if note_ons == 0:
        print("❌ PROBLEMA CRÍTICO: modelo não gera notas.")
        print("   Causa provável: treinamento insuficiente ou bug no pipeline.")
    elif note_ons / total < 0.05:
        print(f"⚠️  PROBLEMA: pouquíssimas notas ({note_ons} em {total} tokens = {note_ons/total*100:.1f}%)")
        print("   Causa provável: modelo aprendeu a gerar silêncio/espera.")
    else:
        print(f"✓  {note_ons} notas geradas em {total} tokens ({note_ons/total*100:.1f}%)")
        print("   Pipeline OK. Problema provavelmente é qualidade musical.")
        print("   Sugestão: tente temperatura entre 0.5 e 0.75.")

    if time_shifts / total > 0.6:
        print(f"⚠️  Muitos TIME_SHIFT ({time_shifts/total*100:.1f}%) — música longa mas esparsa.")

    print(f"\nArquivo MIDI de diagnóstico salvo em: {output_diag}")
    print("Abra no MuseScore ou importe num player MIDI para ouvir.")


if __name__ == '__main__':
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/checkpoint_epoch_89.pt'
    midi_existente = sys.argv[2] if len(sys.argv) > 2 else 'teste1.mid'
    diagnostico_completo(checkpoint, midi_existente)
