"""
Script de treinamento do modelo Transformer Multi-Instrumental.
"""

import os
import json
import pickle
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_processor import MIDIProcessor, MIDITokenizer, prepare_sequences
from model import MultiInstrumentTransformer


class MusicDataset(Dataset):
    """Dataset para sequências musicais tokenizadas."""

    def __init__(self, input_sequences: np.ndarray, target_sequences: np.ndarray):
        self.inputs = torch.LongTensor(input_sequences)
        self.targets = torch.LongTensor(target_sequences)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_config(config_path: str = 'config.json') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _cache_path(data_paths: list, config: dict) -> str:
    """Gera nome único de cache baseado nos datasets e configurações de dados."""
    paths_sorted = ','.join(sorted(os.path.abspath(p) for p in data_paths))
    key = (
        f"{paths_sorted}"
        f"|seq={config['data']['seq_length']}"
        f"|res={config['data']['quantization_resolution']}"
    )
    return f".cache_tokens_{hashlib.md5(key.encode()).hexdigest()[:8]}.pkl"


def load_or_tokenize(data_paths: list, config: dict, rebuild: bool = False) -> dict:
    """
    Tokeniza os arquivos MIDI de múltiplos diretórios ou carrega de cache em disco.
    Evita re-tokenizar 1000+ arquivos a cada restart do treinamento.

    Retorna dict com: sequences (list de token lists), vocab_size (int), vocab (dict)
    """
    path = _cache_path(data_paths, config)

    if not rebuild and os.path.exists(path):
        print(f"Cache encontrado: {path}")
        with open(path, 'rb') as f:
            cached = pickle.load(f)
        print(f"  {len(cached['sequences'])} sequências carregadas do cache.")
        return cached

    proc = MIDIProcessor(config)
    tok = MIDITokenizer(config)

    all_events = []
    for data_path in data_paths:
        print(f"Carregando arquivos MIDI de '{data_path}'...")
        events = proc.load_midi_dataset(data_path)
        all_events.extend(events)

    if not all_events:
        raise RuntimeError(f"Nenhum arquivo MIDI encontrado em: {data_paths}")

    print(f"Tokenizando {len(all_events)} arquivos no total...")
    sequences = []
    for events in tqdm(all_events, desc="Tokenizando"):
        tokens = tok.encode_events(events)
        if len(tokens) > 64:
            sequences.append(tokens)

    print(f"{len(sequences)}/{len(all_events)} arquivos válidos (> 64 tokens)")

    result = {'sequences': sequences, 'vocab_size': tok.vocab_size, 'vocab': tok.vocab}
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    print(f"Cache salvo: {path}")
    return result


def build_loaders(sequences: list, vocab_size: int, config: dict):
    """
    Divide treino/validação 80/20 com shuffle aleatório (seed=42) e cria DataLoaders.
    Shuffle garante que validação não seja enviesada para o último dataset carregado.
    """
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(sequences))
    split = int(len(idx) * 0.8)
    train_seqs = [sequences[i] for i in idx[:split]]
    val_seqs = [sequences[i] for i in idx[split:]]

    seq_len = config['data']['seq_length']
    pad = config['vocab']['special_tokens']['PAD']

    print("Preparando janelas de treino...")
    train_inp, train_tgt = prepare_sequences(train_seqs, seq_len, vocab_size, pad)
    print("Preparando janelas de validação...")
    val_inp, val_tgt = prepare_sequences(val_seqs, seq_len, vocab_size, pad)

    bs = config['training']['batch_size']
    train_loader = DataLoader(
        MusicDataset(train_inp, train_tgt), batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        MusicDataset(val_inp, val_tgt), batch_size=bs, shuffle=False, num_workers=0
    )

    print(f"Treino:    {len(train_inp):,} sequências  ({len(train_loader):,} batches/época)")
    print(f"Validação: {len(val_inp):,} sequências  ({len(val_loader):,} batches/época)")
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion, device, scheduler, grad_clip):
    """Treina uma época. O scheduler é atualizado a cada batch (warmup correto)."""
    model.train()
    total_loss = 0.0
    for inp, tgt in tqdm(loader, desc="Treinando", leave=False):
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()  # por batch — linear warmup funciona corretamente assim
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inp, tgt in tqdm(loader, desc="Validando", leave=False):
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


def diagnostico_rapido(model, tokenizer, config, device, epoch, save_midi=False):
    """
    Gera amostra curta e mostra diversidade de notas geradas.
    Permite detectar mode collapse (gerar sempre a mesma nota) durante o treino.
    """
    model.eval()
    bos = config['vocab']['special_tokens']['BOS']
    eos = config['vocab']['special_tokens']['EOS']
    input_ids = torch.LongTensor([[bos]]).to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_length=256,
            temperature=0.8,
            top_k=20,
            top_p=0.95,
            eos_token_id=eos,
            context_size=config['data']['seq_length'],
        )

    tokens = generated[0].cpu().tolist()
    id_to_token = tokenizer.id_to_token

    pitches = [
        int(id_to_token[t].split('_')[2])
        for t in tokens
        if id_to_token.get(t, '').startswith('NOTE_ON_')
    ]

    unique = len(set(pitches))
    total = len(pitches)

    if unique == 0:
        simbolo = '❌'
        status = 'sem notas!'
    elif unique <= 2:
        simbolo = '⚠️ '
        status = f'colapsou em {unique} pitch(es)'
    elif unique <= 5:
        simbolo = '⚠️ '
        status = f'pouca variação ({unique} pitches)'
    else:
        simbolo = '✓ '
        status = f'OK ({unique} pitches únicos)'

    print(f"  {simbolo} Diagnóstico geração: {total} notas, {status}")

    if save_midi and total > 0:
        from music_utils import tokens_to_midi
        os.makedirs('samples', exist_ok=True)
        amostra_path = f"samples/amostra_epoch_{epoch + 1}.mid"
        tokens_to_midi(tokens, tokenizer, amostra_path, tempo=120)
        print(f"  Amostra MIDI salva: {amostra_path}")

    model.train()
    return unique


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, vocab, vocab_size):
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    path = os.path.join(config['training']['checkpoint_dir'], f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'vocab_size': vocab_size,
        'vocab': vocab,
    }, path)
    print(f"  Checkpoint salvo: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description='Treina modelo Transformer para geração de música')
    parser.add_argument('--data_path', type=str, nargs='+', required=True,
                        help='Um ou mais diretórios com arquivos MIDI de treino')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint .pt para continuar treinamento interrompido')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--rebuild_cache', action='store_true',
                        help='Ignora cache existente e re-tokeniza tudo do zero')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Dados ---
    print(f"Datasets: {args.data_path}")
    cached = load_or_tokenize(args.data_path, config, rebuild=args.rebuild_cache)
    sequences = cached['sequences']
    vocab_size = cached['vocab_size']
    vocab = cached['vocab']
    print(f"Vocabulário: {vocab_size} tokens")

    train_loader, val_loader = build_loaders(sequences, vocab_size, config)

    # Tokenizer para diagnóstico durante o treino
    tokenizer = MIDITokenizer(config)
    tokenizer.vocab = vocab
    tokenizer.id_to_token = {v: k for k, v in vocab.items()}
    tokenizer.vocab_size = vocab_size

    # --- Modelo ---
    mc = config['model']
    model = MultiInstrumentTransformer(
        vocab_size=vocab_size,
        d_model=mc['d_model'],
        nhead=mc['nhead'],
        num_layers=mc['num_layers'],
        dim_feedforward=mc['dim_feedforward'],
        dropout=mc['dropout'],
        max_seq_length=mc['max_seq_length'],
        num_instruments=config['data']['num_instruments'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {n_params:,} parâmetros  "
          f"(d_model={mc['d_model']}, layers={mc['num_layers']}, heads={mc['nhead']})")

    # --- Otimizador ---
    tc = config['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc['learning_rate'],
        weight_decay=0.01  # regularização leve para melhor generalização
    )
    criterion = nn.CrossEntropyLoss(ignore_index=config['vocab']['special_tokens']['PAD'])

    # Scheduler: linear warmup por N batches → cosine decay até 5% do LR inicial
    warmup = tc['warmup_steps']
    total_steps = len(train_loader) * tc['num_epochs']

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0

    if args.resume:
        print(f"Retomando de: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (ValueError, KeyError) as e:
            print(f"  Aviso: estado do otimizador incompatível ({e}).")
            print(f"  Iniciando otimizador do zero (pesos do modelo mantidos).")
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('loss', float('inf'))
        print(f"  Continuando da época {start_epoch}")

    # --- Loop de treinamento ---
    patience = tc.get('early_stop_patience', 20)
    grad_clip = tc.get('gradient_clip', 1.0)

    print(f"\nTreinando por até {tc['num_epochs']} épocas")
    print(f"Early stopping: {patience} validações sem melhora")
    print(f"Warmup: {warmup} batches  |  Total steps planejados: {total_steps:,}\n")

    for epoch in range(start_epoch, tc['num_epochs']):
        print(f"=== Época {epoch + 1}/{tc['num_epochs']} ===")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler, grad_clip
        )
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.4f}  |  LR: {lr_now:.6f}")

        # Validação periódica
        if (epoch + 1) % tc['eval_every'] == 0:
            val_loss = eval_epoch(model, val_loader, criterion, device)
            print(f"  Val Loss:   {val_loss:.4f}")

            # Diagnóstico de geração — detecta mode collapse
            salvar_amostra = (epoch + 1) % tc['save_every'] == 0
            diagnostico_rapido(model, tokenizer, config, device, epoch, save_midi=salvar_amostra)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, config, vocab, vocab_size
                )
                print(f"  Novo melhor modelo! (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Sem melhora: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\nEarly stopping ativado na época {epoch + 1}.")
                    break

        # Checkpoint periódico independente da validação
        if (epoch + 1) % tc['save_every'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, config, vocab, vocab_size
            )

    print("\nTreinamento concluído!")


if __name__ == '__main__':
    main()
