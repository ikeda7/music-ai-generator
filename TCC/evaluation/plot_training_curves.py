"""
Plota a curva de loss do treinamento.

Duas fontes possíveis (combinadas se ambas disponíveis):
1. Checkpoints (.pt) — só captura val_loss em épocas onde salvou checkpoint
2. Log de treino (.txt/.log) — captura train_loss e val_loss por época,
   parseando linhas tipo:
       === Época 74/200 ===
         Train Loss: 2.5012  |  LR: 0.000063
         Val Loss:   2.3988

Uso:
    # Só checkpoints
    python plot_training_curves.py --checkpoint_dir ./checkpoints

    # Com log de treino (recomendado pra curva completa)
    python plot_training_curves.py --log training_log.txt

    # Combinado
    python plot_training_curves.py --checkpoint_dir ./checkpoints --log training_log.txt
"""

import argparse
import glob
import os
import re

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """
    Parseia arquivo de log de treino e extrai (epoch, train_loss, val_loss).
    Retorna duas listas: train_points e val_points, ambas [(epoch, loss)].
    """
    train_points = []
    val_points = []
    current_epoch = None

    re_epoch = re.compile(r'===\s*Época\s+(\d+)\s*/?\s*\d*\s*===')
    re_train = re.compile(r'Train\s*Loss:\s*([\d.]+)')
    re_val = re.compile(r'Val\s*Loss:\s*([\d.]+)')

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re_epoch.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            if current_epoch is None:
                continue
            m = re_train.search(line)
            if m:
                train_points.append((current_epoch, float(m.group(1))))
            m = re_val.search(line)
            if m:
                val_points.append((current_epoch, float(m.group(1))))

    train_points.sort(key=lambda x: x[0])
    val_points.sort(key=lambda x: x[0])
    return train_points, val_points


def collect_losses(checkpoint_dir: str):
    """Escaneia checkpoints e retorna lista [(epoch, loss)] ordenada."""
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nenhum checkpoint encontrado em {checkpoint_dir}")

    points = []
    for path in files:
        m = re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(path))
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            loss = float(ckpt.get('loss', None))
            points.append((epoch, loss))
            print(f"  ep {epoch:3d}: loss = {loss:.4f}")
        except Exception as e:
            print(f"  ep {epoch:3d}: erro lendo ({e})")

    points.sort(key=lambda x: x[0])
    return points


def plot_curve(train_points, val_points, ckpt_points, output_path: str, title: str = None):
    if not train_points and not val_points and not ckpt_points:
        print("Sem pontos para plotar.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if train_points:
        ep_t = [p[0] for p in train_points]
        ls_t = [p[1] for p in train_points]
        ax.plot(ep_t, ls_t, linewidth=1.5, color='#2E86AB',
                label='Train Loss', alpha=0.85)

    if val_points:
        ep_v = [p[0] for p in val_points]
        ls_v = [p[1] for p in val_points]
        ax.plot(ep_v, ls_v, linewidth=1.8, color='#C73E1D',
                label='Validation Loss', marker='.', markersize=4)

    # Checkpoints como pontos sólidos sobrepostos
    if ckpt_points:
        ep_c = [p[0] for p in ckpt_points]
        ls_c = [p[1] for p in ckpt_points]
        ax.scatter(ep_c, ls_c, s=80, color='#F18F01',
                   zorder=5, label='Checkpoints salvos',
                   edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Curva de Treinamento — MultiInstrumentTransformer', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')

    # Anota ep74 como GOLD
    points_all = val_points or ckpt_points or train_points
    for ep, loss in points_all:
        if ep == 74:
            ax.annotate('Checkpoint GOLD (ep74)',
                        xy=(ep, loss), xytext=(ep + 8, loss + 0.15),
                        fontsize=10, color='#1B5E20',
                        arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.2))
            break

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nFigura salva em: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default=None,
                        help='Diretório com checkpoint_epoch_*.pt')
    parser.add_argument('--log', default=None,
                        help='Arquivo de log de treino (train.py stdout salvo em .txt)')
    parser.add_argument('--output', default='figura_treino.png')
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    train_points = []
    val_points = []
    ckpt_points = []

    if args.log and os.path.isfile(args.log):
        print(f"Parseando log {args.log}...")
        train_points, val_points = parse_log(args.log)
        print(f"  Train losses encontradas: {len(train_points)}")
        print(f"  Val   losses encontradas: {len(val_points)}")

    if args.checkpoint_dir:
        print(f"\nLendo checkpoints em {args.checkpoint_dir}...")
        try:
            ckpt_points = collect_losses(args.checkpoint_dir)
        except FileNotFoundError as e:
            print(f"  {e}")

    if not (train_points or val_points or ckpt_points):
        print("\nERRO: forneça --log ou --checkpoint_dir com dados válidos.")
        return

    plot_curve(train_points, val_points, ckpt_points, args.output, args.title)


if __name__ == '__main__':
    main()
