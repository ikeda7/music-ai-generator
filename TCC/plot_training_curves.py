"""
Plota a curva de loss do treinamento a partir dos checkpoints salvos.

Os checkpoints armazenam apenas (epoch, loss) — não há histórico completo
gravado. Este script escaneia checkpoint_epoch_*.pt no diretório, extrai
os pares e plota. Para curva mais densa, é preciso ter mais checkpoints
salvos durante o treino.

Uso:
    python plot_training_curves.py --checkpoint_dir ./checkpoints
    python plot_training_curves.py --checkpoint_dir ./checkpoints --output curva_treino.png
"""

import argparse
import glob
import os
import re

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def plot_curve(points, output_path: str, title: str = None):
    if not points:
        print("Sem pontos para plotar.")
        return

    epochs = [p[0] for p in points]
    losses = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, losses, marker='o', linewidth=1.5, markersize=6,
            color='#2E86AB', label='Validation Loss')
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Curva de Treinamento — MultiInstrumentTransformer', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Anota o checkpoint GOLD (ep74) se presente
    for ep, loss in points:
        if ep == 74:
            ax.annotate('Checkpoint GOLD',
                        xy=(ep, loss), xytext=(ep + 5, loss + 0.05),
                        fontsize=10, color='#C73E1D',
                        arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=1.2))
            break

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nFigura salva em: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--output', default='figura_treino.png')
    parser.add_argument('--title', default=None)
    args = parser.parse_args()

    print(f"Lendo checkpoints em {args.checkpoint_dir}...\n")
    points = collect_losses(args.checkpoint_dir)
    if len(points) < 2:
        print("\nAVISO: menos de 2 checkpoints encontrados — curva não fará sentido.")
        print("Considere re-rodar o treino salvando mais checkpoints intermediários.")
    plot_curve(points, args.output, args.title)


if __name__ == '__main__':
    main()
