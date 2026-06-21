"""
Plota gráfico de barras comparando métricas individuais das 8 amostras MOS.

Não mostra só médias — exibe cada sample com sua cor (Transformer azul,
Markov vermelho), permitindo ver dispersão e outliers. Útil pro artigo
pra demonstrar consistência intra-modelo.

Uso:
    python plot_metrics_comparison.py --metrics eval_samples/metricas.csv \\
        --legend eval_samples/legend.json --output figura_metricas.png
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = [
    ('pitch_diversity', 'Diversidade de Pitches'),
    ('note_density',    'Densidade (notas/s)'),
    ('pitch_range',     'Range de Pitches (semitons)'),
    ('ioi_std',         'IOI std (regularidade rítmica)'),
    ('kl_vs_dataset',   'KL vs MAESTRO'),
]

COLOR_TRANSFORMER = '#1565C0'
COLOR_MARKOV = '#C62828'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', required=True, help='CSV gerado por metrics.py')
    parser.add_argument('--legend', required=True, help='legend.json gerado por make_eval_set.py')
    parser.add_argument('--output', default='figura_metricas.png')
    args = parser.parse_args()

    with open(args.legend, 'r', encoding='utf-8') as f:
        legend = json.load(f)

    # Carrega métricas e cruza com legenda
    rows = []
    with open(args.metrics, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row['file'].replace('.mid', '')
            model_type = legend.get(sample_id, {}).get('model', '?')
            rows.append({'sample': sample_id.replace('sample_', ''),
                         'model': model_type, **row})

    # Ordena: Transformer primeiro, depois Markov, mantendo letra
    rows.sort(key=lambda r: (0 if r['model'] == 'transformer' else 1, r['sample']))

    n_metrics = len(METRIC_KEYS)
    # Layout 3x2: 2 painéis por linha => cada painel ocupa ~metade da largura
    # da coluna do artigo, ficando bem maior/legível. O 6o painel fica oculto.
    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.3 * nrows))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(METRIC_KEYS):
        ax = axes[idx]
        values = []
        colors = []
        labels = []
        for r in rows:
            try:
                v = float(r[key])
            except (ValueError, TypeError):
                v = 0.0
            values.append(v)
            colors.append(COLOR_TRANSFORMER if r['model'] == 'transformer' else COLOR_MARKOV)
            labels.append(r['sample'])

        x = np.arange(len(values))
        ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel('Amostra', fontsize=12)
        ax.set_ylabel('Valor da métrica', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Linha de média por modelo
        t_vals = [v for v, c in zip(values, colors) if c == COLOR_TRANSFORMER]
        m_vals = [v for v, c in zip(values, colors) if c == COLOR_MARKOV]
        if t_vals:
            ax.axhline(y=np.mean(t_vals), color=COLOR_TRANSFORMER,
                       linestyle=':', linewidth=1.2, alpha=0.7)
        if m_vals:
            ax.axhline(y=np.mean(m_vals), color=COLOR_MARKOV,
                       linestyle=':', linewidth=1.2, alpha=0.7)

    # Oculta painéis extras não utilizados (5 métricas em grade 2x3)
    for extra in range(n_metrics, len(axes)):
        axes[extra].axis('off')

    # Legenda compartilhada no topo
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_TRANSFORMER, edgecolor='black', label='Transformer'),
        Patch(facecolor=COLOR_MARKOV, edgecolor='black', label='Markov'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle('Métricas Quantitativas por Amostra — Transformer vs Markov',
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Figura salva em: {args.output}")


if __name__ == '__main__':
    main()
