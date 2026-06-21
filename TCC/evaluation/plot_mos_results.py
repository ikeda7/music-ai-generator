"""
Plota os resultados do MOS: médias Likert (Transformer vs Markov) e a
pergunta-Turing (% percebido como humano), reaproveitando o parsing do
analyze_mos.py (robusto a Forms sem 'Amostra X' no título).

Uso:
    python evaluation/plot_mos_results.py --responses respostas.csv \\
        --legend eval_samples/legend.json --output article/figura_mos.png
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_mos import (_classify_column, _classify_discrimination,
                         _is_profile_col, _profile_bucket, _positional_mapping)

COLOR_T = '#1565C0'
COLOR_M = '#C62828'
CRITS = [('naturalidade', 'Naturalidade'), ('coerencia', 'Coerência\nrítmica'),
         ('harmonia', 'Qualidade\nharmônica'), ('agradabilidade', 'Agradabilidade')]


def parse(responses, legend):
    scores = defaultdict(lambda: defaultdict(list))
    human = defaultdict(lambda: defaultdict(lambda: {'h': 0, 't': 0}))
    with open(responses, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, [])
        crit_idx, disc_idx, profile_idx = {}, {}, None
        for i, col in enumerate(header):
            code, crit = _classify_column(col)
            if code and crit and code in legend:
                crit_idx[i] = (code, crit); continue
            dcode = _classify_discrimination(col)
            if dcode and dcode in legend:
                disc_idx[i] = dcode; continue
            if profile_idx is None and _is_profile_col(col):
                profile_idx = i
        if not crit_idx:
            crit_idx, disc_idx = _positional_mapping(header, legend, profile_idx)
        for row in reader:
            if not any(c.strip() for c in row):
                continue
            for i, (code, crit) in crit_idx.items():
                v = row[i].strip() if i < len(row) else ''
                if not v:
                    continue
                try:
                    rating = float(v)
                except ValueError:
                    m = re.search(r'\d+', v)
                    if not m:
                        continue
                    rating = float(m.group())
                scores[legend[code]['model']][crit].append(rating)
            bucket = _profile_bucket(row[profile_idx]) if (profile_idx is not None
                     and profile_idx < len(row)) else 'todos'
            for i, code in disc_idx.items():
                a = row[i].strip().lower() if i < len(row) else ''
                if not a:
                    continue
                said_human = 'human' in a
                for b in ('todos', bucket):
                    human[legend[code]['model']][b]['t'] += 1
                    if said_human:
                        human[legend[code]['model']][b]['h'] += 1
    return scores, human


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--responses', required=True)
    ap.add_argument('--legend', required=True)
    ap.add_argument('--output', default='article/figura_mos.png')
    args = ap.parse_args()

    with open(args.legend, encoding='utf-8') as f:
        legend = json.load(f)
    scores, human = parse(args.responses, legend)

    n_resp = max((len(scores['transformer'].get(c, [])) for c, _ in CRITS), default=0)
    n_resp = n_resp // 4 if n_resp else 0  # 4 amostras por modelo

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Painel 1: médias Likert ---
    labels = [lbl for _, lbl in CRITS]
    t_means = [np.mean(scores['transformer'][k]) for k, _ in CRITS]
    m_means = [np.mean(scores['markov'][k]) for k, _ in CRITS]
    t_std = [np.std(scores['transformer'][k], ddof=1) for k, _ in CRITS]
    m_std = [np.std(scores['markov'][k], ddof=1) for k, _ in CRITS]
    x = np.arange(len(labels))
    w = 0.38
    ax1.bar(x - w/2, t_means, w, yerr=t_std, capsize=4, color=COLOR_T,
            edgecolor='black', label='Transformer')
    ax1.bar(x + w/2, m_means, w, yerr=m_std, capsize=4, color=COLOR_M,
            edgecolor='black', label='Markov')
    for xi, tv in zip(x, t_means):
        ax1.text(xi - w/2, tv + 0.08, f'{tv:.2f}', ha='center', fontsize=9, fontweight='bold')
    for xi, mv in zip(x, m_means):
        ax1.text(xi + w/2, mv + 0.08, f'{mv:.2f}', ha='center', fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Nota média (escala 1–5)', fontsize=11)
    ax1.set_ylim(0, 5.4)
    ax1.set_title('MOS — médias por critério (*** p<0,01)', fontsize=12)
    ax1.legend(fontsize=10); ax1.grid(True, axis='y', alpha=0.3, linestyle='--')

    # --- Painel 2: pergunta-Turing ---
    buckets = [('todos', 'Geral'), ('músico', 'Músicos'), ('não-músico', 'Não-músicos')]
    t_pct, m_pct, blabels = [], [], []
    for key, lbl in buckets:
        t = human['transformer'].get(key, {'h': 0, 't': 0})
        m = human['markov'].get(key, {'h': 0, 't': 0})
        if not t['t'] and not m['t']:
            continue
        t_pct.append(100 * t['h'] / t['t'] if t['t'] else 0)
        m_pct.append(100 * m['h'] / m['t'] if m['t'] else 0)
        blabels.append(lbl)
    xb = np.arange(len(blabels))
    ax2.bar(xb - w/2, t_pct, w, color=COLOR_T, edgecolor='black', label='Transformer')
    ax2.bar(xb + w/2, m_pct, w, color=COLOR_M, edgecolor='black', label='Markov')
    for xi, tv in zip(xb, t_pct):
        ax2.text(xi - w/2, tv + 1.5, f'{tv:.0f}%', ha='center', fontsize=9, fontweight='bold')
    for xi, mv in zip(xb, m_pct):
        ax2.text(xi + w/2, mv + 1.5, f'{mv:.0f}%', ha='center', fontsize=9)
    ax2.set_xticks(xb); ax2.set_xticklabels(blabels, fontsize=10)
    ax2.set_ylabel('% percebido como "humano"', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title('Pergunta-Turing — "humano ou computador?"', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    fig.suptitle(f'Avaliação Subjetiva (MOS) — N = {n_resp} avaliadores',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Figura salva em: {args.output} (N={n_resp})')


if __name__ == '__main__':
    main()
