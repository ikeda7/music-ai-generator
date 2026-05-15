"""
Análise das respostas do Google Forms — calcula MOS por critério por modelo
e roda teste-t de Welch pra significância estatística.

Espera CSV exportado do Forms com colunas tipo:
    "Carimbo de data/hora", "Sample A — Naturalidade", "Sample A — Coerência", ...

A coluna de cada pergunta deve seguir o padrão "Sample X — Critério" onde X é
o código (A-H) e Critério é um de: Naturalidade, Coerência, Harmônica, Agradabilidade.

Uso:
    python analyze_mos.py --responses respostas.csv --legend legend.json
    python analyze_mos.py --responses respostas.csv --legend legend.json \\
        --output resultados.csv
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple


CRITERIA_KEYWORDS = {
    'naturalidade':   ['natural', 'naturalidade'],
    'coerencia':      ['coerencia', 'coerência', 'ritmo', 'rítmica'],
    'harmonia':       ['harmon', 'qualidade harmônica'],
    'agradabilidade': ['agradabilidade', 'ouviria'],
}


def _classify_column(col_name: str) -> Tuple[str, str]:
    """Retorna (sample_code, criterion) ou (None, None) se não casar."""
    code_match = re.search(r'(?:sample[_\s]*)?([A-H])\b', col_name, re.IGNORECASE)
    if not code_match:
        return None, None
    code = f"sample_{code_match.group(1).upper()}"

    lower = col_name.lower()
    for crit_key, keywords in CRITERIA_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return code, crit_key
    return None, None


def _welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Teste-t de Welch (variâncias diferentes). Retorna (t, p-aprox).
    p é aproximado via tabela conservadora — pra rigor estatístico exato,
    usar scipy.stats.ttest_ind.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se

    # Aproximação grosseira de p: |t| > 2 ~= p < 0.05; |t| > 2.6 ~= p < 0.01
    abs_t = abs(t)
    if abs_t > 3.0:
        p = 0.001
    elif abs_t > 2.6:
        p = 0.01
    elif abs_t > 2.0:
        p = 0.05
    elif abs_t > 1.7:
        p = 0.1
    else:
        p = 0.2  # não significativo
    return t, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses', required=True, help='CSV exportado do Forms')
    parser.add_argument('--legend', required=True, help='legend.json gerado por make_eval_set.py')
    parser.add_argument('--output', default=None, help='CSV de saída (opcional)')
    args = parser.parse_args()

    with open(args.legend, 'r', encoding='utf-8') as f:
        legend = json.load(f)

    # scores[model_type][criterion] = lista de ratings
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with open(args.responses, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Mapeia colunas → (code, criterion)
        col_map = {}
        for col in reader.fieldnames or []:
            code, crit = _classify_column(col)
            if code and crit and code in legend:
                col_map[col] = (code, crit)

        if not col_map:
            print("ERRO: nenhuma coluna válida identificada no CSV.")
            print("Verifique se as perguntas seguem o padrão 'Sample X — Critério'")
            return

        n_responses = 0
        for row in reader:
            n_responses += 1
            for col, (code, crit) in col_map.items():
                val = row.get(col, '').strip()
                if not val:
                    continue
                try:
                    rating = float(val)
                except ValueError:
                    # Pode vir como "5 - Excelente" — extrai primeiro número
                    m = re.search(r'\d+', val)
                    if not m:
                        continue
                    rating = float(m.group())
                model_type = legend[code]['model']
                scores[model_type][crit].append(rating)

    print(f"Respostas processadas: {n_responses}")
    print(f"Colunas mapeadas: {len(col_map)}\n")

    # Tabela comparativa
    criteria = ['naturalidade', 'coerencia', 'harmonia', 'agradabilidade']
    header = f"{'Critério':18s} {'Transformer':>20s} {'Markov':>20s} {'Δ':>8s} {'p':>8s}"
    print(header)
    print("=" * len(header))

    rows_out = []
    for crit in criteria:
        t_scores = scores.get('transformer', {}).get(crit, [])
        m_scores = scores.get('markov', {}).get(crit, [])
        if not t_scores or not m_scores:
            print(f"{crit:18s} {'sem dados':>20s}")
            continue
        mean_t = sum(t_scores) / len(t_scores)
        mean_m = sum(m_scores) / len(m_scores)
        std_t = math.sqrt(sum((x - mean_t) ** 2 for x in t_scores) / max(len(t_scores)-1, 1))
        std_m = math.sqrt(sum((x - mean_m) ** 2 for x in m_scores) / max(len(m_scores)-1, 1))
        _, p = _welch_t_test(t_scores, m_scores)
        delta = mean_t - mean_m
        sig = '***' if p <= 0.01 else '**' if p <= 0.05 else '*' if p <= 0.1 else ''
        print(f"{crit:18s} {mean_t:8.2f} ± {std_t:5.2f}  {mean_m:8.2f} ± {std_m:5.2f}  "
              f"{delta:+7.2f}  {p:>6.3f} {sig}")
        rows_out.append({
            'criterion': crit,
            'transformer_mean': mean_t, 'transformer_std': std_t, 'transformer_n': len(t_scores),
            'markov_mean': mean_m, 'markov_std': std_m, 'markov_n': len(m_scores),
            'delta': delta, 'p_approx': p,
        })

    print("\nSignificância: * p≤0.10  ** p≤0.05  *** p≤0.01")
    print("(p é aproximação; pra valor exato use scipy.stats.ttest_ind)")

    if args.output and rows_out:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResultados salvos em {args.output}")


if __name__ == '__main__':
    main()
