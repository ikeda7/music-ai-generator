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
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Garante saída UTF-8 mesmo em console Windows cp1252 (evita crash no 'Δ' etc.)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


CRITERIA_KEYWORDS = {
    'naturalidade':   ['natural', 'naturalidade'],
    'coerencia':      ['coerencia', 'coerência', 'ritmo', 'rítmica'],
    'harmonia':       ['harmon', 'qualidade harmônica'],
    'agradabilidade': ['agradabilidade', 'ouviria'],
}


def _classify_column(col_name: str) -> Tuple[str, str]:
    """Retorna (sample_code, criterion) ou (None, None) se não casar.

    Ancora em 'Amostra'/'Sample' e captura a letra A-H em CAIXA ALTA. Não usar
    IGNORECASE na letra: senão o 'a' de 'Amostra' casaria como código A.
    """
    code_match = (re.search(r'(?i:amostra|sample)[\s_]*([A-H])\b', col_name)
                  or re.search(r'\b([A-H])\b', col_name))
    if not code_match:
        return None, None
    code = f"sample_{code_match.group(1).upper()}"

    lower = col_name.lower()
    for crit_key, keywords in CRITERIA_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return code, crit_key
    return None, None


# Pergunta-Turing ("composta por humano ou computador?") e perfil do avaliador
DISCRIMINATION_KEYWORDS = ['humano', 'computador', 'máquina', 'maquina', 'composta']
PROFILE_KEYWORDS = ['formação', 'formacao', 'prática musical', 'pratica musical',
                    'experiência musical', 'experiencia musical']


def _classify_discrimination(col_name: str):
    """Detecta a pergunta-Turing → retorna sample_code (ex.: 'sample_A') ou None."""
    lower = col_name.lower()
    if not any(k in lower for k in DISCRIMINATION_KEYWORDS):
        return None
    m = (re.search(r'(?:amostra|sample)[_\s]*([A-H])\b', col_name, re.IGNORECASE)
         or re.search(r'\b([A-H])\b', col_name))
    return f"sample_{m.group(1).upper()}" if m else None


def _is_profile_col(col_name: str) -> bool:
    lower = col_name.lower()
    return any(k in lower for k in PROFILE_KEYWORDS)


def _profile_bucket(value: str) -> str:
    """Classifica a resposta de perfil em 'músico' ou 'não-músico'."""
    v = (value or '').strip().lower()
    if not v:
        return 'desconhecido'
    if 'nenhum' in v or 'leigo' in v or v in ('não', 'nao'):
        return 'não-músico'
    return 'músico'


def _criterion_of(col_name: str):
    """Retorna a chave do critério pela presença de palavra-chave, ou None."""
    lower = col_name.lower()
    for crit_key, keywords in CRITERIA_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return crit_key
    return None


def _positional_mapping(header, legend, profile_idx):
    """Fallback p/ Forms sem 'Amostra X' no título (colunas com nomes repetidos).

    O Forms exporta perguntas com nomes iguais (ex.: 'Naturalidade' 8×), então o
    nome sozinho não identifica a amostra. Mapeamos por POSIÇÃO: as seções estão
    na ordem A, B, ... e a pergunta-Turing marca o fim de cada bloco; o i-ésimo
    bloco vira o i-ésimo código (A..H). Critérios são detectados pelo nome dentro
    do bloco (tolera reordenação interna).

    Retorna (crit_idx, disc_idx) com o ÍNDICE da coluna como chave.
    """
    letters = sorted({code.split('_')[-1] for code in legend})  # ['A'..'H']
    meta = ('carimbo', 'data/hora', 'e-mail', 'email', 'comentário', 'comentario')
    crit_idx, disc_idx = {}, {}
    block, bi = [], 0
    for i, col in enumerate(header):
        if i == profile_idx:
            continue
        lower = col.lower()
        if any(k in lower for k in meta):
            continue
        if any(k in lower for k in DISCRIMINATION_KEYWORDS):  # fim do bloco
            if bi < len(letters):
                code = f"sample_{letters[bi]}"
                if code in legend:
                    for ci, k in block:
                        crit_idx[ci] = (code, k)
                    disc_idx[i] = code
            bi += 1
            block = []
            continue
        crit = _criterion_of(col)
        if crit:
            block.append((i, crit))
    return crit_idx, disc_idx


def _welch_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Teste-t de Welch (variâncias diferentes). Retorna (t, p).

    Usa scipy.stats.ttest_ind(equal_var=False) para p exato quando o scipy
    está disponível. Sem scipy, cai numa aproximação conservadora por tabela
    (apenas para não quebrar o pipeline — o valor reportado na dissertação
    deve vir do scipy).
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0

    if _scipy_stats is not None:
        t, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)

    # --- Fallback sem scipy: aproximação grosseira de p ---
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se

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

    # utf-8-sig descarta BOM eventual do export do Forms/Sheets
    with open(args.responses, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, [])

        # Mapas por ÍNDICE de coluna (robusto a títulos duplicados do Forms)
        crit_idx = {}      # idx -> (code, criterion)
        disc_idx = {}      # idx -> code
        profile_idx = None
        for i, col in enumerate(header):
            code, crit = _classify_column(col)
            if code and crit and code in legend:
                crit_idx[i] = (code, crit)
                continue
            dcode = _classify_discrimination(col)
            if dcode and dcode in legend:
                disc_idx[i] = dcode
                continue
            if profile_idx is None and _is_profile_col(col):
                profile_idx = i

        if not crit_idx:
            # Forms sem 'Amostra X' nos títulos (colunas repetidas): mapeia por
            # posição/bloco, assumindo seções na ordem A..H.
            crit_idx, disc_idx = _positional_mapping(header, legend, profile_idx)
            if crit_idx:
                print("[aviso] Colunas sem 'Amostra X' no título — usando mapeamento")
                print("        POSICIONAL (seções assumidas na ordem A..H).\n")

        if not crit_idx:
            print("ERRO: nenhuma coluna válida identificada no CSV.")
            print("Verifique se há os 4 critérios + a pergunta-Turing por seção,")
            print("ou renomeie as perguntas para o padrão 'Amostra X — Critério'.")
            return

        # human_perception[model][bucket] = {'human': n, 'total': n}
        human_perception = defaultdict(
            lambda: defaultdict(lambda: {'human': 0, 'total': 0}))

        n_responses = 0
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            n_responses += 1
            for i, (code, crit) in crit_idx.items():
                val = row[i].strip() if i < len(row) else ''
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
                scores[legend[code]['model']][crit].append(rating)

            # Pergunta-Turing + segmentação por perfil do avaliador
            prof_val = row[profile_idx] if (profile_idx is not None and profile_idx < len(row)) else ''
            bucket = _profile_bucket(prof_val) if profile_idx is not None else 'todos'
            for i, code in disc_idx.items():
                ans = row[i].strip().lower() if i < len(row) else ''
                if not ans:
                    continue
                said_human = 'human' in ans  # "humano"
                for b in ('todos', bucket):
                    human_perception[legend[code]['model']][b]['total'] += 1
                    if said_human:
                        human_perception[legend[code]['model']][b]['human'] += 1

    print(f"Respostas processadas: {n_responses}")
    print(f"Colunas de critério mapeadas: {len(crit_idx)}\n")

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
    if _scipy_stats is not None:
        print("(p exato via scipy.stats.ttest_ind, teste-t de Welch)")
    else:
        print("(p APROXIMADO — scipy indisponível; instale scipy para valor exato)")

    # --- Pergunta-Turing: % que percebeu a peça como "composta por humano" ---
    has_disc = any(human_perception.get(m, {}).get('todos', {}).get('total', 0)
                   for m in ('transformer', 'markov'))
    if has_disc:
        print("\n" + "=" * 62)
        print("PERGUNTA-TURING — % que percebeu a peça como composta por HUMANO")
        print("=" * 62)
        for bucket in ('todos', 'músico', 'não-músico'):
            t = human_perception.get('transformer', {}).get(bucket, {'human': 0, 'total': 0})
            m = human_perception.get('markov', {}).get(bucket, {'human': 0, 'total': 0})
            if not t['total'] and not m['total']:
                continue
            tp = 100 * t['human'] / t['total'] if t['total'] else 0.0
            mp = 100 * m['human'] / m['total'] if m['total'] else 0.0
            label = {'todos': 'Geral', 'músico': 'Músicos',
                     'não-músico': 'Não-músicos'}[bucket]
            print(f"  {label:13s} Transformer: {tp:5.1f}% ({t['human']}/{t['total']})"
                  f"   Markov: {mp:5.1f}% ({m['human']}/{m['total']})")
        print("\n(Quanto MAIOR a % do Transformer vs Markov, mais o sistema 'convence' o")
        print(" ouvinte de que é música feita por humano — evidência direta do critério.)")

    if args.output and rows_out:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nResultados salvos em {args.output}")


if __name__ == '__main__':
    main()
