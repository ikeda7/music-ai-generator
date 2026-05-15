"""
Gera o conjunto de avaliação MOS — comparação cega Transformer vs Markov.

Produz 8 arquivos .mid (4 Transformer + 4 Markov) com nomes anonimizados
(sample_A a sample_H) numa ordem randomizada. Salva também um legend.json
com o mapeamento código → modelo, usado depois pra cruzar com as respostas
dos avaliadores.

Uso:
    python make_eval_set.py --output_dir ./eval_samples
    python make_eval_set.py --output_dir ./eval_samples --seed 42

Pré-requisitos:
- checkpoints/checkpoint_epoch_74.pt (modelo gold do Transformer)
- datasets/maestro/ (para o Markov aprender bigramas)
"""

import argparse
import json
import os
import random
import subprocess
import sys


# 4 amostras Transformer × 4 amostras Markov pra MOS robusto sem fatigar avaliadores
# Tons variados pra cobrir maior/menor e diferentes "humores"
TRANSFORMER_CONFIGS = [
    {'name': 'transformer_C',  'key': 'C',  'tempo': 100, 'temp': 0.9,  'drum_seed': 11},
    {'name': 'transformer_G',  'key': 'G',  'tempo': 95,  'temp': 0.95, 'drum_seed': 23},
    {'name': 'transformer_Am', 'key': 'Am', 'tempo': 90,  'temp': 0.95, 'drum_seed': 37},
    {'name': 'transformer_Em', 'key': 'Em', 'tempo': 95,  'temp': 0.9,  'drum_seed': 53},
]

MARKOV_CONFIGS = [
    {'name': 'markov_1', 'tempo': 100, 'seed': 1},
    {'name': 'markov_2', 'tempo': 95,  'seed': 2},
    {'name': 'markov_3', 'tempo': 90,  'seed': 3},
    {'name': 'markov_4', 'tempo': 95,  'seed': 4},
]


def run_transformer(checkpoint: str, output: str, config: dict) -> bool:
    """Roda generate.py com modo trio + solid_base + bateria (banda completa).
    drum_seed varia por amostra pra evitar baterias bit-a-bit idênticas no MOS."""
    cmd = [
        sys.executable, 'generate.py',
        '--checkpoint', checkpoint,
        '--output', output,
        '--key', config['key'],
        '--tempo', str(config['tempo']),
        '--temperature', str(config['temp']),
        '--top_k', '50',
        '--render_as_trio',
        '--solid_base',
        '--add_drums',
        '--drum_seed', str(config.get('drum_seed', 42)),
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERRO: {result.stderr[-500:]}")
        return False
    return True


def run_markov(dataset: str, output: str, config: dict, duration: float = 60.0) -> bool:
    """Roda markov_baseline.py (em evaluation/)."""
    cmd = [
        sys.executable, 'evaluation/markov_baseline.py',
        '--dataset', dataset,
        '--output', output,
        '--tempo', str(config['tempo']),
        '--duration', str(duration),
        '--seed', str(config['seed']),
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERRO: {result.stderr[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./eval_samples')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint_epoch_74.pt')
    parser.add_argument('--dataset', default='./datasets/maestro')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duração dos samples do Markov (s)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed pra randomizar a ordem dos códigos')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, 'work')
    os.makedirs(work_dir, exist_ok=True)

    # 1) Gera cada amostra com nome descritivo (privado, pra debug)
    all_samples = []
    print("=" * 60)
    print("Gerando amostras Transformer...")
    print("=" * 60)
    for cfg in TRANSFORMER_CONFIGS:
        out = os.path.join(work_dir, cfg['name'] + '.mid')
        if run_transformer(args.checkpoint, out, cfg):
            all_samples.append(('transformer', cfg, out))

    print("\n" + "=" * 60)
    print("Gerando amostras Markov...")
    print("=" * 60)
    for cfg in MARKOV_CONFIGS:
        out = os.path.join(work_dir, cfg['name'] + '.mid')
        if run_markov(args.dataset, out, cfg, duration=args.duration):
            all_samples.append(('markov', cfg, out))

    # 2) Randomiza ordem e renomeia pra códigos anônimos (sample_A, sample_B, ...)
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)

    legend = {}
    print("\n" + "=" * 60)
    print("Renomeando pra códigos anônimos...")
    print("=" * 60)
    for i, (model_type, cfg, src_path) in enumerate(all_samples):
        code = f"sample_{chr(ord('A') + i)}"
        dst_path = os.path.join(args.output_dir, code + '.mid')
        # Copia (não move — preserva /work pra debug)
        with open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
            f_out.write(f_in.read())
        legend[code] = {
            'model': model_type,
            'config': cfg,
            'source_file': os.path.basename(src_path),
        }
        print(f"  {code}.mid  ←  {model_type:11s} {cfg['name']}")

    # 3) Salva legenda (NÃO compartilhar com avaliadores — só pra análise final)
    legend_path = os.path.join(args.output_dir, 'legend.json')
    with open(legend_path, 'w', encoding='utf-8') as f:
        json.dump(legend, f, indent=2, ensure_ascii=False)

    print(f"\n{len(all_samples)} amostras geradas em {args.output_dir}/")
    print(f"Legenda salva em {legend_path} (CONFIDENCIAL — não compartilhar)")
    print(f"\nPróximo passo: converter .mid → .mp3 e subir no Google Forms.")
    print(f"Veja {args.output_dir}/MOS_GUIDE.md (se gerado).")


if __name__ == '__main__':
    main()
