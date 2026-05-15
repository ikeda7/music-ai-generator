# Sistema de Geração de Música com IA — Transformer + Constrained Decoding

Sistema de geração de música multi-instrumental simbólica (formato MIDI) baseado em arquitetura Transformer combinada com técnicas de *Constrained Decoding* e pós-processamento algorítmico inspirado em teoria musical. TCC desenvolvido por Lucas Vinícius de Carvalho Ikeda sob orientação do Prof. Dr. Danillo Roberto Pereira — UNESP.

## Início Rápido

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar datasets (MAESTRO, Groove, POP909)
python tools/download_datasets.py --all

# 3. Treinar (ou usar o checkpoint pronto, ep74)
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove

# 4. Gerar música no modo canônico (trio de piano: solo + base + baixo)
python generate.py \
  --checkpoint checkpoints/checkpoint_epoch_74.pt \
  --output minha_musica.mid \
  --key C --tempo 100 --temperature 0.9 --top_k 40 \
  --render_as_trio --solid_base

# 5. Visualizar piano roll colorido por papel funcional
python show_midi.py minha_musica.mid
```

## Estrutura do Projeto

```
TCC/
├── data_processor.py       # Tokenização REMI-like, dataset, cache
├── model.py                # MultiInstrumentTransformer (decoder-only)
├── train.py                # Loop de treinamento (AMP, AdamW, label smoothing)
├── generate.py             # Geração com Constrained Decoding
├── music_utils.py          # tokens → MIDI, modos trio/band/solid_base/drums
├── diagnostico.py          # Detecção de mode collapse
├── show_midi.py            # Piano roll PNG colorido (Solo/Base/Baixo/Bateria)
├── config.json             # Hiperparâmetros
│
├── tools/                  # Utilitários auxiliares
│   ├── download_datasets.py
│   └── trim_eval_samples.py
│
├── evaluation/             # Baseline, métricas, MOS, plotagem
│   ├── markov_baseline.py
│   ├── metrics.py
│   ├── make_eval_set.py
│   ├── analyze_mos.py
│   ├── plot_comparison_rolls.py
│   ├── plot_metrics_comparison.py
│   └── plot_training_curves.py
│
└── article/                # Artigo + figuras
    ├── ARTIGO_TCC.md
    ├── MOS_GUIDE.md
    ├── figura_metricas.png
    └── figura_comparacao.png
```

Datasets, checkpoints e samples não são versionados (estão no `.gitignore`).

## Modos de Geração

| Modo | Flag | Quando usar |
|------|------|-------------|
| Piano solo single-track | (default) | Output cru do modelo, debug |
| Trio piano | `--render_as_trio` | 3 tracks de piano (solo/base/baixo) com filtros funcionais |
| **Trio + base sintética** | `--render_as_trio --solid_base --key X` | **Modo canônico** — 3 vozes de piano (solo do modelo + base/baixo algorítmicos) |
| Banda GM | `--render_as_band` | Demo com timbres GM distintos (qualidade limitada) |
| + Bateria (exploratório) | `--add_drums` | Não-canônico — desincroniza com modelo em peças longas. Mantido como flag opcional. |

## Instalação

### Requisitos

- Python 3.8+
- PyTorch + CUDA (recomendado para treino — funciona em CPU para geração)
- ~20 GB de espaço para datasets

### Dependências

```bash
pip install -r requirements.txt
```

### Datasets

```bash
python tools/download_datasets.py --all
# Ou específicos:
python tools/download_datasets.py --maestro
python tools/download_datasets.py --pop909
python tools/download_datasets.py --groove
```

| Dataset | Tamanho | Uso |
|---------|---------|-----|
| MAESTRO v3.0.0 | ~200h piano | Voz principal (modelo aprende) |
| POP909 | 2.897 arquivos | Estrutura multi-track |
| Groove MIDI | 1.150 padrões | Bateria humana (reservado p/ trabalho futuro) |

## Treinamento

```bash
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove
```

**Parâmetros:**
- `--data_path`: um ou mais diretórios MIDI (obrigatório)
- `--resume`: checkpoint `.pt` para continuar
- `--reset_best_loss`: ignorar `best_val_loss` do checkpoint (use ao mudar função de perda)
- `--device`: `auto`, `cpu` ou `cuda`

O treino salva checkpoints em `checkpoints/checkpoint_epoch_X.pt` quando a loss de validação melhora. Diagnóstico automático a cada 5 épocas detecta mode collapse.

**Decisão deste TCC:** o modelo foi congelado em **ep74** (val_loss 2,3988). Checkpoints posteriores (89, 99, 109) apresentaram tendência ao colapso e foram descartados.

## Geração — exemplos completos

```bash
# Trio de piano em menor (modo recomendado para apresentação)
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt \
  --output banda_Am.mid --key Am --tempo 95 --temperature 0.95 --top_k 50 \
  --render_as_trio --solid_base

# Apenas trio sem fundação algorítmica (modelo puro)
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt \
  --output trio_puro.mid --key C --tempo 100 --render_as_trio

# Múltiplas amostras
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt \
  --output musica.mid --num_generations 5 --auto_key
```

**Parâmetros principais:**
- `--checkpoint`: caminho do `.pt` (obrigatório)
- `--output`: arquivo `.mid` de saída
- `--key`: tonalidade (`C`, `G`, `Am`, etc.) — habilita chord backbone e solid_base
- `--auto_key`: detecção automática via Krumhansl-Schmuckler
- `--temperature` / `--top_k` / `--top_p`: parâmetros de sampling
- `--tempo`: BPM de renderização (default 100)
- `--num_generations`: gerar múltiplas peças

## Avaliação

```bash
# Métricas quantitativas (CSV)
python evaluation/metrics.py --input ./eval_samples --output metricas.csv

# Gera conjunto de avaliação MOS (4 Transformer + 4 Markov, ordem randomizada)
python evaluation/make_eval_set.py --output_dir ./eval_samples

# Padroniza duração para comparação justa
python tools/trim_eval_samples.py --input_dir ./eval_samples --duration 60

# Baseline trivial pra comparação acadêmica
python evaluation/markov_baseline.py --dataset ./datasets/maestro --output markov.mid

# Análise das respostas do Google Forms
python evaluation/analyze_mos.py --responses respostas.csv --legend eval_samples/legend.json
```

Veja `article/MOS_GUIDE.md` para o guia operacional completo do estudo subjetivo.

## Arquitetura Técnica

### Modelo (`MultiInstrumentTransformer`)

- Decoder-only Transformer (causal mask) sobre `nn.TransformerEncoder`
- d_model=256, nhead=4, num_layers=4, dim_feedforward=1024
- ~3,2M parâmetros — cabe em 8 GB VRAM
- Weight tying entre embedding e projeção de saída

### Tokenização

REMI-like + Bar-Relative Encoding, vocabulário ~349 tokens:
- `INSTRUMENT_0..4`, `NOTE_ON_21..108`, `NOTE_OFF_21..108`
- `VELOCITY_0..31`, `TIME_SHIFT_1..128` (steps de 0,0625s)
- `BAR`, `BEAT_2`, `BEAT_3`, `BEAT_4`

### Pipeline de Geração — Constrained Decoding

A geração não é amostragem pura — o Transformer produz logits e uma função de restrição (`vocab_constraint_fn`) os modifica antes da amostragem:

1. **Restrição gramatical** — VELOCITY só após NOTE_ON
2. **Slot único** — força INSTRUMENT_0; vozes emergem dos registros
3. **Voice leading** — penaliza saltos grandes por registro
4. **Repetition penalty soft** — desconto –1,0 em tokens recentes
5. **Bloqueio de EOS** — primeiros 300 tokens
6. **Chord backbone** — bônus para pitches consonantes (opt-in via `--key`)
7. **Re-entry bias dinâmico** — bônus positivo crescente em registros silentes (> 60 tokens)

Após a amostragem, as notas são separadas em tracks por registro de pitch e cada uma recebe filtros funcionais (monofonia no solo/baixo, quantização a 1/8 de tempo, cap de duração 1,2s).

## Artigo Acadêmico

O artigo do TCC está em `article/ARTIGO_TCC.md` (estrutura SBC). Inclui:
- Fundamentação teórica (Transformer, REMI, MIDI, sampling)
- Trabalhos relacionados (Music Transformer, MuseGAN, Pop MT, Markov)
- Metodologia completa
- Métricas quantitativas (Tabela 1, Figura 1)
- Análise qualitativa (Figura 2)
- Conclusão e trabalhos futuros

Para compilar em PDF/DOCX via pandoc:
```bash
pandoc article/ARTIGO_TCC.md -o article/artigo.pdf
```

## Troubleshooting

**CUDA out of memory:** reduza `batch_size` em `config.json` ou rode com `--device cpu`.

**"Nenhum arquivo MIDI encontrado":** verifique se o `--data_path` aponta para diretório com `.mid`/`.midi`.

**Mode collapse durante o treino:** o diagnóstico automático detecta e avisa. Se ocorrer, retome de um checkpoint anterior e considere reduzir LR ou aumentar `label_smoothing`.

**Geração sem som de banda:** verifique se está usando `--render_as_trio` e/ou `--render_as_band`. Modo default é piano solo single-track.

## Licença

Trabalho acadêmico de TCC. Datasets utilizados mantêm suas licenças originais (MAESTRO, POP909, Groove são CC).

## Referências

- Vaswani et al. (2017). Attention is all you need. *NeurIPS*.
- Huang et al. (2018). Music Transformer. *arXiv:1809.04281*.
- Huang & Yang (2020). Pop Music Transformer. *ACM MM*.
- Hawthorne et al. (2019). MAESTRO dataset. *ICLR*.
- Holtzman et al. (2020). Neural text degeneration. *ICLR*.
