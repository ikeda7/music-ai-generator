# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto do Projeto

TCC de Lucas Vinícius de Carvalho Ikeda, orientado pelo Prof. Dr. Danillo Roberto Pereira — UNESP.
**Objetivo:** Sistema de IA que gera composições musicais em formato `.mid` com coerência melódica, harmônica e rítmica reais.

**Critério de aprovação:** Um humano ouvindo o `.mid` consegue identificar que é música (mesmo simples) e não confunde com ruído aleatório.

## Estrutura do Repositório

```
TCC/                          ← raiz do repositório git
├── PC1/                      ← fase 1 (concluída): anteprojeto + revisão bibliográfica
├── PC2/                      ← fase 2: protótipo notebook anterior + slides
└── TCC/                      ← código-fonte do sistema final
    ├── config.json           ← fonte da verdade para hiperparâmetros
    │
    ├── [CORE — pipeline ML]
    ├── data_processor.py     ← tokenização REMI-like e dataset
    ├── model.py              ← arquitetura Transformer
    ├── train.py              ← loop de treinamento (com AMP)
    ├── generate.py           ← geração com filtros de escala/registro e modos de render
    ├── music_utils.py        ← conversão tokens → MIDI (trio / band / solid_base / drums)
    ├── diagnostico.py        ← diagnóstico de mode collapse
    ├── show_midi.py          ← piano roll PNG colorido por papel
    │
    ├── tools/                ← utilitários auxiliares
    │   ├── download_datasets.py  ← downloader (MAESTRO/Groove/POP909)
    │   └── trim_eval_samples.py  ← padroniza duração dos samples
    │
    ├── evaluation/           ← baseline, métricas, MOS, plotagem
    │   ├── markov_baseline.py    ← cadeia de Markov ordem 1
    │   ├── metrics.py            ← métricas quantitativas → CSV
    │   ├── make_eval_set.py      ← gerador set MOS (Transformer + Markov)
    │   ├── analyze_mos.py        ← análise das respostas do Forms
    │   ├── plot_comparison_rolls.py
    │   ├── plot_metrics_comparison.py
    │   └── plot_training_curves.py
    │
    ├── article/              ← artigo + figuras
    │   ├── ARTIGO_TCC.md         ← artigo (markdown)
    │   ├── MOS_GUIDE.md          ← guia operacional do MOS
    │   ├── figura_metricas.png
    │   ├── figura_comparacao.png
    │   └── artigo.docx / artigo.pdf  ← versões compiladas
    │
    └── README.md             ← documentação geral (foco usuário externo)

notes.md  ← esqueleto da dissertação final (raiz do repo)
```

## Hardware

- GPU: NVIDIA RTX 4060 Ti (8 GB VRAM)
- Python 3.8.10 | PyTorch | CUDA 11.2
- OS: Windows

## Datasets (`TCC/datasets/` — não versionados)

| Dataset     | Arquivos | Uso                                       |
|-------------|----------|-------------------------------------------|
| MAESTRO     | 1276     | Piano solo, dados limpos (voz principal)  |
| POP909      | 2897     | Multi-instrumental (melodia, piano, baixo)|
| Groove MIDI | 1150     | Bateria humana                            |
| Lakh MIDI   | 15.754   | Requer filtragem antes de usar            |

**Filtros obrigatórios para Lakh:** mínimo 3 instrumentos, duração 120–600s, densidade 8–15 notas/s, excluir tonalidade C_major (artefato de scraping).

## Comandos

```bash
cd "h:/Meu Drive/TCC/TCC"

# Treinamento completo (3 datasets) — AMP habilitado automaticamente em GPU
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove

# Retomar do checkpoint GOLD (ep74)
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove --resume checkpoints/checkpoint_epoch_74.pt

# Se mudar a função de perda entre runs, use --reset_best_loss
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove --resume checkpoints/checkpoint_epoch_74.pt --reset_best_loss

# ============ GERAÇÃO ============
# MODO RECOMENDADO PRO TCC: TRIO + SOLID BASE (3 vozes de piano, sem bateria)
# Bass marca tônica/quinta no compasso; base toca acordes I-V-vi-IV (ou i-VI-iv-V em
# menor); solo (registro >65) vem do modelo. Funciona em maior (--key C) e menor (--key Am).
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output banda.mid --key Am --temperature 0.95 --top_k 50 --tempo 95 --render_as_trio --solid_base

# Modo TRIO sem solid_base — 3 tracks de piano (solo/base/baixo) com filtros
# funcionais (bass quantizado ao BAR, base só em clusters, solo monofônico).
# Tudo vem do modelo, sem fundação sintética.
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output trio.mid --key C --temperature 0.9 --top_k 40 --tempo 100 --render_as_trio

# Modo PIANO SOLO single-track (output cru do modelo, sem split por registro)
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --key C --temperature 0.9 --top_k 40 --tempo 100

# Modo BANDA (piano remapeado pra Bass GM + Nylon Guitar + Lead Guitar)
# AVISO: sintetizadores GM de guitarra/baixo são de baixa fidelidade — uso para demo só
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output banda.mid --key C --temperature 0.9 --top_k 40 --tempo 100 --render_as_band

# Tonalidade automática via Krumhansl-Schmuckler
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --auto_key

# Múltiplas gerações
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --num_generations 3

# Diagnóstico de mode collapse
python diagnostico.py checkpoints/checkpoint_epoch_74.pt

# Piano roll PNG colorido por papel funcional (Solo/Base/Baixo/Bateria)
python show_midi.py arquivo.mid

# ============ AVALIAÇÃO (rodar do diretório TCC/) ============
# Baseline Markov pra comparação
python evaluation/markov_baseline.py --dataset ./datasets/maestro --output markov.mid

# Métricas quantitativas (CSV)
python evaluation/metrics.py --input ./eval_samples --output metricas.csv

# Gera set MOS (4 Transformer + 4 Markov anonimizados)
python evaluation/make_eval_set.py --output_dir ./eval_samples

# Padroniza duração dos samples
python tools/trim_eval_samples.py --input_dir ./eval_samples --duration 60

# Análise das respostas do Forms
python evaluation/analyze_mos.py --responses respostas.csv --legend eval_samples/legend.json

# Figuras pro artigo
python evaluation/plot_comparison_rolls.py --inputs eval_samples/sample_A.mid eval_samples/sample_B.mid --labels Transformer Markov --output article/figura_comparacao.png
python evaluation/plot_metrics_comparison.py --metrics eval_samples/metricas.csv --legend eval_samples/legend.json --output article/figura_metricas.png
```

## Arquitetura do Sistema

### Pipeline de dados
```
Arquivo .mid → MIDIProcessor.load_midi()      → List[MusicalEvent]
             → _quantize_and_add_time_shifts() → TIME_SHIFT armazena steps (não segundos)
             → MIDITokenizer.encode_events()   → List[int]  (com BAR/BEAT intercalados)
             → prepare_sequences()             → janelas de 512 tokens, overlap 50%
             → MusicDataset / DataLoader       → batches
```

### Vocabulário (~349 tokens)
- `PAD=0, BOS=1, EOS=2, MASK=3`
- `INSTRUMENT_0..4` → Piano, Melodia, Baixo, Bateria, Harmonia
- `NOTE_ON_21..108` + `NOTE_OFF_21..108` (88 pitches × 2 = 176 tokens)
- `VELOCITY_0..31` (32 bins)
- `TIME_SHIFT_1..128` (steps; `ticks_per_step = 1/quantization_resolution = 0.0625s`)
- `BAR`, `BEAT_2`, `BEAT_3`, `BEAT_4` (Bar-Relative Encoding — estrutura métrica)

O vocabulário é salvo dentro do checkpoint — `generate.py` restaura o vocab do `.pt`.

### Modelo (`model.py`)
`MultiInstrumentTransformer` — **decoder-only Transformer (GPT-style)**:
- Usa `nn.TransformerEncoder` do PyTorch com causal mask
- d_model=256, nhead=4, num_layers=4, dim_feedforward=1024 (~3,2M params, cabe em 8 GB VRAM)
- Weight tying: `output_projection.weight` compartilha tensor com `token_embedding.weight`
- `generate()` aceita `note_mask` (filtro de escala), `vocab_constraint_fn` (restrições) e `temperature_fn` (temperatura dinâmica)

### Estratégia de Geração — Piano Solo com 3 Vozes

A abordagem final **força INSTRUMENT_0 (Piano)** durante a geração. As 3 vozes (solo/base/baixo) **emergem naturalmente** dos registros de pitch aprendidos do MAESTRO:

| Voz | Registro MIDI | Descrição |
|-----|---------------|-----------|
| Solo/Melodia | 66–108 (F#4–C8) | Mão direita — frases melódicas |
| Base/Harmonia | 48–65 (C3–F4) | Acordes de acompanhamento |
| Baixo | 21–47 (A0–B2) | Mão esquerda — notas-raiz e oitavas |

No modo `--render_as_band`, essas 3 vozes são remapeadas para timbres distintos em canais GM separados (Bass + Nylon Guitar + Lead Guitar), simulando uma banda a partir do modelo treinado em piano solo.

### Modos de render — escolha rápida

| Modo | Flag | Quando usar |
|------|------|-------------|
| Piano solo single-track | (default) | Output cru do modelo, debug, baseline |
| Trio piano | `--render_as_trio` | 3 tracks de piano (solo/base/baixo) com filtros funcionais |
| Trio piano + base sintética | `--render_as_trio --solid_base --key X` | **Modo canônico do TCC** — 3 vozes de piano (solo do modelo + base/baixo algorítmicos) |
| Banda GM | `--render_as_band` | Demo com timbres GM distintos (qualidade limitada de sintetizadores) |
| + Bateria (exploratório) | `--add_drums` | **Não-canônico** — desincroniza com saída do modelo em peças >30s. Mantido como feature, retirado do TCC. |

**Detalhes do `--solid_base`** (só ativa se `--key` for fornecido):
- Progressão: I-V-vi-IV em maior; i-VI-iv-V em menor (V harmônico com 3ª maior)
- Bass: tônica no tempo 1 + quinta no tempo 3 de cada BAR (root-fifth movement)
- Base: chord stamp em tempo 1 sempre forte; tempo 3 com "breath bar" a cada 4 compassos
- Solo: monofonia + gap mínimo 0.25s entre NOTE_ONs; notas do registro Base são promovidas (+12 semitons) para enriquecer a melodia
- Velocity boost 1.6× (mín 80, máx 120) — compensa dynamics suaves do MAESTRO no GM Piano

### Restrições de vocabulário na geração (`generate.py`)

`build_vocab_constraint()` implementa via `vocab_constraint_fn`:

1. **VELOCITY só após NOTE_ON** (gramatical) — previne loop de VELOCITY
2. **Força INSTRUMENT_0** (piano único) — bloqueia INSTRUMENT_1..4
3. **Voice leading por registro** — solo penaliza saltos > 7 semitons; baixo/base > 12 semitons
4. **Repetition penalty soft** (-1.0) em NOTE_ON/TIME_SHIFT nas últimas 16 posições
5. **Bloqueio de EOS** nos primeiros 300 tokens
6. **Chord backbone I-V-vi-IV** — opt-in via `--key`; aplica ao registro Base (48–65)
7. **Re-entry bias por REGISTRO silente** — bônus positivo em NOTE_ON do registro ausente há >60 tokens

Removidas (modelo aprende do dataset): cycling forçado, penalidade de steps ímpares, rhythm stability, bass anchor por slot, penalidade de TIME_SHIFT repetido.

### Conversão para MIDI (`music_utils.py`)
- Apenas NOTE_ON vai para `instrument_tracks`; NOTE_OFF gerados pelo hanging-note fix
- `max_note_duration=1.0s` — nenhuma nota dura mais que isso (cap anti-hanging)
- NOTE_OFF ordenado antes de NOTE_ON no mesmo timestamp
- Modo **trio** (`render_as_trio=True`): mesma separação por registro do modo band, mas mantém Grand Piano em todas as tracks (qualidade GM superior)
- Modo **band** (`render_as_band=True`): override do slot do token baseado no registro de pitch
  - Bass: pitch ≤47 → canal 2, program 33 (Finger Bass)
  - Base: 48–65 → canal 1, program 24 (Nylon Guitar)
  - Solo: ≥66 → canal 0, program 30 (Distortion Guitar)
- Filtros funcionais (`_apply_band_filters`): bass quantizado ao BAR (±0.15s, groove 15%); base só em clusters (≥2 notas em 0.5s); bass e solo monofônicos (cada NOTE_ON fecha a nota anterior do mesmo registro)
- `_inject_solid_foundation`: substitui bass+base pelo I-V-vi-IV sintético (ativado por `--solid_base`)
- `--tempo` no `generate.py` controla o BPM de renderização (default 100)

### Treinamento (`train.py`)
- AdamW, lr=0.0001, weight_decay=0.01, gradient clip=1.0
- LR: warmup linear 2000 steps → cosine decay até 5% do LR inicial
- `CrossEntropyLoss(label_smoothing=0.1)` — evita overconfidence em tokens "seguros" (TIME_SHIFT)
- **AMP (`torch.cuda.amp`):** autocast no forward + GradScaler no backward; reduz VRAM ~40%, acelera ~2×
- Data augmentation: transposição em [0, +4, +8] semitons (bateria excluída, slot 3)
- Checkpoint salvo quando val_loss melhora OU a cada `save_every` épocas (sem duplicata)
- Flag `--reset_best_loss` para runs que mudam a função de perda
- Diagnóstico automático a cada 5 épocas detecta mode collapse

## Status do Treinamento

- **Checkpoint GOLD:** `checkpoint_epoch_74.pt` — validado visual e auditivamente com re-entry bias funcional
- **Checkpoints úteis disponíveis:** `ep49`, `ep74` (GOLD), `ep79` — todos os outros foram descartados
- **Epoch 89/94/99/104/109:** colapsaram para dyade fixo após platô longo em LR baixo — deletados
- **Decisão:** congelar ep74 como checkpoint final do TCC; trabalho agora é refinar geração e rodar avaliação

## Pendências para a Defesa Final

1. **Gerar amostras finais modo trio + solid_base** (`--render_as_trio --solid_base --key X`) com ep74 em maior e menor, e validar com orientador
2. **Pipeline de bateria** — pendente: treino separado mínimo com Groove MIDI OU algoritmo determinístico
3. **Avaliação com usuários** — 5–10 pessoas, escala MOS (Mean Opinion Score), avaliação cega
4. **Baseline Markov chain** — bigrama de pitches do MAESTRO, comparação quantitativa
5. **Métricas quantitativas** — diversidade de pitches, densidade, similaridade com dataset
6. **Atualizar slides PC2** — mostra config antiga (6 camadas, 8 heads, MAESTRO+Groove)
7. **Dissertação final** — ainda não escrita; esqueleto em `notes.md`
8. **Confirmar template** com Prof. Danillo: SBC double-column vs. ABNT UNESP

## Restrições Inegociáveis

- Saída sempre em `.mid` compatível com MuseScore/FL Studio
- PyTorch — não voltar para TF/Keras
- Separação de módulos: `data_processor` / `model` / `train` / `generate`
- `random_seed = 42` em todo lugar
- Comentários em **português**
- Sem modelos externos ou APIs de geração musical
