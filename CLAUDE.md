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
    ├── data_processor.py     ← tokenização REMI-like e dataset
    ├── model.py              ← arquitetura Transformer
    ├── train.py              ← loop de treinamento (com AMP)
    ├── generate.py           ← geração com filtros de escala/registro e modo banda
    ├── music_utils.py        ← conversão tokens → MIDI (com render_as_band)
    ├── diagnostico.py        ← diagnóstico de mode collapse
    └── show_midi.py          ← piano roll PNG
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

# Geração modo PIANO SOLO (output original do modelo)
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --key C --temperature 0.9 --top_k 40 --tempo 100

# Geração modo TRIO (recomendado) — 3 tracks de piano (solo/base/baixo)
# com filtros funcionais: bass quantizado ao BAR, base só em clusters, solo livre
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output trio.mid --key C --temperature 0.9 --top_k 40 --tempo 100 --render_as_trio

# Geração modo TRIO + SOLID BASE (banda híbrida ML+algorítmica)
# Bass marca tônica/quinta no compasso; base toca acordes I-V-vi-IV;
# solo vem do modelo. Funciona em maior (--key C) e menor (--key Am).
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output banda.mid --key Am --temperature 0.95 --top_k 50 --tempo 95 --render_as_trio --solid_base

# Geração modo BANDA (piano remapeado pra Bass GM + Nylon Guitar + Lead Guitar)
# AVISO: sintetizadores GM de guitarra/baixo são de baixa fidelidade
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output banda.mid --key C --temperature 0.9 --top_k 40 --tempo 100 --render_as_band

# Tonalidade automática via Krumhansl-Schmuckler
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --auto_key

# Múltiplas gerações
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt --output musica.mid --num_generations 3

# Diagnóstico de mode collapse
python diagnostico.py checkpoints/checkpoint_epoch_74.pt

# Piano roll PNG
python show_midi.py arquivo.mid
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
- `max_note_duration=1.5s` — nenhuma nota dura mais que isso
- NOTE_OFF ordenado antes de NOTE_ON no mesmo timestamp
- Modo **band** (`render_as_band=True`): override do slot do token baseado no registro de pitch
  - Bass: pitch ≤47 → canal 2, program 33 (Finger Bass)
  - Base: 48–65 → canal 1, program 24 (Nylon Guitar)
  - Solo: ≥66 → canal 0, program 30 (Distortion Guitar)
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
- **Epoch 99/109:** colapsaram para dyade fixo após platô longo em LR baixo — descartados
- **Decisão:** congelar ep74 como checkpoint final do TCC; trabalho agora é refinar geração e rodar avaliação

## Pendências para a Defesa Final

1. **Gerar amostras finais modo banda** (`--render_as_band`) com ep74 e validar com orientador
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
