# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto do Projeto

TCC de Lucas Vinícius de Carvalho Ikeda, orientado pelo Prof. Dr. Danillo Roberto Pereira — UNOESTE.
**Objetivo:** Sistema de IA que gera composições musicais em formato `.mid` com coerência melódica, harmônica e rítmica reais.

**Critério de aprovação:** Um humano ouvindo o `.mid` consegue identificar que é música (mesmo simples) e não confunde com ruído aleatório.

## Estrutura do Repositório

```
TCC/                          ← raiz do repositório
├── PC1/                      ← fase 1 do TCC (concluída)
│   ├── anteprojeto/          ← proposta entregue em 29/08/2024
│   ├── artigos/              ← PDFs das referências bibliográficas
│   └── revisao_bibliografica/ ← revisão entregue em 13/11/2024
├── PC2/                      ← fase 2 do TCC (em andamento)
│   ├── prototipo/            ← prototipo.ipynb (versão notebook anterior)
│   ├── slide.pdf / slide.pptx ← apresentação PC2 (desatualizada — ver pendências)
│   └── *.mp3                 ← amostras geradas nas versões anteriores
└── TCC/                      ← código-fonte do sistema final
    ├── config.json           ← fonte da verdade para hiperparâmetros
    ├── data_processor.py     ← tokenização REMI-like e dataset
    ├── model.py              ← arquitetura Transformer
    ├── train.py              ← loop de treinamento
    ├── generate.py           ← geração com filtros de escala/instrumento
    ├── music_utils.py        ← conversão tokens → MIDI
    ├── diagnostico.py        ← diagnóstico de mode collapse
    └── show_midi.py          ← piano roll PNG
```

## Fases do TCC

### PC1 — Fundamentação (ago–nov 2024) ✓
- Anteprojeto (`PC1/anteprojeto/anteprojeto.pdf`) e Revisão Bibliográfica (`PC1/revisao_bibliografica/revisao_bibliografica.pdf`) entregues.
- Artigos-chave em `PC1/artigos/`: "Attention Is All You Need" (Vaswani 2017), fundamentos LSTM.

### PC2 — Protótipo e Defesa Final (2025) ← em andamento
- Protótipo anterior: `PC2/prototipo/prototipo.ipynb` (versão simplificada em notebook)
- Sistema final: pasta `TCC/` — pipeline completo treinado em MAESTRO + POP909 + Groove MIDI

## Hardware

- GPU: NVIDIA RTX 4060 Ti (8 GB VRAM)
- Python 3.8.10 | PyTorch | CUDA 11.2
- OS: Windows

## Datasets (`TCC/datasets/` — não versionados)

| Dataset     | Arquivos | Uso                                      |
|-------------|----------|------------------------------------------|
| MAESTRO     | 1276     | Piano solo, dados limpos                 |
| POP909      | 2897     | Multi-instrumental (melodia, piano, baixo)|
| Groove MIDI | 1150     | Bateria humana                           |
| Lakh MIDI   | 15.754   | Requer filtragem antes de usar           |

**Filtros obrigatórios para Lakh:** mínimo 3 instrumentos, duração 120–600s, densidade 8–15 notas/s, excluir tonalidade C_major (artefato de scraping).

## Comandos

```bash
cd TCC/

# Instalação
pip install -r requirements.txt

# Treinamento completo (3 datasets)
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove

# Retomar de checkpoint
python train.py --data_path ./datasets/maestro ./datasets/pop909 ./datasets/groove --resume checkpoints/checkpoint_epoch_N.pt

# Geração com filtro de escala
python generate.py --checkpoint checkpoints/checkpoint_epoch_N.pt --output musica.mid --key C --temperature 0.9 --top_k 40

# Geração com detecção automática de tonalidade
python generate.py --checkpoint checkpoints/checkpoint_epoch_N.pt --output musica.mid --auto_key

# Diagnóstico (mode collapse, análise de tokens)
python diagnostico.py checkpoints/checkpoint_epoch_N.pt

# Piano roll PNG
python show_midi.py arquivo.mid
```

## Arquitetura do Sistema (`TCC/`)

### Pipeline de dados
```
Arquivo .mid → MIDIProcessor.load_midi()      → List[MusicalEvent]
             → _quantize_and_add_time_shifts() → TIME_SHIFT em steps (não segundos)
             → MIDITokenizer.encode_events()   → List[int]
             → prepare_sequences()             → janelas de 512 tokens, overlap 50%
             → MusicDataset / DataLoader       → batches
```

### Vocabulário (~345 tokens)
- `PAD=0, BOS=1, EOS=2, MASK=3`
- `INSTRUMENT_0..4` → Piano, Melodia, Baixo, Bateria, Harmonia
- `NOTE_ON_21..108` + `NOTE_OFF_21..108` (88 pitches × 2)
- `VELOCITY_0..31` (32 bins)
- `TIME_SHIFT_1..128` (steps de 1/16 de beat — `ticks_per_step = 0.0625s`)

O vocabulário é salvo dentro do checkpoint — `generate.py` restaura o vocab do `.pt`.

### Modelo (`model.py`)
`MultiInstrumentTransformer` — encoder-only com causal masking:
- d_model=256, nhead=4, num_layers=4, dim_feedforward=1024 (calibrado para 8 GB VRAM)
- Weight tying: `output_projection.weight` compartilha tensor com `token_embedding.weight`
- `generate()` aceita `note_mask` (filtro de escala) e `vocab_constraint_fn` (restrições de vocabulário)

### Restrições de vocabulário na geração (`generate.py`)
`build_vocab_constraint()` implementa duas regras via `vocab_constraint_fn`:
1. **VELOCITY só após NOTE_ON** — previne loop infinito de tokens VELOCITY
2. **Cycling de instrumentos** — força troca de instrumento a cada 16 notas geradas (ciclando INSTRUMENT_0..4)

### Conversão para MIDI (`music_utils.py`)
- Apenas NOTE_ON vai para `instrument_tracks`; todos os NOTE_OFF são gerados pelo hanging-note fix
- `max_note_duration=1.5s` — nenhuma nota dura mais que isso
- NOTE_OFF é ordenado antes de NOTE_ON no mesmo timestamp (evita notas contínuas longas)
- Drums no canal 9 (GM), sem `program_change`

### Filtro de escala (`generate.py`)
- `build_note_mask(key_str, tokenizer, device)` — aplica -inf em NOTE_ON fora da escala
- `detect_key()` — Krumhansl-Schmuckler para detecção automática de tonalidade
- Uso: `--key C` ou `--auto_key`

### Treinamento (`train.py`)
- AdamW, lr=0.0001, warmup linear 2000 steps → cosine decay até 5% do LR
- Diagnóstico automático a cada 5 épocas (detecta mode collapse)
- `num_epochs=200` em `config.json` — treino atual vai de epoch 100 a 199

## Pendências para a Defesa Final

1. **Resultado sonoro aceitável** — treino MAESTRO+POP909+Groove em andamento (epoch 100→199)
2. **Avaliação com usuários** — 5–10 pessoas ouvem 3 amostras, respondem formulário (criatividade, coerência, agradabilidade). Prometido no anteprojeto e no slide PC2.
3. **Métricas quantitativas** — diversidade de pitches, densidade de notas, similaridade com dataset
4. **Atualizar slides PC2** — `PC2/slide.pdf` mostra config antiga (6 camadas, 8 heads, MAESTRO+Groove). Config real: 4 camadas, 4 heads, MAESTRO+POP909+Groove
5. **Dissertação final** — documento não encontrado. Precisa ser escrito.

## Restrições Inegociáveis

- Saída sempre em `.mid` compatível com MuseScore/FL Studio
- PyTorch (projeto foi reescrito de TF para PyTorch — não voltar para TF/Keras)
- Separação de módulos: `data_processor` / `model` / `train` / `generate`
- `random_seed = 42` em todo lugar
- Comentários em **português**
- Sem modelos externos ou APIs de geração musical
