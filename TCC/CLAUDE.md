# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto do Projeto

TCC de Lucas Vinícius de Carvalho Ikeda, orientado pelo Prof. Dr. Danillo Roberto Pereira.
**Objetivo:** Sistema de IA que gera composições musicais em formato `.mid` com coerência melódica, harmônica e rítmica reais — não apenas música sintaticamente correta.

**Critério de aprovação:** Um humano ouvindo o `.mid` consegue identificar que é música (mesmo simples) e não confunde com ruído aleatório.

## Fases do TCC

### PC1 — Fundamentação (ago–nov 2024) ✓
Localização: `../PC1/`

Entregáveis concluídos:
- **Anteprojeto** (29/08/2024): proposta de 4–5 páginas com metodologia e referências iniciais (`anteprojeto/anteprojeto.pdf`)
- **Revisão Bibliográfica** (13/11/2024): revisão de 20+ referências cobrindo RNNs, LSTMs, Transformers e geração musical (`revisao_bibliografica/revisao_bibliografica.pdf`)

Artigos de referência chave em `../PC1/artigos/`:
- `1706.03762v7.pdf` — "Attention Is All You Need" (Vaswani et al., 2017) — base da arquitetura
- `GERACAO_DE_MUSICA_COM_APRENDIZADO_DE_MAQUINA.pdf` — contexto em português
- `lstm.pdf` — fundamentos de LSTM (baseline comparativo)

### PC2 — Protótipo e Apresentação (2025) ← em andamento
Localização: `../PC2/`

Entregáveis:
- **Protótipo funcional**: este repositório (`TCC/`) — pipeline completo de geração
- **Amostras geradas**: `../PC2/composition_01_classical_0.6.mp3`, `generated_music_02.mp3`
- **Apresentação**: `../PC2/slide.pdf`

O código em `TCC/` É o protótipo do PC2. Qualquer melhoria aqui impacta diretamente a nota final.

## Hardware

- GPU: NVIDIA RTX 4060 Ti (8 GB VRAM)
- CPU: AMD Ryzen 7
- Python 3.8.10 | PyTorch | CUDA 11.2 / cuDNN 8
- OS: Windows

## Datasets (localmente disponíveis em `./datasets/`)

| Dataset | Arquivos | Horas | Uso |
|---------|----------|-------|-----|
| MAESTRO | 839 | 74.4h | Piano solo, dados limpos |
| POP909 | 909 | — | Multi-instrumental (melodia, piano, baixo) |
| Groove MIDI | 407 | 12.6h | Bateria humana |
| Lakh MIDI | 15.754 | 1011h | Requer filtragem agressiva antes de usar |

**Filtros obrigatórios para Lakh:**
- Mínimo 3 instrumentos simultâneos
- Duração entre 120s e 600s
- Densidade entre 8–15 notas/s
- Excluir tonalidade C_major em 100% dos casos (artefato de scraping)

## Comandos

```bash
# Instalação
pip install -r requirements.txt

# Treinamento (multi-dataset — comando padrão atual)
python train.py --data_path ./datasets/maestro ./datasets/pop909

# Retomar treinamento interrompido
python train.py --data_path ./datasets/maestro ./datasets/pop909 --resume checkpoints/checkpoint_epoch_N.pt

# Forçar re-tokenização (apagar cache e reprocessar)
python train.py --data_path ./datasets/maestro ./datasets/pop909 --rebuild_cache

# Geração
python generate.py --checkpoint checkpoints/checkpoint_epoch_N.pt --output musica.mid
python generate.py --checkpoint checkpoints/checkpoint_epoch_N.pt --output musica.mid --temperature 0.8 --top_k 20
python generate.py --checkpoint checkpoints/checkpoint_epoch_N.pt --output musica.mid --priming referencia.mid

# Diagnóstico (detecta mode collapse, analisa tokens e MIDI)
python diagnostico.py checkpoints/checkpoint_epoch_N.pt

# Visualizar MIDI como piano roll (salva PNG)
python show_midi.py arquivo.mid
```

## Arquitetura

### Pipeline de dados
```
Arquivo .mid → MIDIProcessor.load_midi()       → List[MusicalEvent]
             → _quantize_and_add_time_shifts()  → TIME_SHIFT entre eventos (em steps, não segundos)
             → MIDITokenizer.encode_events()    → List[int]  (sequência plana)
             → prepare_sequences()              → janelas de 512 tokens com overlap 50%
             → MusicDataset / DataLoader        → batches para o modelo
```

### Vocabulário (~345 tokens)
Construído em `_build_vocab()` ([data_processor.py:264](data_processor.py#L264)):
- `PAD=0, BOS=1, EOS=2, MASK=3` (especiais — MASK não é usado, LM causal)
- `INSTRUMENT_0..4` → Piano, Melodia, Baixo, Bateria, Harmonia
- `NOTE_ON_21..108` + `NOTE_OFF_21..108` (88 pitches × 2 = 176 tokens)
- `VELOCITY_0..31` (32 bins quantizados)
- `TIME_SHIFT_1..128` (steps de tempo — `event.time` armazena contagem de steps, não segundos)

O vocabulário é **salvo dentro do checkpoint**, então `generate.py` restaura o vocab do próprio `.pt`.

### Modelo (`model.py`)
`MultiInstrumentTransformer` — encoder-only com causal masking:
- Causal mask gerada automaticamente em `forward()` via `_generate_square_subsequent_mask()`
- `PositionalEncoding`: seno/cosseno, formato `(seq_len, batch, d_model)`
- **Weight tying**: `output_projection.weight` compartilha tensor com `token_embedding.weight`
- `generate()`: amostragem autoregressiva, trunca contexto a 512 tokens (janela de treino)

**Configuração atual** (`config.json`): d_model=256, nhead=4, num_layers=4, dim_feedforward=1024 — calibrada para RTX 4060 Ti (8 GB VRAM).

### Treinamento (`train.py`)
- `AdamW`, lr=0.0001, weight_decay=0.01, gradient clip=1.0
- LR: warmup linear (2000 steps) → cosine decay até 5% do LR inicial
- `CrossEntropyLoss(ignore_index=PAD)`
- Split 80/20 treino/validação (seed=42)
- Checkpoint salvo a cada 10 épocas + sempre que val_loss melhorar
- **Diagnóstico automático** a cada 5 épocas: mostra diversidade de pitches gerados (detecta mode collapse)
- **Amostras MIDI** salvas em `samples/amostra_epoch_N.mid` a cada 10 épocas

### Conversão para MIDI (`music_utils.py`)
Canais General MIDI por instrumento: Piano→0, Melodia→1, Baixo→2, Bateria→**9**, Harmonia→3.
Bateria no canal 9 não recebe `program_change` (convenção GM).

### Diagnóstico de mode collapse
O modelo pode colapsar para prever sempre a mesma nota. Sintoma: `diagnostico.py` mostra
`pitches únicos = 1`. Causa: dataset pouco diverso. Solução: adicionar POP909 ao treino.

## Status

**Treino atual:** MAESTRO + POP909 combinados (`python train.py --data_path ./datasets/maestro ./datasets/pop909`).

O treino anterior (100 épocas só com MAESTRO) colapsou para nota única — checkpoints descartados.
Cache `.cache_tokens_*.pkl` é gerado automaticamente na primeira execução de cada combinação de datasets.

## Restrições Inegociáveis

- Saída sempre em `.mid` compatível com MuseScore/FL Studio
- Framework: PyTorch (não TF/Keras — projeto foi reescrito de TF para PyTorch)
- Separação de módulos: `data_processor` / `model` / `train` / `generate`
- `random_seed = 42` em todo lugar
- Comentários em **português**
- Sem modelos externos ou APIs de geração musical

## Pendências para a Defesa Final

O diploma depende destes itens. Em ordem de prioridade:

1. **Resultado sonoro aceitável** — treino MAESTRO+POP909 em andamento. Critério: `diagnostico.py` mostra >5 pitches únicos e o MIDI gerado é reconhecível como música.
2. **Filtro de escala na geração** — a implementar em `generate.py` após treino convergir. Garante coerência harmônica via mascaramento de logits fora da escala detectada. ~40 linhas.
3. **Avaliação com usuários** — prometida no anteprojeto E no slide PC2. Mínimo: 5–10 pessoas ouvem 3 amostras e respondem formulário (criatividade, coerência, agradabilidade). Simples mas obrigatório.
4. **Métricas quantitativas** — diversidade de pitches, densidade de notas, similaridade com dataset. Base já existe em `diagnostico.py`.
5. **Atualizar slides PC2** — `../PC2/slide.pdf` desatualizado: mostra "6 camadas, 8 heads" mas config real é 4/4; menciona MAESTRO+Groove mas treino atual é MAESTRO+POP909.
6. **Dissertação final** — documento não encontrado nos arquivos. Precisa ser escrito.

## Como Trabalhar Neste Projeto

- `config.json` é a fonte da verdade para hiperparâmetros
- Antes de editar qualquer módulo, leia-o completo
- Para testar mudanças rapidamente: `"max_files": 50` no bloco `"data"` do `config.json`
- Alterações grandes (reescrever módulo inteiro): confirmar com o usuário antes
