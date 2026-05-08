# Documento 2 — Pipeline de Dados e Vocabulário

## Datasets utilizados

O treinamento combina três corpora MIDI complementares, totalizando ~5.300 arquivos:

| Dataset | Arquivos | Conteúdo | Função no treino |
|---|---|---|---|
| **MAESTRO v3** | 1.276 | 74 horas de piano clássico solo (concertos internacionais, captado de pianos digitais Disklavier) | Voz principal, dados limpos, alta qualidade — ensina coerência harmônica e fraseado |
| **POP909** | 2.897 | Música pop chinesa multi-instrumental com tracks separados (melodia, piano, bass) | Ensina separação funcional entre vozes e estruturas de acompanhamento |
| **Groove MIDI** | 1.150 | Performances humanas de bateria (com micro-timing) | Conteúdo rítmico para slot de percussão |
| Lakh MIDI | 15.754 | Não usado neste TCC | Demanda filtragem agressiva (densidade, instrumentação, exclusão de C major artefato de scraping) |

A combinação de MAESTRO + POP909 + Groove cobre simultaneamente: música solo expressiva (MAESTRO), arranjos multi-instrumentais (POP909) e percussão idiomática (Groove). A diversidade evita o **mode collapse** observado em treinos só com MAESTRO, em que o modelo aprendeu a repetir uma única nota.

## Pipeline de processamento

```
Arquivo .mid   →  MIDIProcessor.load_midi()           [parsing com mido]
               →  _quantize_and_add_time_shifts()      [discretização temporal]
               →  MIDITokenizer.encode_events()        [conversão para inteiros]
               →  prepare_sequences()                  [janelas 512 com overlap 50%]
               →  MusicDataset / DataLoader            [batches no PyTorch]
```

A primeira etapa carrega o arquivo MIDI usando a biblioteca `mido` e extrai uma lista de objetos `MusicalEvent` (NOTE_ON, NOTE_OFF, TIME_SHIFT). O `_quantize_and_add_time_shifts` mapeia o tempo absoluto em "steps" discretos: a `quantization_resolution = 16` significa 16 steps por segundo, ou seja, cada step corresponde a `1/16 = 0.0625s`. A 120 BPM, 1 beat = 0.5s = 8 steps, alinhando naturalmente à colcheia.

## Vocabulário REMI-like (~349 tokens)

O vocabulário é inspirado no esquema **REMI** (REvamped MIDI-derived events) de Huang & Yang (2020), com adaptações para representação multi-instrumental e estrutura de compasso:

### Tokens especiais (4)
- `PAD = 0` — padding para sequências menores que a janela.
- `BOS = 1` — beginning-of-sequence, usado como prompt inicial na geração.
- `EOS = 2` — end-of-sequence, sinaliza o fim natural da peça.
- `MASK = 3` — reservado (não usado, modelo é causal LM, não BERT-style).

### Tokens de instrumento (5)
`INSTRUMENT_0..4` mapeiam para slots funcionais:
- **0 — Piano** (Acoustic Grand)
- **1 — Melodia** (Nylon Guitar como timbre default)
- **2 — Baixo** (Acoustic Bass)
- **3 — Bateria** (canal 9 GM)
- **4 — Harmonia** (String Ensemble)

A conversão de programas MIDI para slots ocorre em `_create_instrument_map()`: programas 0–7 viram slot 0; 24–31 → slot 1; 32–39 → slot 2; 128 (drums) → slot 3; 48–55 → slot 4. Programas fora dessas faixas são distribuídos por módulo.

### Tokens de altura (176)
- `NOTE_ON_21..108` — 88 pitches MIDI (extensão de piano padrão A0–C8).
- `NOTE_OFF_21..108` — fim de cada nota.

### Tokens de dinâmica (32)
- `VELOCITY_0..31` — velocity quantizada em 32 bins. Reduz vocabulário (vs 128 valores brutos) sem perder expressividade audível.

### Tokens temporais (128)
- `TIME_SHIFT_1..128` — avança o tempo em N steps. Permite saltos de até 8 segundos sem fragmentação.

### Tokens estruturais (Bar-Relative Encoding, 4)
- `BAR` — marca início de compasso (a cada 32 steps em 4/4 a 120 BPM).
- `BEAT_2`, `BEAT_3`, `BEAT_4` — marcam tempos 2, 3 e 4 dentro do compasso.

A inclusão de BAR e BEAT é uma decisão arquitetural crítica: ensina ao modelo a noção de **forma métrica**, permitindo que constraints na geração (chord backbone, bass anchor) operem sobre essa estrutura.

## Sequências de treinamento

Após a tokenização, cada arquivo MIDI vira uma lista plana de inteiros. Esta lista é fatiada em janelas de 512 tokens com **overlap de 50%** (passo de 256 tokens). O overlap garante que o modelo veja transições entre janelas durante o treino, melhorando a capacidade de gerar sequências contínuas. A função `prepare_sequences()` retorna pares `(input, target)` onde target é input deslocado por 1 — paradigma autoregressivo padrão.

## Cache de tokenização

A tokenização de 5.300 arquivos é custosa (~30 minutos). O sistema gera um arquivo `.cache_tokens_<hash>.pkl` cuja chave hash inclui caminhos dos datasets, `seq_length`, `quantization_resolution` e parâmetros de augmentation. Isso evita re-tokenização entre execuções de `train.py`. A flag `--rebuild_cache` força recálculo.

## Data Augmentation: transposição

Para aumentar a diversidade musical sem coletar mais dados, aplica-se transposição em três variantes: `[0, +4, +8]` semitons. Isso triplica efetivamente o dataset. Notas que ultrapassam o range MIDI 21–108 após transposição são **descartadas silenciosamente** dos eventos. **Bateria (slot 3) é excluída da transposição** — pitch em drums não é nota tonal, é índice de peça (kick, snare, hihat), e transpor quebraria semântica.

A função responsável é `_transpose_events(events, semitones, min_pitch, max_pitch)` em `train.py`. Os offsets `[0, +4, +8]` foram escolhidos para cobrir terças maiores e quintas — intervalos musicalmente significativos — sem expandir o conjunto a um ponto que a memória não comporte.

## Persistência do vocabulário

O vocabulário completo (mapeamento token → id) é **salvo dentro do checkpoint** `.pt`. Isso significa que `generate.py` sempre reconstrói o tokenizer a partir do checkpoint, garantindo consistência mesmo se a definição em código mudar entre execuções. É o que permite voltar a checkpoints antigos sem se preocupar com versionamento explícito do vocab.
