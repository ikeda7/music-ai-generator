# Documento 4 — Constrained Decoding e Lógica de Inferência

## O paradigma: amostragem com restrições musicais

A geração musical neste TCC não é apenas amostragem livre da distribuição aprendida pelo Transformer. O script `generate.py` aplica **Constrained Decoding** — modificações nos logits a cada passo da geração — para incorporar conhecimento musical explícito que o modelo não captura completamente. Esta é uma técnica padrão em geração condicional de texto e música, formalizada como:

```
P_final(x_t) = softmax(logits_t / T + bias_t)
```

onde `logits_t` vêm do modelo, `T` é a temperatura, e `bias_t` é uma máscara aditiva computada por uma função `vocab_constraint_fn(last_token_id) → tensor de tamanho V`.

## Parâmetros base de amostragem

- **Temperature = 0.9**: divide os logits antes do softmax. `T < 1` torna a distribuição mais "pontiaguda" (favorece tokens prováveis); `T > 1` achata. O valor 0.9 foi calibrado empiricamente como ótimo — abaixo disso o modelo entra em loop, acima disso fica errático.
- **Top-k = 40**: filtra os 40 tokens mais prováveis antes da amostragem; outros recebem `-inf`. Reduz cauda longa de tokens improváveis.
- **Top-p = 0.95**: nucleus sampling — remove tokens cuja probabilidade cumulativa excede 0.95. Filtragem complementar ao top-k.
- **Temperatura dinâmica (opcional)**: via `temperature_fn(step)`, permite rampa linear de uma temperatura inicial baixa (mais determinístico no começo) para uma temperatura final alta (mais criativo no desenvolvimento). Não é usado por padrão.

## Filtro de escala via Krumhansl-Schmuckler

Quando o usuário passa `--key C` (ou outra tonalidade), todos os tokens `NOTE_ON_x` cujo pitch class `x % 12` não pertence à escala recebem `-inf` permanentemente em uma `note_mask`. Sete pitch classes válidas em maior, sete em menor (com escala harmônica).

Quando se passa `--auto_key`, executa-se o algoritmo **Krumhansl-Schmuckler**: o modelo gera 100 tokens iniciais sem filtro, conta o histograma de pitch classes, e correlaciona contra os 24 perfis tonais (12 maiores + 12 menores) propostos por Krumhansl & Schmuckler (1986). A tonalidade com maior correlação Pearson é escolhida e o filtro de escala é ativado para o restante da geração. Esta detecção é uma técnica clássica em MIR (Music Information Retrieval).

## A função vocab_constraint_fn — núcleo do controle

A função `build_vocab_constraint(tokenizer, device, key_str)` retorna uma closure `constraint_fn(last_token_id) → tensor`. Esta closure mantém estado mutável (ex: contador de compassos, slot do instrumento ativo, posições recentes) e, a cada passo, retorna um vetor de viés somado aos logits.

As 7 restrições implementadas atualmente:

### 1. VELOCITY só após NOTE_ON (gramatical)

Tokens `VELOCITY_*` recebem `-inf` exceto quando o token anterior foi um `NOTE_ON`. Isso evita o degenerado em que o modelo gera sequências ininterruptas de `VELOCITY VELOCITY VELOCITY...` — comportamento observado em iterações iniciais sem essa proteção.

### 2. Voice Leading por slot

Saltos melódicos grandes recebem penalidade. Há duas máscaras de intervalo:

- **Tight** (slot 1, Melodia): penaliza pitches a mais de 7 semitons do último (5ª justa).
- **Wide** (demais slots): penaliza acima de 12 semitons (oitava).

A magnitude é progressiva: `-0.6 - (jump - 7) × 0.1` para tight; `-0.8 - (jump - 12) × 0.1` para wide. Calibrada para que o pior caso fique em torno de `-2.0` em logit space, mantendo o token candidato com ~13% da probabilidade original — penalizado mas ainda escolhível.

### 3. Repetition Penalty

Tokens `NOTE_ON` ou `TIME_SHIFT` que apareceram nas últimas 16 posições recebem `-1.0`. Reduz repetição mecânica sem proibir totalmente. O valor `-1.0` foi reduzido de `-2.5` (versão anterior) após observação de que penalidades fortes sufocavam a distribuição.

### 4. Bloqueio de EOS nos primeiros 300 tokens

Em iterações iniciais do projeto, o modelo aprendeu a gerar EOS muito cedo (~80 tokens), produzindo amostras curtas demais. A solução foi forçar `EOS = -inf` enquanto `state['steps'] < 300`. Isso garante peças de duração mínima razoável (~30s).

### 5. Bass Anchor no início de BAR (slot 2)

Quando o slot ativo é o Baixo (INSTRUMENT_2) e o token imediatamente anterior foi um `BAR`, todos os pitches cujo `pitch % 12` não é tônica nem quinta da tonalidade recebem `-2.0`. Isto encoraja a função tradicional do baixo: marcar harmonicamente a fundamental no tempo forte. Opt-in via `--key`.

### 6. Chord Backbone I-V-vi-IV (slot 4)

A progressão pop universal I-V-vi-IV é aplicada ao slot Harmonia. Cada compasso usa um acorde distinto, rotativo. Pitches fora do tríade do acorde corrente recebem `-1.5`. Para tonalidade C maior:

- Compasso 1: I = C–E–G
- Compasso 2: V = G–B–D
- Compasso 3: vi = A–C–E
- Compasso 4: IV = F–A–C

Após o compasso 4, recomeça em I. O índice é calculado como `chord_idx = max(0, bar_count - 1) % 4`, garantindo que o primeiro compasso use I (e não V) — correção de um off-by-one identificado em iteração anterior.

### 7. Soft Instrument Re-entry Bias

Esta é a inovação polifônica do TCC. Cada slot `INSTRUMENT_0..4` tem um campo `last_note_step[slot]` no estado. A cada passo, calcula-se `silence = current_step - last_note_step[slot]`. Se algum slot está silencioso há mais de `silence_threshold = 80` tokens, o token `INSTRUMENT_X` correspondente recebe um **bônus positivo crescente**:

```
bonus = min(silence_max_bonus, (silence - silence_threshold) × silence_slope)
```

com `silence_max_bonus = 1.2` e `silence_slope = 0.02`. Após ~140 tokens de silêncio, o bônus satura.

Esta abordagem é fundamentalmente diferente da estratégia de **hard cycling** (forçar troca de instrumento a cada N notas) usada em iterações anteriores. O hard cycling causa **distributional shift**: força o modelo a produzir tokens contra sua distribuição aprendida, gerando padrões antinaturais. O re-entry bias, por outro lado, é uma **recompensa probabilística suave**: o modelo continua livre para escolher, mas é gentilmente "convidado" a re-engajar slots silentes. Quando combinado com a polifonia natural do MAESTRO+POP909, produz texturas de banda sem quebrar a distribuição.

## Razão pela qual penalidades acumuladas falharam

Em uma iteração anterior, foram empilhadas oito constraints com magnitudes de `-1.5`, `-2.5`, `-3.5` etc. No pior caso, um candidato `NOTE_ON` poderia receber `-8.5` em logit space — equivalente a multiplicar sua probabilidade por `e^-8.5 ≈ 0.0002`. O efeito composto era que **todos os candidatos prováveis ficavam suprimidos simultaneamente** e a amostra caía na cauda longa, escolhendo tokens musicalmente estranhos. Resultado audível: alta densidade de notas que "comem umas às outras" e quebra da naturalidade.

A versão atual reduziu drasticamente as magnitudes (worst case ~-4.0) e removeu três constraints redundantes (penalidade de TIME_SHIFT ímpar, penalidade rítmica sequencial, cycling forçado). O modelo voltou a gerar com fluência.

## Modos de renderização para MIDI

Após a geração de tokens, `tokens_to_midi()` produz o arquivo `.mid` final. Três modos:

- **Padrão**: cada slot vai para seu canal GM definido (`Piano→0, Melody→1, Bass→2, Drums→9, Harmony→3`). Programs MIDI são `0, 24, 32, 0, 48` respectivamente.
- **`--render_as_band`**: ignora o slot do token, roteia cada NOTE_ON pelo **registro de pitch** em três timbres GM (Bass, Nylon Guitar, Distortion Lead). Aviso: timbres GM de guitarra/baixo são de baixa fidelidade.
- **`--render_as_trio` (recomendado)**: mesma divisão por registro, mas todas as três tracks usam Grand Piano. Preserva a qualidade de timbre que o MAESTRO ensinou ao modelo.

Em ambos `band` e `trio`, aplica-se um filtro funcional pós-geração:

- **Bass (pitch ≤ 47)**: NOTE_ON é mantido apenas próximo a um BAR boundary (±0.15s); 15% de chance para notas em outros tempos (groove esporádico).
- **Base (48–65)**: apenas notas em **clusters** (≥2 notas em 0.3s) são mantidas — só acordes passam, notas isoladas do registro médio são descartadas.
- **Solo (≥66)**: livre. Monofonia forçada apenas em bass e solo (cada novo NOTE_ON fecha o anterior do mesmo registro).
- **Duração máxima da nota**: cap em 1.0 segundo (ajustável via `--max_note_duration`), prevenindo notas que se sobrepõem por longo tempo.

Esses filtros são post-processing puro: não afetam o modelo e podem ser ajustados sem regerar tokens.
