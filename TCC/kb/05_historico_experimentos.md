# Documento 5 — Histórico de Experimentos e Diagnósticos

## Cronologia das fases do TCC

### PC1 — Fundamentação teórica (ago–nov 2024)

A fase inicial consistiu em revisão bibliográfica de 20+ referências sobre geração musical com Deep Learning, cobrindo RNNs, LSTMs e Transformers. Os artigos centrais que orientaram a escolha arquitetural foram:

- **Vaswani et al. (2017) — "Attention Is All You Need"**: base do Transformer.
- **Huang et al. (2018) — "Music Transformer"**: aplicação de relative attention a música.
- **Huang & Yang (2020) — "Pop Music Transformer / REMI"**: vocabulário usado neste TCC.

A decisão de adotar Transformer (vs LSTM clássico) foi tomada nesta fase. Pré-projeto e revisão foram entregues e aprovados.

### PC2 — Protótipo (2025–2026)

Implementação completa do sistema. O código atual em `TCC/` é o produto desta fase. Datasets foram baixados localmente (MAESTRO, POP909, Groove). Primeira versão funcional rodou em ~80 épocas no MAESTRO solo, mas colapsou para nota única.

### Migração de dataset: MAESTRO+POP909+Groove

O **mode collapse** do treino só com MAESTRO foi atribuído à pouca diversidade — todo MAESTRO é piano clássico solo, levando o modelo a convergir para padrões muito estreitos. Acrescentaram-se POP909 (multi-instrumental) e Groove (bateria). Cache de tokenização foi reconstruído. A partir desse ponto, todos os treinamentos usam os três datasets juntos.

## Treinamento principal e checkpoints relevantes

O treinamento contínuo gerou uma série de checkpoints. Os marcantes são:

### Época 49 — primeiro resultado válido

Após ~12 horas de treino, o checkpoint da época 49 foi o primeiro a produzir um `.mid` reconhecível como música. Apresentava melodia com contorno, baixo identificável, mas com transições rítmicas abruptas. Validado como "baseline funcional" em discussão com Gemini (assistente externo de análise) e o orientador Prof. Danillo Roberto Pereira (UNESP).

### Época 74 — checkpoint GOLD

Cinco épocas depois, o checkpoint 74 representou um salto qualitativo: a amostra `teste_ep74.mp3` foi descrita como "divisor de águas no TCC" — primeira vez que a peça gerada apresenta:

- Fraseado real (frase começa, desenvolve, descansa).
- Dinâmica audível (notas com forças diferentes).
- Textura limpa, sem o "paredão sufocado" de versões anteriores.
- Coerência tonal preservada com `--key C`.

A duração de ~108 segundos com 415 notas em range MIDI 30–90 mostra distribuição rica entre registros baixo, médio e agudo. Este checkpoint foi **congelado como referência final do TCC**.

### Épocas 99 e 109 — mode collapse

Após platôs longos no val_loss (~2.39) com learning rate baixo (~5×10⁻⁵), o modelo entrou em um regime de overfitting local que produziu collapse audível:

- **Época 99**: drone notes — uma única nota repetida com pequenos silêncios.
- **Época 109**: dyade fixo — duas notas (E3 + G3) tocando em paralelo continuamente por toda a peça, mais um pitch grave esporádico no final.

A análise do piano roll de `teste_ep109.png` mostrou duas linhas horizontais paralelas em pitches 52 e 55, ocupando os 78 segundos da geração. Este é um caso textbook de **memorização local de padrão frequente** — o modelo, sem capacidade de gradiente significativo (LR baixo + label smoothing já saturado), reforçou padrões super-frequentes do MAESTRO.

### Decisão tática: rollback para ep74

Os checkpoints 89, 94, 99, 104 e 109 foram descartados (deletados do disco). A decisão foi documentada em `CLAUDE.md`: **congelar ep74 como produto final do TCC**, evitando treinamento além do ponto de retorno positivo. Recursos passaram a ser dedicados a refinar a geração (constraints, render) e às pendências acadêmicas (avaliação MOS, baseline Markov, dissertação).

## Evolução das constraints de geração

A função `build_vocab_constraint` passou por quatro fases:

### Fase 1 — Hard constraints empilhadas

A primeira versão tinha **8 constraints** com penalidades fortes:

1. VELOCITY gate
2. Cycling forçado de instrumentos a cada 16 notas (-inf em outros INSTRUMENT)
3. Penalidade `-1.5` em TIME_SHIFT com steps ímpares
4. Repetition penalty `-2.5`
5. Penalidade extra `-3.5` para TIME_SHIFT idêntico 3× seguidas
6. Penalidade progressiva por intervalo `>12` semitons
7. Voice leading `-1.5` extra para Melodia
8. EOS block

**Soma worst-case: −8.5 em logit space** = candidato com ~0.02% da probabilidade original. Resultado: distribuição sufocada, modelo amostrando da cauda longa, gerando tokens estranhos. Densidade de notas explodia (acúmulo) porque TIME_SHIFTs eram penalizados em demasiadas dimensões.

### Fase 2 — Remoção do que era redundante

Removidas constraints 2, 3, 5 (cycling, ímpar, rítmico-sequencial). Constraint 4 reduzida de `-2.5` para `-1.0`. Worst-case caiu para ~−4.0. Geração melhorou drasticamente — esta é a versão que produziu `teste_ep74`.

### Fase 3 — Refactor por registro (regressão)

Tentou-se substituir a lógica baseada em **slots de instrumento** por **registros de pitch** (baixo ≤47, base 48–65, solo ≥66). Adicionou-se "force INSTRUMENT_0" para gerar apenas piano. Foi a hipótese de que três pianos sobrepostos resolveriam a polifonia.

**Resultado**: regressão severa. O modelo, treinado em dataset onde MAESTRO domina (74h piano solo no slot 0) mas POP909 separa instrumentos em slots 1–4, sofreu **distributional shift**. Forçar `INSTRUMENT_0` único produziu monotonia (`teste_piano_banda.png` mostrou range 67–72, sem baixo) e ocasionalmente colapso em dyade (`teste_banda.png` mostrou pitches 40+48 fixos quase a peça inteira).

### Fase 4 — Reverter para slot-based, render por registro

A versão final mantém o decoder na **lógica slot-based** (mesma que produziu ep74). O remapeamento por registro foi movido para o **render** (post-processing em `music_utils.py`), via flags `--render_as_band` ou `--render_as_trio`. Esta separação de responsabilidades é arquiteturalmente correta: o modelo gera tokens conforme aprendeu, o renderizador decide como atribuir timbres.

## Lições de design extraídas

### 1. Soft biases > hard constraints

A diferença entre `-inf` (proibição) e `-2.0` (penalidade soft) é qualitativamente importante. Hard constraints quebram a distribuição quando o modelo "queria" ir naquela direção; soft constraints preservam a estrutura aprendida e apenas redirecionam suavemente. Esta é uma observação que vale a pena destacar na seção de Discussão da dissertação.

### 2. Distributional shift é real e mensurável

Forçar tokens raros (no caso, eliminar slots 1–4) tem custo audível imediato. O modelo não generaliza além do que viu no treino na distribuição em que viu. Isso reforça a importância de não introduzir constraints que reflitam um cenário de geração diferente do cenário de treino.

### 3. Label smoothing é essencial em vocabulários desbalanceados

A correção do collapse da época 99 veio de adicionar `label_smoothing=0.1` no retreino. Sem essa intervenção matemática, o modelo continuou colapsando por overconfidence em TIME_SHIFT. Esta decisão é a mais defensável e quantificável da metodologia.

### 4. Render-time post-processing é uma alavanca subutilizada

Muitas modificações que pareciam exigir retreinamento podem ser feitas no render: filtros funcionais (bass quantizado ao BAR, base só em clusters), monofonia forçada, redução de duração máxima, remapeamento de timbre. Isso permitiu iteração rápida nas últimas etapas, sem custar GPU-hours.

### 5. Hardware modesto é viável com AMP e modelo compacto

Treinar em RTX 4060 Ti (8 GB) parecia limitante a princípio. Com `d_model=256`, 4 camadas, AMP e gradient clipping, foram necessárias ~12h para chegar ao ep74. Música simbólica não exige modelos gigantes — o gargalo principal é o **dataset** e a **calibração de loss**, não a contagem de parâmetros.

## Pendências para a defesa

1. **Avaliação MOS** com 5–10 ouvintes humanos (Mean Opinion Score com 4 dimensões: naturalidade, coerência rítmica, qualidade harmônica, agradabilidade).
2. **Baseline Markov chain** (bigrama de pitches do MAESTRO) para comparação quantitativa — exigência de revisores.
3. **Métricas quantitativas**: diversidade de pitches, densidade de notas/segundo, similaridade com dataset (KL-divergence sobre histogramas).
4. **Atualização dos slides PC2** (configuração antiga ainda mostra 6 layers/8 heads).
5. **Dissertação final** — esqueleto em `notes.md`.

## Reprodutibilidade

Todos os experimentos usam `random_seed = 42`. Os checkpoints relevantes preservados são `checkpoint_epoch_49.pt`, `checkpoint_epoch_74.pt` (GOLD) e `checkpoint_epoch_79.pt`. O comando final canônico para gerar uma amostra de defesa é:

```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_74.pt \
    --output samples/composicao_final.mid \
    --key C --temperature 0.9 --top_k 40 --tempo 100 --render_as_trio
```
