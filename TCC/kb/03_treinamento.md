# Documento 3 — Dinâmica de Treinamento e Otimização

## Objetivo do treinamento

O modelo aprende uma **distribuição de probabilidade autoregressiva** sobre o vocabulário REMI: dado um histórico de tokens musicais, prever o próximo token. Formalmente, o modelo aproxima `P(x_t | x_<t)` para cada posição `t` na sequência, onde `x` são tokens do vocabulário de 349 elementos. O treinamento minimiza a divergência entre essa distribuição prevista e a empírica do dataset.

## Função de perda: Cross-Entropy com Label Smoothing

A perda usada é `nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)`. O parâmetro `label_smoothing=0.1` é uma das decisões mais importantes do projeto e merece análise matemática detalhada.

### Cross-entropy padrão (hard labels)

Sem label smoothing, o alvo de cada posição é um vetor **one-hot**: probabilidade 1.0 no token correto, 0.0 em todos os outros. A perda CE com targets one-hot é:

```
L = -log P(x_t = target | contexto)
```

Para minimizar essa perda a zero, o modelo precisaria atribuir probabilidade exatamente 1.0 ao token alvo, o que, via softmax, exige que o **logit** desse token tenda a `+∞`. Esse é o mecanismo do **overconfidence**: o modelo aprende a saturar sua distribuição.

### O problema concreto: collapse para tokens frequentes

Quando o vocabulário é desbalanceado, esse comportamento é catastrófico. No nosso caso, **TIME_SHIFT é o token mais frequente** (~40% das ocorrências em sequências do MAESTRO/POP909). O modelo aprende rapidamente que, em quase qualquer contexto, prever um TIME_SHIFT pequeno é "estatisticamente seguro". Os logits de TIME_SHIFT saturam, o gradiente para tokens NOTE_ON morre por dois motivos:

1. A probabilidade já está perto de zero para NOTE_ON, então a contribuição do gradiente vira ruído.
2. A norma dos logits de TIME_SHIFT explode, dominando o softmax.

O resultado prático observado neste TCC: no checkpoint da Época 99 (e novamente na 109), o modelo entrou em **mode collapse**, gerando longas sequências de TIME_SHIFTs alternados com uma única NOTE_ON repetida — efeito audível de "drone + silêncio".

### Como label smoothing resolve isso

Label smoothing modifica os targets de one-hot para uma mistura entre o one-hot e a distribuição uniforme:

```
target_LS = (1 - α) · onehot + α · (1/V)
```

onde `α = 0.1` e `V = 349` (tamanho do vocabulário). Isso significa que o modelo precisa atribuir 90% de probabilidade ao token correto e ~0.029% de probabilidade a cada um dos outros 348 tokens. **Não há mais incentivo para saturar logits ao infinito** — a perda mínima alcançável é estritamente positiva (`H(uniform 0.1, onehot 0.9) ≈ 0.51`).

Em termos de regularização, label smoothing impõe um **piso de entropia** sobre a distribuição prevista, prevenindo overconfidence sem alterar qual token é o mais provável. É uma técnica canônica em modelagem de linguagem desde Vaswani et al. (2017) e Szegedy et al. (2016).

### Efeito colateral importante

Como label smoothing impede que a perda chegue a zero, **o val_loss numérico não é diretamente comparável** entre runs com e sem essa flag. Quando reativamos `label_smoothing=0.1` em um treino retomado de checkpoint que foi salvo sem ela, todos os val_losses subsequentes parecerão "piorados" em ~0.3–0.5. Por isso o `train.py` aceita uma flag `--reset_best_loss` que zera o `best_val_loss` armazenado, evitando que o early stopping dispare prematuramente.

## Otimizador AdamW

`torch.optim.AdamW` com:
- `lr = 1e-4`
- `weight_decay = 0.01` — regularização L2 desacoplada (Loshchilov & Hutter, 2019).
- `betas = (0.9, 0.999)` (defaults).

AdamW separa o decaimento de pesos do passo de gradiente, o que estabiliza o treinamento de Transformers em comparação ao Adam clássico. O `weight_decay=0.01` é a recomendação padrão da literatura.

## Learning rate schedule: warmup + cosine decay

```python
def lr_lambda(step):
    if step < warmup:
        return step / warmup       # warmup linear
    progress = (step - warmup) / (total_steps - warmup)
    return max(0.05, 0.5 * (1.0 + cos(π * progress)))   # cosine decay
```

Este schedule, executado a cada batch (não cada época), tem três fases:

1. **Warmup linear** durante 2.000 batches: o LR sobe linearmente de 0 até 1e-4. Isso evita instabilidade nos primeiros passos quando os gradientes ainda são caóticos.
2. **Cosine decay**: o LR decresce suavemente seguindo uma curva cosseno até atingir 5% do valor inicial (`max(0.05, ...)` impede chegar a zero — mantém aprendizado mínimo no fim).
3. O total de steps é calculado como `len(train_loader) × num_epochs`, então mudar `num_epochs` recalibra a curva.

## Gradient clipping

A norma global dos gradientes é limitada a `max_norm = 1.0` via `torch.nn.utils.clip_grad_norm_`. Isso protege contra **exploding gradients** ocasionais — particularmente importante em modelos com janelas longas e atenção causal, onde gradientes podem se acumular ao longo de 512 posições.

## Automatic Mixed Precision (AMP)

O treinamento usa `torch.cuda.amp.autocast` + `GradScaler` para executar forward/backward em FP16:

```python
with torch.cuda.amp.autocast(enabled=use_amp):
    logits = model(inp)
    loss = criterion(logits.reshape(-1, V), tgt.reshape(-1))

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

O `GradScaler` multiplica a perda por um fator grande (~2^16) antes de `backward()` para evitar **underflow** de gradientes pequenos em FP16. O `unscale_` é chamado antes do clipping para que `max_norm` opere no espaço FP32 real. Se `scaler.step()` detectar inf/NaN nos gradientes, ele pula o passo e reduz o fator dinamicamente.

Resultado prático: VRAM cai de ~7 GB para ~4 GB; throughput sobe de 19 it/s para 39 it/s na RTX 4060 Ti — ganho de ~2×.

## Split treino/validação e checkpointing

O dataset é dividido 80/20 com `random_seed=42`. Validação é executada a cada 5 épocas (`eval_every`). Um novo checkpoint é salvo:

1. Sempre que `val_loss` melhora (com prefixo `_best` na log).
2. A cada `save_every` épocas (intervalo padrão = 5).

Uma flag `saved_as_best` evita salvar o checkpoint duas vezes em uma época que combine ambas as condições.

## Diagnóstico automático de mode collapse

A cada 5 épocas, `diagnostico_rapido()` gera uma amostra curta (sem constraints, só com o BOS) e conta pitches únicos. Se houver menos de 3 pitches distintos, exibe um aviso `⚠️ pouca variação`. Se houver mais de 5, `✓ OK`. Isso permite detectar visualmente o início de um colapso ainda durante o treinamento.

## Early stopping

Após 20 validações consecutivas sem melhora (`patience=20`), o treino é interrompido. Como label smoothing impede convergência abaixo de um piso, o early stopping evita ficar treinando indefinidamente em um platô.
