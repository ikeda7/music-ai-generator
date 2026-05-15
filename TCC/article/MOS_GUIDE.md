# Guia operacional — Avaliação MOS

Pipeline pra coletar avaliação cega comparativa Transformer vs Markov baseline.

## Etapa 1 — Gerar o conjunto de avaliação

```bash
cd "h:/Meu Drive/TCC/TCC"
python make_eval_set.py --output_dir ./eval_samples
```

Produz `eval_samples/sample_A.mid` até `sample_H.mid` (ordem randomizada) e um `legend.json` confidencial mapeando código → modelo. **Não compartilhar a legenda com avaliadores.**

## Etapa 2 — Converter .mid → .mp3

Avaliadores não tocam .mid no navegador. Converter usando soundfont GM:

**Opção A — fluidsynth + soundfont (Linux/WSL):**
```bash
for f in eval_samples/sample_*.mid; do
    fluidsynth -ni -F "${f%.mid}.wav" /path/to/soundfont.sf2 "$f"
    ffmpeg -i "${f%.mid}.wav" -b:a 192k "${f%.mid}.mp3"
done
```

**Opção B — MuseScore (mais fácil no Windows):**
1. Abrir cada `.mid` no MuseScore
2. File → Export → MP3 (192 kbps)
3. Salvar com mesmo nome

**Opção C — script Python (online):**
Usar `midi2audio` (pip install midi2audio) que envolve fluidsynth.

## Etapa 3 — Calcular métricas quantitativas

```bash
python metrics.py --input ./eval_samples --output ./eval_samples/metricas.csv
```

CSV resultante vai pra dissertação (tabela comparativa Transformer vs Markov).

## Etapa 4 — Google Forms — estrutura sugerida

**Configuração:**
- Forms → criar novo formulário
- Tema: Avaliação de Composições Musicais — TCC
- Coletar email: opcional
- Limitar a 1 resposta por pessoa: sim
- Permitir editar resposta: sim

**Instruções no topo:**
> Você ouvirá 8 trechos curtos de música (~60s cada). Avalie cada um em 4 critérios usando escala de 1 (péssimo) a 5 (excelente). Você não saberá quem (ou o quê) compôs cada peça. Tempo estimado: 15 minutos.

**Para cada sample_A.mp3 até sample_H.mp3, criar uma seção com 4 perguntas:**

### Seção: Amostra A
*(inserir link do áudio ou embed do Drive)*

1. **Naturalidade** — Soa como música composta por uma pessoa?
   1 (mecânico/aleatório) ... 5 (humano)
2. **Coerência rítmica** — O ritmo faz sentido? Notas caem nos tempos certos?
   1 (caótico) ... 5 (perfeitamente alinhado)
3. **Qualidade harmônica** — As notas se encaixam? Acordes coerentes?
   1 (dissonante) ... 5 (consonante e claro)
4. **Agradabilidade** — Você ouviria de novo voluntariamente?
   1 (de jeito nenhum) ... 5 (com prazer)

Repetir pras 8 amostras (B até H).

**Pergunta final aberta:**
> Tem algum comentário sobre as amostras? (opcional)

## Etapa 5 — Distribuir

Compartilhar link com 5-10 pessoas. Idealmente:
- Mix de músicos e não-músicos
- Pedir pra ouvir em fone (não no autofalante de notebook)
- Prazo: 1 semana

## Etapa 6 — Análise

Quando tiver as respostas:

```bash
# Forms → Respostas → Download CSV
# Salvar como eval_samples/respostas.csv

python analyze_mos.py --responses eval_samples/respostas.csv \
                     --legend eval_samples/legend.json
```

*(script `analyze_mos.py` ainda a fazer — calcula média por critério por modelo, teste t pra significância)*

## Critério de sucesso do TCC

- Transformer pontua **estatisticamente maior** (p < 0.05) em pelo menos 2 dos 4 critérios
- Naturalidade e Agradabilidade são os mais importantes
- Se Transformer perder em todos → revisar antes da defesa
- Se Transformer empatar → ainda defensável (mostra que ML alcança baseline trivial sem perdas)
