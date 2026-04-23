# Notas do TCC — Geração Musical com Transformer
**Lucas Vinícius de Carvalho Ikeda**
Orientador: Prof. Dr. Danillo Roberto Pereira — UNESP
Última atualização: abril/2026

---

## Avaliação Técnica do Projeto

> *Perspectiva de especialista — o que está bem e o que precisa de atenção antes da defesa.*

### O que está bem feito

O projeto tem **solidez conceitual acima da média para TCC de graduação**. A escolha do Transformer autoregressivo com tokenização REMI-like está alinhada com o estado da arte — é a mesma abordagem usada em papers como [REMI (Huang et al., 2020)](https://arxiv.org/abs/2002.00212) e [Music Transformer (Huang et al., Google Brain, 2018)](https://arxiv.org/abs/1809.04281).

Pontos positivos:
- Três datasets complementares: piano solo (MAESTRO) + multi-instrumental (POP909) + percussão (Groove MIDI)
- Data augmentation por transposição (±4, ±8 semitons)
- Pipeline completo do zero: tokenização → treinamento → geração → avaliação
- BAR/BEAT tokens para estrutura rítmica
- Constrained decoding para coerência musical durante inferência
- Ciclo diagnóstico → correção → iteração conduzido de forma engenheira

---

### Problemas que um revisor vai apontar

#### 1. Terminologia errada — o mais crítico
O modelo é chamado de "encoder-only com causal masking". Isso é uma contradição — encoder-only é o BERT (bidirecional). O que está implementado é um **decoder-only Transformer (estilo GPT)**.

> **Ação:** Corrigir em toda a dissertação antes de qualquer outra coisa.

#### 2. Representação temporal não normalizada
`TIME_SHIFT` usa tempo absoluto em segundos. Uma música a 90 BPM e outra a 120 BPM geram padrões completamente diferentes para o mesmo ritmo. O ideal seria normalizar para beat-relative (quantizar em função do BPM de cada arquivo). Os tokens BAR ajudam, mas não resolvem completamente.

#### 3. Constraints de geração como "muleta"
O filtro de escala, penalidade de intervalo e VELOCITY constraint são correções pós-treino de deficiências do modelo. Um modelo bem treinado com dados suficientes aprenderia essas regras sozinho.

> **Como apresentar:** Enquadrar como **constrained decoding** — técnica legítima amplamente usada na literatura (ChatGPT, modelos de código, Anticipatory Music Transformer - Google 2023). Não é fraqueza, é uma decisão de projeto consciente.

**Parágrafo sugerido para a dissertação:**
> *"Durante a inferência, aplicamos restrições de vocabulário baseadas em teoria musical — filtro de escala via Krumhansl-Schmuckler e grade rítmica — como técnica de constrained decoding. Essa abordagem é complementar ao aprendizado do modelo e documentada na literatura como forma de melhorar a coerência musical sem comprometer a natureza generativa do sistema."*

#### 4. Avaliação fraca
5–10 pessoas com questionário subjetivo é o mínimo aceitável. Um revisor vai pedir:
- Escala Likert padronizada (**MOS — Mean Opinion Score**)
- Avaliação cega (sem dizer que é IA)
- Comparação com pelo menos um baseline (ex: cadeia de Markov de ordem 2)

#### 5. Modelo pequeno para o problema
3,2M de parâmetros é viável para TCC mas limitado. Music Transformer usa 35M+. O resultado sonoro reflete isso — é uma limitação honesta a documentar.

---

### Tabela de prioridades

| Prioridade | Ação |
|---|---|
| Alta | Corrigir "encoder-only" → "decoder-only" na dissertação |
| Alta | Adicionar 1 baseline simples (Markov chain) para comparação quantitativa |
| Alta | Usar MOS score na avaliação com usuários |
| Média | Documentar as constraints como decisões de inferência, não arquitetura |
| Baixa | Normalizar TIME_SHIFT por BPM (melhora qualidade mas dá trabalho) |

> **Veredicto:** Para um TCC de graduação no Brasil, o projeto está acima da média — poucos trabalhos chegam a treinar um Transformer do zero com pipeline completo. A dissertação precisa de atenção na terminologia e na seção de avaliação.

---

## Evolução do Projeto — O que Mudou e Por Quê

> Documentar a evolução não é fraqueza — o revisor espera ver que você entendeu o problema, tomou uma decisão e o resultado melhorou. Isso é **rigor científico**.

| O que mudou | Por que mudou | Onde citar no artigo |
|---|---|---|
| LSTM → Transformer | Transformers capturam dependências longas sem vanishing gradient | Fundamentação / Metodologia |
| MAESTRO-only → +POP909+Groove | Mode collapse para pitch único com dataset homogêneo | Metodologia / Resultados |
| Tokenização simples → REMI-like com BAR/BEAT | Grid rítmico necessário para coerência temporal | Metodologia |
| Geração livre → constrained decoding | VELOCITY loops, notas fora de escala, sem troca de instrumento | Metodologia |
| `max_note_duration` 4s → 1,5s | Notas drone longas degradavam qualidade perceptual | Metodologia |
| Modelo TensorFlow → PyTorch | Melhor ecossistema de pesquisa, suporte CUDA mais maduro | Introdução / Metodologia |
| 6 camadas, 8 heads → 4 camadas, 4 heads | Limitação de VRAM (8 GB) — calibração para RTX 4060 Ti | Metodologia / Experimentos |

---

## Estrutura do Artigo

> Formato **SBC** (Sociedade Brasileira de Computação) — padrão usado na UNESP/UNOESTE para TCC com orientação do Prof. Danillo. Double-column, 8–12 páginas.

```
Título + Autores
Abstract (PT + EN)          ← escrever por último
1. Introdução               ← anteprojeto.pdf + motivação geral
2. Fundamentação Teórica    ← revisao_bibliografica.pdf (Transformers, REMI, datasets)
3. Metodologia              ← arquitetura + decisões + evolução do projeto
   3.1 Pipeline de dados e tokenização
   3.2 Arquitetura do modelo (decoder-only Transformer)
   3.3 Treinamento e data augmentation
   3.4 Geração com constrained decoding
4. Experimentos             ← datasets, hardware, hiperparâmetros, métricas
5. Resultados               ← piano rolls, curva de loss, avaliação MOS
6. Conclusão                ← o que funcionou, limitações, trabalhos futuros
Referências                 ← PC1/artigos/ + datasets oficiais
```

### Onde usar cada fonte de conteúdo

| Seção | Fonte principal |
|---|---|
| Introdução | `PC1/anteprojeto/anteprojeto.pdf` |
| Fundamentação Teórica | `PC1/revisao_bibliografica/revisao_bibliografica.pdf` + artigos |
| Metodologia | Código em `TCC/` + decisões documentadas aqui |
| Experimentos | `TCC/config.json` + logs de treino |
| Resultados | `TCC/samples/` + piano rolls + formulário MOS |
| Referências | `PC1/artigos/` (Vaswani 2017 obrigatório) |

### Ordem recomendada para escrever

1. **Metodologia** — você já sabe tudo, é só documentar o código
2. **Experimentos** — copia do config.json + logs
3. **Resultados** — após testar época 200
4. **Fundamentação Teórica** — usa o NotebookLM com os PDFs do PC1
5. **Introdução e Conclusão** — mais fácil depois que o resto está escrito
6. **Abstract** — sempre por último

---

## Workflow de Ferramentas para Escrever o Artigo

| Tarefa | Ferramenta |
|---|---|
| Revisão bibliográfica, citações, embasamento teórico | **NotebookLM** (sobe os PDFs de `PC1/artigos/`) |
| Análise musical dos outputs, diagnóstico de qualidade | **Gemini** (consegue ouvir áudio) |
| Engenharia de código, debugging, pipeline técnico | **Claude Code** |
| Rascunho de seções técnicas (Metodologia, Resultados) | **Claude Code** + revisão sua |
| Organização de ideias, estrutura do documento | **NotebookLM** |

> **Dica:** Antes de começar a escrever, peça ao Claude para gerar um esqueleto completo da dissertação com as seções mapeadas para o que foi implementado. Assim você vai pro NotebookLM com estrutura definida, não com página em branco.

---

## Referências Essenciais (já disponíveis em `PC1/artigos/`)

| Arquivo | Referência |
|---|---|
| `1706.03762v7.pdf` | Vaswani et al. (2017) — "Attention Is All You Need" — **obrigatório** |
| `lstm.pdf` | Fundamentos de LSTM — útil como baseline comparativo |
| `GERACAO_DE_MUSICA_COM_APRENDIZADO_DE_MAQUINA.pdf` | Contexto em português — bom para a introdução |
| `Harmony_and_algorithm_...pdf` | Revisão de IA generativa musical |
| `Generative_Adversarial_Network_...pdf` | GANs para música — comparação arquitetural |
| `MellisAI.pdf` | Sistema similar — comparação de abordagem |

**Referências a citar que não estão nos arquivos (buscar):**
- Huang et al. (2020) — REMI: Pop Music Transformer
- Huang et al. (2018) — Music Transformer (Google Brain)
- Anticipatory Music Transformer (Thickstun et al., Google, 2023) — constrained decoding
- MAESTRO dataset paper (Hawthorne et al., 2019)
- POP909 dataset paper (Wang et al., 2020)

---

## Pendências para a Defesa

- [ ] Treino completar 200 épocas
- [ ] Testar checkpoint final com `generate.py`
- [ ] Enviar amostras ao Gemini para diagnóstico de qualidade
- [ ] Conduzir avaliação com usuários (5–10 pessoas, escala MOS, avaliação cega)
- [ ] Adicionar baseline Markov chain para comparação quantitativa
- [ ] Corrigir "encoder-only" → "decoder-only" em toda documentação
- [ ] Atualizar slides PC2 (ainda mostram 6 camadas/8 heads/MAESTRO+Groove)
- [ ] Confirmar com Prof. Danillo o template exato (SBC vs. ABNT UNESP)
- [ ] Escrever dissertação final

---

## GPU e Hardware

- **RTX 4060 Ti (8 GB VRAM)** — 90%+ de uso durante treino é normal e desejado
- **55°C** em carga total é frio — temperatura segura fica abaixo de 83°C
- Média de ~25 min por época no dataset atual (MAESTRO + POP909 + Groove)
