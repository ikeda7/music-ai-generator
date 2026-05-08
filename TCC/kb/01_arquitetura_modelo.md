# Documento 1 — Visão Geral e Arquitetura do Modelo

## Contexto do projeto

Este TCC desenvolve um sistema de geração de música simbólica em formato MIDI utilizando uma arquitetura Transformer treinada do zero. O critério de aprovação prático é que um ouvinte humano consiga identificar o `.mid` gerado como música — não como ruído estatístico. A composição deve apresentar coerência melódica, rítmica e harmônica suficientes para soar intencional.

## Tipo de arquitetura: Transformer Decoder-only

O modelo `MultiInstrumentTransformer` é um Transformer **decoder-only** no estilo GPT. Embora a literatura clássica de Vaswani et al. (2017) descreva um Transformer encoder-decoder voltado a tradução automática, modelos generativos autoregressivos (GPT-2, MuseNet, Music Transformer) usam apenas a pilha de atenção causal. Tecnicamente, na implementação PyTorch, isso é construído com `nn.TransformerEncoder` aplicando uma **máscara causal triangular superior**: cada token só pode atender aos tokens anteriores. O resultado é matematicamente equivalente a uma pilha de blocos de decoder GPT-style — a nomenclatura `Encoder` é apenas um detalhe da API do PyTorch, que não disponibiliza um `TransformerDecoder` standalone.

A escolha por decoder-only se justifica pelo paradigma de geração autoregressiva: a cada passo o modelo prevê o próximo token musical condicionado nos tokens já gerados. Não há sequência de entrada separada — a janela de contexto e a saída ocupam o mesmo espaço de representação.

## Hiperparâmetros e dimensionamento

A configuração adotada é deliberadamente compacta:

- **`d_model = 256`** — dimensão do espaço latente de embedding e estados intermediários.
- **`nhead = 4`** — número de cabeças de atenção. Cada cabeça opera em `d_model / nhead = 64` dimensões, valor canônico para atenção multi-head.
- **`num_layers = 4`** — quatro blocos de Transformer empilhados. Cada bloco contém atenção multi-head + feedforward + residual + layer norm.
- **`dim_feedforward = 1024`** — `4 × d_model`, razão padrão da literatura.
- **`max_seq_length = 2048`** suportado, mas o treinamento e a geração usam janela de **512 tokens** com overlap de 50%.
- **Contagem total: ~3,2 milhões de parâmetros**.

## Embedding posicional senoidal

O modelo utiliza `PositionalEncoding` baseado em senos e cossenos com frequências geometricamente espaçadas, exatamente como descrito no artigo original. As frequências são fixas (não treináveis) e somadas ao embedding de tokens antes da primeira camada. A escolha por encoding senoidal — em vez de `Learned Positional Embedding` ou RoPE — é justificada por duas razões: (1) janela de contexto curta (512 tokens) torna a vantagem de codificações posicionais mais sofisticadas marginal; (2) os tokens estruturais BAR e BEAT já fornecem ao modelo âncoras métricas explícitas, reduzindo a dependência da posição absoluta.

## Weight Tying entre embedding e projeção de saída

Uma técnica de regularização e compressão importante: o tensor de pesos da camada de embedding de tokens (`token_embedding.weight`) é o **mesmo tensor** usado pela camada de projeção de saída (`output_projection.weight`). Em código PyTorch:

```python
self.output_projection.weight = self.token_embedding.weight
```

O efeito prático é triplo:
1. Reduz ~89.000 parâmetros do total (vocabulário 349 × d_model 256).
2. Força consistência entre representações de entrada e saída — o vetor que codifica um token deve ser semanticamente compatível com o vetor que prediz o mesmo token.
3. Estabiliza o treinamento e melhora generalização, conforme demonstrado em Press & Wolf (2017).

## Justificativa de tamanho frente ao hardware

O hardware de treinamento é uma **NVIDIA RTX 4060 Ti com 8 GB de VRAM**. Esta restrição limita modelos significativamente maiores:

- Music Transformer (Huang et al., 2018): d_model=512, 6 camadas, 8 cabeças.
- MuseNet (OpenAI): 24 camadas, 1024 dimensões — exige clusters de A100.

Com `d_model=256` e 4 camadas, conseguimos treinar com `batch_size=16` e janela 512 confortavelmente em 8 GB. A introdução do **Automatic Mixed Precision (AMP)** via `torch.cuda.amp.autocast` reduz o consumo de VRAM em ~40% e acelera o passo de treinamento em ~2×, mantendo os pesos em FP32 mas executando o forward/backward em FP16. Isso permitiria escalar para 6 camadas e `d_model=384` em iterações futuras, mas para o escopo deste TCC a configuração compacta provou ser suficiente para gerar músicas reconhecíveis.

## Forward pass

O fluxo dentro de `forward()` é:

1. Input `(batch_size, seq_len)` → embedding → `(seq_len, batch_size, d_model)` (note a transposição para o formato `seq_first` esperado pelo PyTorch).
2. Soma do encoding posicional.
3. Aplicação da máscara causal: `_generate_square_subsequent_mask()` cria uma matriz triangular superior preenchida com `-inf`, transformada em zeros na diagonal e abaixo. Quando somada aos scores de atenção pré-softmax, garante que cada posição só veja seu passado.
4. Pilha de 4 blocos Transformer com ativação `GELU`.
5. Dropout (taxa 0.1).
6. Projeção final para o vocabulário (com weight tying), produzindo logits sobre os 349 tokens.

## Geração autoregressiva

A função `generate()` itera token a token: a cada passo, faz forward pass com o contexto atual, aplica filtros opcionais (filtro de escala, restrições de vocabulário, temperatura dinâmica), amostra do top-k/top-p com temperatura calibrada, e concatena o novo token. O contexto é truncado a 512 tokens (a mesma janela de treinamento), evitando degradação fora-de-distribuição.

## Limitações arquiteturais reconhecidas

A profundidade de 4 camadas é o piso da literatura para geração musical. Cada camada equivale aproximadamente a um nível de composicionalidade sintática. Com 4 camadas, o modelo raciocina bem sobre motivos curtos e progressões locais, mas estrutura de longo prazo (forma binária, ABA, desenvolvimento temático) é fraca. Esta limitação é documentada na seção de "Trabalhos Futuros" da dissertação.
