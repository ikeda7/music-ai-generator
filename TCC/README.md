# Sistema de Geração de Música com IA - Transformer Multi-Instrumental

Sistema completo de geração de música usando arquitetura Transformer, capaz de processar e gerar composições multi-instrumentais a partir de arquivos MIDI.

## 🚀 Início Rápido

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Baixar datasets (MAESTRO, Groove, POP909)
python download_datasets.py --all

# 3. Treinar o modelo (use todos os datasets para melhor qualidade)
python train.py --data_path ./datasets

# 4. Gerar música
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --output minha_musica.mid
```

**💡 Dica:** Para gerar música de **qualidade profissional**, use múltiplos datasets e treine por muitas épocas (100+).

## Características

- **Arquitetura Transformer**: Baseado em `nn.TransformerEncoder` do PyTorch
- **Multi-Instrumental**: Suporta 5 instrumentos simultâneos (Piano, Melodia, Baixo, Bateria, Harmonia)
- **Processamento MIDI**: Pipeline robusto para pré-processar arquivos MIDI
- **Tokenização REMI-like**: Representação eficiente de eventos musicais
- **Geração Autoregressiva**: Gera música usando sampling com temperatura

## Estrutura do Projeto

```
TCC/
├── data_processor.py      # Processamento e tokenização de MIDI
├── model.py               # Arquitetura Transformer multi-instrumental
├── music_utils.py         # Utilitários musicais e conversão MIDI
├── train.py               # Script de treinamento
├── generate.py            # Script de geração/inferência
├── config.json            # Hiperparâmetros
├── requirements.txt       # Dependências
└── README.md             # Este arquivo
```

## Instalação

### 1. Requisitos

- Python 3.8 ou superior
- PyTorch (veja [instalação oficial](https://pytorch.org/))
- ~10-20 GB de espaço em disco para datasets
- GPU recomendada (mas funciona em CPU também)

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
pip install torch mido pretty_midi numpy pandas tqdm matplotlib scipy
```

### 3. Baixar Datasets

Para treinar um modelo de qualidade, você precisa de datasets MIDI. Execute o script de download:

```bash
# Baixar todos os datasets disponíveis
python download_datasets.py --all

# Ou baixar datasets específicos
python download_datasets.py --maestro
python download_datasets.py --groove
python download_datasets.py --pop909

# Especificar diretório de destino
python download_datasets.py --all --datasets_dir ./meus_datasets
```

**Datasets disponíveis:**
- **MAESTRO**: ~200 horas de performances de piano (recomendado)
- **Groove**: Padrões de bateria MIDI de alta qualidade
- **POP909**: 909 músicas pop com múltiplos tracks (melodia, piano, baixo)
- **Lakh MIDI**: Dataset gigante (~170k músicas) - requer download manual

**Nota:** Alguns downloads podem falhar automaticamente. Nesse caso, o script fornecerá links e instruções para download manual.

## Uso Rápido

### Passo a Passo Completo

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Baixar datasets:**
   ```bash
   python download_datasets.py --all
   ```

3. **Treinar o modelo:**
   ```bash
   python train.py --data_path ./datasets
   ```

4. **Gerar música:**
   ```bash
   python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --output minha_musica.mid
   ```

### Treinamento

Para treinar o modelo com um dataset MIDI:

```bash
python train.py --data_path /caminho/para/dataset/midi
```

**Parâmetros:**
- `--data_path`: Caminho para diretório ou arquivo MIDI (obrigatório)
- `--config`: Caminho para arquivo de configuração (padrão: `config.json`)
- `--resume`: Caminho para checkpoint para continuar treinamento (opcional)
- `--device`: Device para treinamento: `auto`, `cpu`, ou `cuda` (padrão: `auto`)

**Exemplos:**
```bash
# Treinar com todos os datasets baixados (RECOMENDADO para melhor qualidade)
python train.py --data_path ./datasets

# Treinar apenas com MAESTRO
python train.py --data_path ./datasets/maestro

# Continuar treinamento a partir de checkpoint
python train.py --data_path ./datasets --resume checkpoints/checkpoint_epoch_50.pt

# Treinar em GPU específica
python train.py --data_path ./datasets --device cuda
```

### Geração

Para gerar música usando um modelo treinado:

```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --output minha_musica.mid
```

**Parâmetros:**
- `--checkpoint`: Caminho para checkpoint do modelo (obrigatório)
- `--output`: Caminho para salvar o arquivo MIDI gerado (padrão: `generated_music.mid`)
- `--config`: Caminho para arquivo de configuração (padrão: `config.json`)
- `--length`: Comprimento máximo da geração em tokens (padrão: do config)
- `--temperature`: Temperatura para sampling (padrão: do config)
- `--top_k`: Top-k sampling (padrão: do config)
- `--top_p`: Top-p (nucleus) sampling (padrão: do config)
- `--priming`: Arquivo MIDI para usar como priming/condicionamento inicial (opcional)
- `--num_generations`: Número de músicas para gerar (padrão: 1)
- `--device`: Device para geração: `auto`, `cpu`, ou `cuda` (padrão: `auto`)

**Exemplos:**
```bash
# Geração básica
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt

# Geração com temperatura mais alta (mais criativa)
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --temperature 1.2

# Geração com priming (condicionamento inicial)
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --priming referencia.mid

# Gerar múltiplas músicas
python generate.py --checkpoint checkpoints/checkpoint_epoch_50.pt --num_generations 5
```

## Configuração

O arquivo `config.json` contém todos os hiperparâmetros configuráveis:

### Modelo
- `d_model`: Dimensão do modelo (padrão: 512)
- `nhead`: Número de heads de atenção (padrão: 8)
- `num_layers`: Número de camadas do encoder (padrão: 6)
- `dim_feedforward`: Dimensão da camada feedforward (padrão: 2048)
- `dropout`: Taxa de dropout (padrão: 0.1)

### Dados
- `seq_length`: Comprimento das sequências de treinamento (padrão: 512)
- `num_instruments`: Número de instrumentos (padrão: 5)
- `quantization_resolution`: Resolução de quantização temporal (padrão: 16)

### Treinamento
- `batch_size`: Tamanho do batch (padrão: 8)
- `learning_rate`: Taxa de aprendizado (padrão: 0.0001)
- `num_epochs`: Número de épocas (padrão: 100)
- `save_every`: Salvar checkpoint a cada N épocas (padrão: 10)
- `eval_every`: Avaliar a cada N épocas (padrão: 5)

### Geração
- `temperature`: Temperatura para sampling (padrão: 1.0)
- `max_length`: Comprimento máximo da geração (padrão: 1024)
- `top_k`: Top-k sampling (padrão: 50)
- `top_p`: Nucleus sampling (padrão: 0.95)

## Formatos de Dados

### Datasets Suportados

O sistema suporta qualquer dataset MIDI, incluindo:
- **MAESTRO**: Dataset de performances de piano (~200 horas)
- **Groove**: Dataset de bateria MIDI de alta qualidade
- **POP909**: 909 músicas pop com múltiplos tracks
- **Lakh MIDI**: Dataset extenso (~170k músicas)
- Qualquer coleção de arquivos `.mid` ou `.midi`

### Preparação de Dados

Após baixar os datasets, eles estarão organizados assim:

```
datasets/
├── maestro/
│   ├── arquivo1.mid
│   ├── arquivo2.mid
│   └── ...
├── groove/
│   ├── arquivo1.mid
│   └── ...
└── pop909/
    ├── arquivo1.mid
    └── ...
```

O sistema processará automaticamente todos os arquivos MIDI encontrados.

### Treinamento com Múltiplos Datasets

Para obter melhor qualidade, treine com todos os datasets:

```bash
# Treinar com todos os datasets no diretório
python train.py --data_path ./datasets

# Ou especificar múltiplos diretórios (um por vez)
python train.py --data_path ./datasets/maestro
python train.py --data_path ./datasets/groove --resume checkpoints/checkpoint_epoch_50.pt
```

**Dica:** Para treinamento extensivo, combine múltiplos datasets. Quanto mais dados de qualidade, melhor será a música gerada!

## Arquitetura

### Modelo Transformer

O modelo utiliza uma arquitetura baseada em `nn.TransformerEncoder`:
- **Embedding de Tokens**: Converte tokens em vetores de dimensão `d_model`
- **Encoding Posicional**: Adiciona informação temporal
- **Multi-Head Attention**: Captura dependências de longo prazo
- **Feedforward Layers**: Processa representações
- **Head de Saída**: Gera distribuição sobre vocabulário

### Tokenização

O sistema usa uma abordagem REMI-like:
- **Tokens Especiais**: PAD, BOS, EOS, MASK
- **Tokens de Instrumento**: Identificam qual instrumento está tocando
- **Tokens de Pitch**: NOTE_ON e NOTE_OFF para cada nota
- **Tokens de Velocity**: Intensidade das notas (quantizada)
- **Tokens de Tempo**: TIME_SHIFT para sincronização temporal

## Dicas de Uso

### Treinamento
- **Use múltiplos datasets** para melhor qualidade musical
- Comece com datasets pequenos para testar o pipeline rapidamente
- Para qualidade máxima, use MAESTRO + POP909 + Groove (ou mais)
- Ajuste `batch_size` no `config.json` de acordo com a memória disponível
- Use `gradient_clip` para estabilizar o treinamento
- Monitore o loss de validação para evitar overfitting
- **Treine por muitas épocas** (100+) para resultados de qualidade profissional
- Quanto mais dados de treinamento, melhor a qualidade final

### Geração
- **Temperatura baixa (0.5-0.8)**: Gera música mais conservadora e previsível
- **Temperatura média (1.0-1.2)**: Equilíbrio entre criatividade e coerência
- **Temperatura alta (1.5-2.0)**: Gera música mais variada e criativa
- Use `priming` para condicionar a geração com um início específico
- Experimente `top_k` e `top_p` para controlar a diversidade

## Troubleshooting

### Erro: "Nenhum arquivo MIDI encontrado"
- Verifique se o caminho do dataset está correto
- Certifique-se de que os arquivos têm extensão `.mid` ou `.midi`

### Erro: "CUDA out of memory"
- Reduza `batch_size` no `config.json`
- Reduza `seq_length` no `config.json`
- Use `--device cpu` para treinar em CPU

### Geração muito lenta
- Use GPU para geração: `--device cuda`
- Reduza `max_length` no `config.json` ou `--length`

### Música gerada não soa bem
- Verifique se o modelo foi treinado por épocas suficientes
- Experimente diferentes valores de `temperature`
- Tente usar `priming` com uma música de referência

## Estrutura de Checkpoints

Os checkpoints salvos contêm:
- Estado do modelo (`model_state_dict`)
- Estado do optimizer (`optimizer_state_dict`)
- Época atual (`epoch`)
- Loss atual (`loss`)
- Configurações (`config`)
- Vocabulário (`vocab`)

## Desenvolvimento

### Módulos Principais

- **data_processor.py**: `MIDIProcessor` e `MIDITokenizer`
- **model.py**: `MultiInstrumentTransformer`
- **music_utils.py**: Funções de conversão e análise
- **train.py**: Loop de treinamento completo
- **generate.py**: Geração autoregressiva

## Licença

Este projeto é parte de um trabalho de TCC (Trabalho de Conclusão de Curso).

## Referências

- Arquitetura Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- Tokenização REMI: "Pop Music Transformer" (Huang & Yang, 2020)
- Processamento MIDI: Biblioteca `mido` e `pretty_midi`

## Contato

Para dúvidas ou problemas, consulte a documentação do código ou entre em contato com o autor do projeto.

