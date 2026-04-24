"""
Arquitetura do modelo Transformer Multi-Instrumental.
Implementa um modelo baseado em Transformer para geração de música multi-instrumental.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Encoding posicional para sequências de tokens.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializa o encoding posicional.
        
        Args:
            d_model: Dimensão do modelo
            max_len: Comprimento máximo da sequência
            dropout: Taxa de dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Cria matriz de encoding posicional
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Registra como buffer (não é parâmetro)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica encoding posicional.
        
        Args:
            x: Tensor de shape (seq_len, batch_size, d_model)
            
        Retorna:
            Tensor com encoding posicional adicionado
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiInstrumentTransformer(nn.Module):
    """
    Decoder-only Transformer para geração autoregressiva de música multi-instrumental.
    Usa nn.TransformerEncoder do PyTorch com causal mask — equivalente a um GPT-style decoder.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_seq_length: int = 2048,
                 num_instruments: int = 5):
        """
        Inicializa o modelo Transformer.
        
        Args:
            vocab_size: Tamanho do vocabulário
            d_model: Dimensão do modelo (deve ser divisível por nhead)
            nhead: Número de heads de atenção
            num_layers: Número de camadas do encoder
            dim_feedforward: Dimensão da camada feedforward
            dropout: Taxa de dropout
            max_seq_length: Comprimento máximo da sequência
            num_instruments: Número de instrumentos
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding de tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Encoding posicional
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer (decoder-only via causal mask — PyTorch não tem TransformerDecoder standalone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # Usamos (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Head de saída para geração de tokens
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Weight tying: output_projection compartilha pesos com token_embedding.
        # Força coerência entre representar e prever tokens; melhora treinamento.
        self.output_projection.weight = self.token_embedding.weight

        # Dropout adicional
        self.dropout = nn.Dropout(dropout)

        # Inicialização dos pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa os pesos do modelo."""
        initrange = 0.1
        # token_embedding e output_projection compartilham o mesmo tensor (weight tying)
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            src: Tensor de input de shape (batch_size, seq_len)
            src_mask: Mask de atenção para evitar olhar para o futuro (opcional)
            src_key_padding_mask: Mask para padding tokens (opcional)
            
        Retorna:
            Tensor de logits de shape (batch_size, seq_len, vocab_size)
        """
        # src: (batch_size, seq_len)
        batch_size, seq_len = src.size()
        
        # Embedding + pos encoding
        # Transpõe para (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)  # (seq_len, batch_size)
        src = self.token_embedding(src) * math.sqrt(self.d_model)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)  # (seq_len, batch_size, d_model)
        
        # Aplica transformer encoder
        # src_mask deve ser de shape (seq_len, seq_len) para causal masking
        if src_mask is None:
            # Cria mask causal padrão
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Transformer encoder espera (seq_len, batch_size, d_model)
        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (seq_len, batch_size, d_model)
        
        # Aplica dropout
        output = self.dropout(output)
        
        # Projeta para vocabulário
        output = self.output_projection(output)  # (seq_len, batch_size, vocab_size)
        
        # Transpõe de volta para (batch_size, seq_len, vocab_size)
        output = output.transpose(0, 1)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Gera mask causal para evitar que o modelo olhe para tokens futuros.
        
        Args:
            sz: Tamanho da sequência
            
        Retorna:
            Mask de shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 1024,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                 eos_token_id: int = 2, context_size: int = 512,
                 note_mask: torch.Tensor = None,
                 vocab_constraint_fn=None,
                 temperature_fn=None) -> torch.Tensor:
        """
        Gera uma sequência de tokens autoregressivamente.

        Args:
            input_ids: Tokens iniciais de shape (batch_size, seq_len)
            max_length: Comprimento máximo da geração
            temperature: Temperatura fixa para sampling (ignorada se temperature_fn for fornecida)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: ID do token de fim de sequência
            context_size: Tamanho máximo da janela de contexto (deve ser igual ao seq_length do treino)
            vocab_constraint_fn: Função opcional (last_token_id: int) -> Tensor de máscara de logits.
                                 Usada para restringir vocabulário com base no token anterior.
            temperature_fn: Função opcional (step: int) -> float para temperatura dinâmica.
                            Se fornecida, sobrescreve o parâmetro temperature.

        Retorna:
            Sequência gerada de shape (batch_size, generated_length)
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        generated = input_ids.clone()

        with torch.no_grad():
            for step in range(max_length - input_ids.size(1)):
                # Trunca contexto para a janela de treino — evita degradação fora da distribuição
                context = generated[:, -context_size:]
                # Forward pass
                logits = self.forward(context)  # (batch_size, context_len, vocab_size)

                # Temperatura dinâmica: permite rampa crescente durante a geração
                temp = temperature_fn(step) if temperature_fn is not None else temperature

                # Pega logits do último token
                next_token_logits = logits[:, -1, :] / temp  # (batch_size, vocab_size)

                # Aplica máscara de escala musical (notas fora da escala → -inf)
                if note_mask is not None:
                    next_token_logits = next_token_logits + note_mask.unsqueeze(0)

                # Aplica restrição de vocabulário baseada no último token gerado
                if vocab_constraint_fn is not None:
                    last_token_id = generated[0, -1].item()
                    constraint = vocab_constraint_fn(last_token_id)
                    next_token_logits = next_token_logits + constraint.unsqueeze(0)

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens com probabilidade cumulativa acima do threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Concatena token gerado
                generated = torch.cat([generated, next_token], dim=1)
                
                # Para se todos os tokens são EOS
                if (next_token == eos_token_id).all():
                    break
        
        return generated

