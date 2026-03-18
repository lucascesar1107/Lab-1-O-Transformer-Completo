import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. FUNÇÕES BASE

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_model]
    mask:    [batch, q_len, k_len] ou [q_len, k_len]
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # [1, q_len, k_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)



# 2. CAMADA DE EMBEDDING + POSITIONAL ENCODING


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]



# 3. BLOCOS DO ENCODER


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x):
        # Self-Attention
        attn_output, _ = scaled_dot_product_attention(x, x, x, mask=None)
        x = self.addnorm1(x, attn_output)

        # FFN
        ffn_output = self.ffn(x)
        x = self.addnorm2(x, ffn_output)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x)

        return x  # memória Z



# 4. BLOCOS DO DECODER


def generate_causal_mask(seq_len, device):
    # triangular inferior = 1, futuro = 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, y, Z):
        # 1) Masked Self-Attention
        seq_len = y.size(1)
        causal_mask = generate_causal_mask(seq_len, y.device)
        masked_attn_output, _ = scaled_dot_product_attention(y, y, y, mask=causal_mask)
        y = self.addnorm1(y, masked_attn_output)

        # 2) Cross-Attention (Q vem de y, K e V vêm de Z)
        cross_attn_output, _ = scaled_dot_product_attention(y, Z, Z, mask=None)
        y = self.addnorm2(y, cross_attn_output)

        # 3) FFN
        ffn_output = self.ffn(y)
        y = self.addnorm3(y, ffn_output)

        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, y, Z):
        y = self.embedding(y)
        y = self.positional_encoding(y)

        for layer in self.layers:
            y = layer(y, Z)

        logits = self.output_linear(y)
        probs = F.softmax(logits, dim=-1)
        return probs



# 5. TRANSFORMER COMPLETO


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=32, d_ff=64, num_layers=2, max_len=100):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_ff, num_layers, max_len)

    def forward(self, encoder_input, decoder_input):
        Z = self.encoder(encoder_input)
        probs = self.decoder(decoder_input, Z)
        return probs



# 6. TESTE DE INFERÊNCIA AUTO-REGRESSIVA


def greedy_decode(model, encoder_input, start_token_id, eos_token_id, max_steps=10):
    model.eval()

    with torch.no_grad():
        Z = model.encoder(encoder_input)

        decoder_input = torch.tensor([[start_token_id]], dtype=torch.long, device=encoder_input.device)

        generated_tokens = [start_token_id]

        step = 0
        while step < max_steps:
            probs = model.decoder(decoder_input, Z)  # [1, seq_len, vocab_size]
            next_token = torch.argmax(probs[:, -1, :], dim=-1).item()

            generated_tokens.append(next_token)

            if next_token == eos_token_id:
                break

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=encoder_input.device)
            decoder_input = torch.cat([decoder_input, next_token_tensor], dim=1)
            step += 1

        return generated_tokens



# 7. EXEMPLO COMPLETO COM FRASE TOY


if __name__ == "__main__":
    # Vocabulário fictício
    vocab = {
        "<PAD>": 0,
        "<START>": 1,
        "<EOS>": 2,
        "Thinking": 3,
        "Machines": 4,
        "Máquinas": 5,
        "Pensantes": 6
    }

    id_to_token = {idx: token for token, idx in vocab.items()}

    src_vocab_size = len(vocab)
    tgt_vocab_size = len(vocab)

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=32,
        d_ff=64,
        num_layers=2,
        max_len=20
    )

    # Frase simulando "Thinking Machines"
    encoder_input = torch.tensor([[vocab["Thinking"], vocab["Machines"]]], dtype=torch.long)

    generated_ids = greedy_decode(
        model=model,
        encoder_input=encoder_input,
        start_token_id=vocab["<START>"],
        eos_token_id=vocab["<EOS>"],
        max_steps=8
    )

    generated_tokens = [id_to_token[token_id] for token_id in generated_ids]

    print("Entrada do Encoder:", ["Thinking", "Machines"])
    print("Saída gerada pelo Decoder:", generated_tokens)