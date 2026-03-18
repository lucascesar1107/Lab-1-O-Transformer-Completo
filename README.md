# Lab P1-04 – Implementação do Transformer Completo (Encoder-Decoder)

Este projeto implementa a arquitetura completa de um Transformer (Encoder-Decoder), conforme descrito no artigo científico *Attention Is All You Need*.

A implementação integra os principais componentes do modelo, incluindo Encoder, Decoder, mecanismo de atenção, máscara causal e um loop de geração auto-regressiva de tokens.

A biblioteca utilizada para a construção do modelo foi o PyTorch.

---

## Requisitos

- Python 3.x  
- PyTorch  

Instalação do PyTorch:

pip install torch

---

## Execução

Para executar o exemplo de teste, utilize:

python main.py

O script executa:

- criação de um vocabulário fictício  
- simulação de entrada no Encoder  
- geração de sequência pelo Decoder  

---

## Arquitetura

O modelo segue a estrutura Encoder-Decoder.

### Encoder

- embedding + positional encoding  
- self-attention  
- add & norm  
- feed-forward network  
- add & norm  

### Decoder

- masked self-attention (com máscara causal)  
- add & norm  
- cross-attention com a saída do encoder  
- add & norm  
- feed-forward network  
- add & norm  
- camada linear + softmax  

---

## Máscara Causal

No decoder, é utilizada uma máscara triangular inferior para impedir que o modelo acesse tokens futuros durante a geração.

Essa abordagem garante que cada predição seja feita apenas com base nos tokens já gerados.

---

## Inferência Auto-Regressiva

A geração de sequência ocorre de forma iterativa:

1. o decoder inicia com o token `<START>`  
2. a cada passo, o modelo prevê o próximo token  
3. o token é concatenado à entrada  
4. o processo continua até gerar `<EOS>` ou atingir o limite de passos  

---

## Exemplo

Entrada do encoder:

```
["Thinking", "Machines"]
```

Saída gerada pelo decoder (exemplo):

```
["<START>", "Máquinas", "Pensantes", "<EOS>"]
```

Os resultados podem variar, pois o modelo não foi treinado.
