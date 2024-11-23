# Llama2 specifications

### Llama2 model structure
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((4096,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((4096,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)
```

### Actual values
- context window size: 4096
- hidden size: 4096 (size of each hidden state)
- number of hidden state outputted: 33 (1 + 32, since it includes the input)
- number of layers of transformer decoder: 32
- vocab_size: 32000
- intermediate size: 11008 (size of vector btwn feature extractor (decode layer) and the linear layer)
- attention heads: 32


These values refer to the Llama2-7B model.

### MAC computation
1) Embedding: $N×4096$
2) Attention operations:
    - q/k/v projections: $3×2×N×4096×4096$.
        - 2 b.c. is Multiply AND ACC, 3 b.c. is one for each proj.
        - at this step, for each token, the Q/K/V are created, which are 3 different representations.
            - Q -> what the token is looking for
            - K -> how relevant is this token to a query
            - V -> the information carried by a token
    - attention score computations in the dot product attention:
        - $2×N^2×4096$.
        - softmax application is ignored b.c. it is not significant.
    - attention output projection (mapping of concatenated attention heads back to size 4096):
        - $2×N×4096×4096$
3) MLP Layer:
    - gate (controlled activation) + up scale proj: $2×2×N×4096×11008$
    - downscale proj: $2×N×11008×4096$
4) Layer Normalization (often ignored): $N×4096$
5) Output Layer - projecting to vocab size: $2×N×4096×32000$


TOTAL:
1) emb: $N×4096$
2) 32\*att: $32×(4×2×N×4096×4096+2×N^2×4096)$
3) 32\*MLP: $32×(3×N×(4096×11008))$
4) out: $2×N×4096×32000$


### MACs per config
The MACs for the best configs are:
- baseline: $8,882,726,776×N+262,144×N^2$
- 0-23_27-32: $8,089,532,288×N+237,568×N^2$
- 15_single_skip: $8,595,077,304×N+253,952×N^2$
- mid_expansion_with_repeats (29): $8,089,532,288×N+237,568×N^2$
- 2/3rds of llama (22): $6,185,271,872×N+180,224×N^2$
- 2/3rds of llama plus (27): $7,534,650,056×N+221,184×N^2$
- skip_near_end_keep_last_two (29): $8,089,532,288×N+237,568×N^2$