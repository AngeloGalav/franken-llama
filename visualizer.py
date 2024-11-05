import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_map(attentions, tokens, layer_idx = 0, head_idx = 0, batch_idx = 0):
    attention_map = attentions[layer_idx][head_idx][batch_idx].detach().cpu().numpy()
    single_head_attention_map = attention_map[head_idx]  # Now shape is (5, 5)

    plt.figure(figsize=(10, 10))

    # Since Llama uses CAUSAL attention, the attention maps get bigger and bigger
    # each layer (since it is generating a string until it reaches max_len seq length)

    # Since it is masked, the attention map from layer2 onwards is computed only for the FINAL token,
    # this is done in order to have higher efficiency.

    if (layer_idx > 0) :
        sns.heatmap(single_head_attention_map, cmap="viridis", cbar=True, square=True)
        plt.ylabel("Final Seq. Token")
        plt.xlabel('Hidden states/"Tokens"')
    else:
        sns.heatmap(single_head_attention_map, cmap="viridis", cbar=True, square=True,
                    xticklabels=tokens, yticklabels=tokens)
        plt.xlabel("Tokens")
        plt.ylabel("Tokens")

    plt.title(f"Attention Heatmap - Head {head_idx+1} - Layer {layer_idx+1}")
    plt.show()

def visualize_feature_map():
    ...