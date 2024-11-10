import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

DISPLAY_MODE = True

def set_display_mode(val):
    global DISPLAY_MODE
    DISPLAY_MODE = val
    
    # set the batch_mode matplotlib backend to save memory
    if not DISPLAY_MODE :
        matplotlib.use('agg')
    return DISPLAY_MODE

def plot_attention_map(attentions, tokens, layer_idx = 0, head_idx = 0, batch_idx = 0, save_path=None):
    global DISPLAY_MODE
    attention_map = attentions[layer_idx][head_idx][batch_idx].detach().cpu().numpy()
    single_head_attention_map = attention_map[head_idx]  # Now shape is (5, 5)
    if DISPLAY_MODE:
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
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close('all')

def visualize_fft(hidden_states, in_layer, out_layer, device, head_idx=0, max_points=512, save_path=None):
    """
    Visualizes the Fourier Transform between the input and output of a layer (or layers).
    """
    global DISPLAY_MODE

    if device != "cpu" :
        hidden_state_in = hidden_states[in_layer][head_idx].cpu().numpy()
        hidden_state_out = hidden_states[out_layer][head_idx].cpu().numpy()
    else:
        hidden_state_in = hidden_states[in_layer]
        hidden_state_out = hidden_states[out_layer]


    # Apply FFT across the sequence dimension (axis=0)
    fft_in = np.fft.fft(hidden_state_in, axis=0)
    fft_out = np.fft.fft(hidden_state_out, axis=0)

    # Calculate the magnitudes for each frequency component
    magnitude_in = (np.abs(fft_in).mean(axis=1))[0]
    magnitude_out = (np.abs(fft_out).mean(axis=1))[0]  # Average across hidden_dim

    if DISPLAY_MODE:
        plt.figure(figsize=(12, 6))
        plt.plot(magnitude_in, label=f'Layer {in_layer}')
        plt.plot(magnitude_out, label=f'Layer {out_layer}')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title(f'Fourier Transform of Hidden States - Layer {in_layer} vs Layer {out_layer}')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
