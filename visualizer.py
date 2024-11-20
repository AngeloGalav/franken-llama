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

# TODO: add support for old fourier time domain
def visualize_fft(hidden_states, in_layer, out_layer, device, head_idx=0, max_points=512, attention=False, batch_idx=0, spatial_domain=True, save_path=None):
    """
    Visualizes the Fourier Transform between the input and output of a layer (or layers).
        - attetion: use attention map as input
    """
    global DISPLAY_MODE

    if (attention):
        if device != "cpu" :
            # batch_0 for now
            hidden_state_in = hidden_states[in_layer][head_idx][batch_idx].detach().cpu().numpy()
            hidden_state_out = hidden_states[out_layer][head_idx][batch_idx].detach().cpu().numpy()
        else:
            hidden_state_in = hidden_states[in_layer][head_idx]
            hidden_state_out = hidden_states[out_layer][head_idx]
    else:
        if device != "cpu" :
            hidden_state_in = hidden_states[in_layer][head_idx].cpu().numpy()
            hidden_state_out = hidden_states[out_layer][head_idx].cpu().numpy()
        else:
            hidden_state_in = hidden_states[in_layer][head_idx]
            hidden_state_out = hidden_states[out_layer][head_idx]


    # Apply FFT across the sequence dimension (axis=0)
    fft_in = np.fft.fft(hidden_state_in, axis=0)
    fft_out = np.fft.fft(hidden_state_out, axis=0)

    # Calculate the magnitudes for each frequency component
    mean_magnitude_in = (np.abs(fft_in).mean(axis=1))[0]
    mean_magnitude_out = (np.abs(fft_out).mean(axis=1))[0]  # Average across hidden_dim

    # Frequency domain (normalized frequencies)
    freqs_in = np.fft.fftfreq(hidden_state_in.shape[2])
    freqs_out = np.fft.fftfreq(hidden_state_out.shape[2])

    plt.figure(figsize=(12, 6))

    if DISPLAY_MODE:
        # input FFT Magnitude
        plt.subplot(1, 2, 1)
        plt.plot(freqs_in, mean_magnitude_in, marker='o')
        plt.title(f"First Layer Attention Map - FFT Magnitude")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid()

        # output FFT Magnitude
        plt.subplot(1, 2, 2)
        plt.plot(freqs_out, mean_magnitude_out, marker='o')
        plt.title(f"Last Layer Attention Map - FFT Magnitude")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
