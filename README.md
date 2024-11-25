# franken-Llama

<p align="center">
  <img src="res//image.png" alt="Frankenstein Llama" width="400px"/>
</p>

> _"Stitching together a monstrous, savage and efficient LLM!"_

Exploring the modification and enhancement of Llama transformer models by selectively removing and reattaching attention blocks. This process involves altering the existing Llama codebase (modeling_llama module in particular) and analyzing the resulting attention maps.

25 configs in total

Tasks:
- [x] Visualize attention maps for target layers (modified llama)
- [ ] Write a short analysis on the attention maps
- [x] Save data to csv
- [x] Add a fourier transform
  - [x] Update fourier transform to work on attention maps
- [x] Add NQ dataset support

- [ ] complete slides
- [x] complete MAC computations

- [ ] test some possible extension with kV cache compression (optional)
- [ ] improve README.md

### Examples
Examples here