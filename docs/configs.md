# franken-Llama configurations
The configuration of layers by which franken-Llama was tested on are only 11, since testing all possible 4294967296 configuration would be quite the feat (even more if you consider that you can repeat layers).

Instead, I decided to test franken-Llama with a select number of configs, and each of them has a motivation for it.


1. **Front-Focused Reduction**
   - **Keep Layers:** 1–8
   - **Skip Layers:** 9–32
   - **Explanation:** This setup retains only the initial layers. Early layers focus on detecting syntactic and simpler semantic features, which can be effective for lightweight models that don’t require deep contextualization.
   - **Use Case:** Ideal for simple sentence-level tasks or fast autocomplete with minimal depth.

2. **Top-Heavy Emphasis**
   - **Repeat Layers:** 24–32 (repeat each layer 2–3 times)
   - **Skip Layers:** 1–23
   - **Explanation:** This configuration focuses on the last layers, which typically capture complex, high-level abstractions. By repeating the final layers, the model could simulate depth without processing all preceding layers.
   - **Use Case:** Great for complex, contextualized predictions where the initial processing can be “skipped” in favor of deeper abstractions.

3. **Even-Spread Reduction**
   - **Keep Layers:** 1, 8, 16, 24, 32
   - **Skip Layers:** All others
   - **Explanation:** This sparsified configuration keeps layers evenly spaced throughout the model, capturing information from different depths and creating a compact but effective representation.
   - **Use Case:** Suitable for tasks that require a balance between fast execution and contextual comprehension across multiple layers.

4. **Middle-Focused Processing**
   - **Keep Layers:** 12–20
   - **Skip Layers:** 1–11, 21–32
   - **Explanation:** Focusing on the middle layers allows the model to catch intermediary representations without needing the full stack of layers, essentially compressing the model into its central processing.
   - **Use Case:** Effective for scenarios where intermediate abstractions are enough, such as summarization and medium-length text completion.

5. **Recursive Narrowing**
   - **Repeat Layers:** 16, 24, and 32 (repeat each 2–4 times)
   - **Skip Layers:** All others
   - **Explanation:** This configuration skips most of the model, focusing on key layers at the start, middle, and end. Repeating each layer multiple times introduces more depth where it's most impactful.
   - **Use Case:** Useful for generating concise outputs by repeating layers that refine rather than building up from scratch.

6. **First and Last Only**
   - **Keep Layers:** 1–4, 29–32
   - **Skip Layers:** 5–28
   - **Explanation:** This extreme reduction keeps only the initial and final layers, bypassing much of the internal structure but keeping beginning feature detection and final synthesis.
   - **Use Case:** Ideal for applications that only need lightweight processing with end-stage refinement, like brief summaries or keyword extraction.

7. **Shallow with High Repetition**
   - **Repeat Layers:** 1–8 (repeat each 3 times)
   - **Skip Layers:** 9–32
   - **Explanation:** By repeating just the initial layers, this configuration keeps the model shallow but with more repetitions on the syntactic and basic semantic features.
   - **Use Case:** Good for shallow tasks such as grammatical correction, token classification, or simple text expansion.

8. **Middle Layers Only with Repetitions**
   - **Repeat Layers:** 10–20 (repeat each layer 2 times)
   - **Skip Layers:** 1–9, 21–32
   - **Explanation:** This configuration captures and deepens the intermediate layers, which often represent a mix of surface-level and deeper contextual features.
   - **Use Case:** Effective for dialogue generation or multi-turn conversation, where mid-range context is crucial but deep abstractions aren’t needed.

9. **End Layers Compression**
   - **Repeat Layers:** 28–32 (repeat each 4–5 times)
   - **Skip Layers:** 1–27
   - **Explanation:** An aggressive approach that skips almost all layers and repeats the final layers heavily to maximize the high-level comprehension of the model.
   - **Use Case:** Useful for dense context generation, particularly where only the concluding synthesis of ideas is essential, such as creative writing prompts or summary generation.

10. **Layer Skip with Alternating Repeats**
   - **Keep Layers:** 1, 8, 16, 24, 32 (repeat each of these layers 2 times)
   - **Skip Layers:** All others
   - **Explanation:** By repeating layers spread out across the model, you ensure each layer captures broad representations of data with some degree of “depth” at each level.
   - **Use Case:** Versatile for general-purpose language generation with good contextual depth-to-speed ratio.


(Admittedly these configuration were suggested by ChatGPT (oops) but honestly they make a lot of sense).