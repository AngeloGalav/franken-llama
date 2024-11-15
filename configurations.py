# example configurations (check docs for explanations)
configurations = [
    # baseline
    {"layers_to_repeat": [], "layers_to_skip": [], "name": "baseline", "repeats": 1},
    # Front-Focused Reduction
    {"layers_to_repeat": [], "layers_to_skip": list(range(8, 32)), "name": "0-7", "repeats": 1},
    # Top-Heavy Emphasis
    {"layers_to_repeat": list(range(23, 32)), "layers_to_skip": list(range(0, 23)), "name": "23-31r3", "repeats": 3},
    # Even-Spread Reduction
    {"layers_to_repeat": [], "layers_to_skip": [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30], "name": "0_7_15_23_31", "repeats": 1},
    # Middle-Focused Processing
    {"layers_to_repeat": [], "layers_to_skip": list(range(0, 11)) + list(range(20, 32)), "name": "11-19", "repeats": 1},
    # Recursive Narrowing
    {"layers_to_repeat": [15, 23, 31], "layers_to_skip": [i for i in range(32) if i not in [15, 23, 31]], "name": "15r3_23r3_31r3", "repeats": 3},
    # First and Last Only
    {"layers_to_repeat": [], "layers_to_skip": list(range(4, 28)), "name": "0-3_28-31", "repeats": 1},
    # Shallow with High Repetition
    {"layers_to_repeat": list(range(0, 8)), "layers_to_skip": list(range(8, 32)), "name": "0-7r3", "repeats": 3},
    # Middle Layers Only with Repetitions
    {"layers_to_repeat": list(range(9, 20)), "layers_to_skip": list(range(0, 9)) + list(range(20, 32)), "name": "9-19r2", "repeats": 2},
    # End Layers Compression
    {"layers_to_repeat": list(range(27, 32)), "layers_to_skip": list(range(0, 27)), "name": "27-31r5", "repeats": 5},
    # Layer Skip with Alternating Repeats
    {"layers_to_repeat": [0, 7, 15, 23, 31], "layers_to_skip": [i for i in range(32) if i not in [0, 7, 15, 23, 31]], "name": "0_7r2_15r2_23r2_31r2", "repeats": 2},
    # Low-efficiency/High-precision skip
    {"layers_to_repeat": [], "layers_to_skip": [24,25,26], "name": "0-23_27-31", "repeats": 1},
    # Single layer skip (very high accuracy)
    {"layers_to_repeat": [], "layers_to_skip": [15], "name": "15_single_skip", "repeats": 1},
    # do all layers aside from last 2 (which are the beefiest)
    {"layers_to_repeat": [], "layers_to_skip": list(range(0, 30)), "name": "all_except_last_two", "repeats": 1},
    # do only even layers
    {"layers_to_repeat": [], "layers_to_skip": [i for i in range(32) if i % 2 != 0], "name": "only_even_layers", "repeats": 1},
    # do only odd layers
    {"layers_to_repeat": [], "layers_to_skip": [i for i in range(32) if i % 2 == 0], "name": "only_odd_layers", "repeats": 1},
    # repeat first 8 and last 8 twice
    {"layers_to_repeat": list(range(0, 8)) + list(range(24, 32)), "layers_to_skip": list(range(8, 24)), "name": "first_last_8r2", "repeats": 2},
    # repeat first and last layers, skip in the middle
    {"layers_to_repeat": [0, 31], "layers_to_skip": [15, 16, 24, 25], "name": "first_last_2_with_skips", "repeats": 2},
    # middle expansion with repeats
    {"layers_to_repeat": list(range(14, 19)), "layers_to_skip": [6, 7, 8, 9, 25, 26, 27, 28], "name": "mid_expansion_with_repeats", "repeats": 2},
    # 2/3rd of Llama
    {"layers_to_repeat": [], "layers_to_skip": list(range(11, 21)), "name": "2_3rd_of_llama", "repeats": 1},
    # a bit more than 2/3rds
    {"layers_to_repeat": [], "layers_to_skip": list(range(11, 21, 2)), "name": "2_3rds_plus_llama", "repeats": 1},
    # skip every 2 layers
    {"layers_to_repeat": [], "layers_to_skip": [i for i in range(32) if i % 3 != 0], "name": "skip_every_2_layers", "repeats": 1},
    # skip every 2 layers, repeat each layer
    {"layers_to_repeat": [], "layers_to_skip": [i for i in range(32) if i % 3 != 0], "name": "skip_every_2_layers_r2", "repeats": 2},
    # skip some layers at the end
    {"layers_to_repeat": [], "layers_to_skip": [27, 28, 29], "name": "skip_near_end_keep_last_two", "repeats": 1},
    # keep essential layers, skip middle, repeat end
    {"layers_to_repeat": [30, 31], "layers_to_skip": list(range(10, 26)), "name": "essential_layers_with_final_repeats", "repeats": 2}

]
