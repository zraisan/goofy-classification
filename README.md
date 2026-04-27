# gopher

A gopher identifier. Because somebody had to.

This is a learning project, shipped in stages. Each stage is a more complicated and detailed neural network than the last, starting from one neuron and ending in a CNN. Once a NumPy-equivalent library lands for Go, the whole thing gets rewritten in Go. A gopher, in gopher.

## Stages

### Stage 1: logistic regression in NumPy *(current)*
One neuron. No frameworks. Flatten the pixels, sigmoid the dot product, cry at the accuracy.

- `main.ipynb`: the whole thing (load, train, predict).
- `build_dataset.py`: pulls ~2000 real images from the iNaturalist API (real pocket gophers vs. other rodents), resizes to 32x32, writes `data/gopher_dataset.npz`.

### Stage 2: deeper nets, still in NumPy
Add hidden layers. Backprop by hand. Then a CNN, still by hand, because that's the whole point.

### Stage 3: port to Go
Once the numpy guy ships the library, the whole thing gets rewritten in Go. A gopher written in gopher. This was always the joke.

## Dataset

- **Positives:** family *Geomyidae* (real pocket gophers) from [iNaturalist](https://www.inaturalist.org/taxa/44027-Geomyidae).
- **Negatives:** order *Rodentia* minus *Geomyidae*, so squirrels, mice, chipmunks, voles, rats. Similar-looking small mammals, so the task actually demands gopher-specific features.
- 1000 per class, 80/20 train/test split, 32x32x3 uint8.
- Photos are CC-BY-NC (iNaturalist contributors).

Rebuild any time:
```bash
uv run python build_dataset.py
```

## Run

```bash
uv sync
uv run jupyter lab main.ipynb
```
