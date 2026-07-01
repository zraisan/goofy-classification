# gopher

A gopher identifier. Because somebody had to.

This is a learning project, shipped as one notebook per stage. Each stage is a more complicated and detailed neural network than the last, starting from one neuron and ending in a CNN. Once a NumPy-equivalent library lands for Go, the whole thing gets rewritten in Go. A gopher, in gopher.

## Stages

### Stage 1: logistic regression in NumPy

`01_logistic_regression.ipynb`

A single sigmoid neuron trained with binary cross-entropy and vanilla gradient descent on flattened image pixels.

### Stage 2: deep neural network in NumPy

`02_deep_neural_network.ipynb`

A fully connected network with hidden layers, ReLU activations, sigmoid output, cached activations, and backprop implemented by hand in NumPy.

### Stage 3: optimization in NumPy _(next)_

Improve training with optimization-focused changes such as better initialization, learning-rate experiments, and optimizer updates.

### Later: CNN in NumPy

Add convolutional layers and train a small CNN by hand, still without deep learning frameworks.

### Later: port to Go

Once the numpy guy ships the library, the whole thing gets rewritten in Go. A gopher written in gopher. This was always the joke.

`main.ipynb` is left as a working notebook for the current or next stage. The numbered notebooks are the educational sequence.

## Dataset

- **Positives:** family _Geomyidae_ (real pocket gophers) from [iNaturalist](https://www.inaturalist.org/taxa/44027-Geomyidae).
- **Negatives:** order _Rodentia_ minus _Geomyidae_, so squirrels, mice, chipmunks, voles, rats. Similar-looking small mammals, so the task actually demands gopher-specific features.
- 1000 per class, 80/20 train/test split.
- Photos are CC-BY-NC (iNaturalist contributors).
- The notebooks expect `data/gopher_dataset.npz` with `X_train`, `y_train`, `X_test`, and `y_test`.

## Run

```bash
uv sync
uv run jupyter lab 01_logistic_regression.ipynb
uv run jupyter lab 02_deep_neural_network.ipynb
```
