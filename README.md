# Zero-shot in-silico perturbation exploration

In this repository, we explore zero-shot perturbation to infer drugs efficacy based on a RNA-based foundation model.

## Getting started

If not already done, install `uv` (as shown [here](https://docs.astral.sh/uv/getting-started/installation/)), and then the following command at the root of the repository:

```sh
uv sync --dev
```

## Methods

### 1. Encoder-based backpropagation

As a first approach, I wanted to try a straightforward approach that doesn't even require a decoder:
1. In the latent space, compute the euclidean distance between the centroid of the healthy samples and the diseased samples.
2. Using this distance, we define a loss as the square of this distance.
3. Then, we backpropagate this loss through diseased samples.
4. For each input gene, a positive gradient indicates that an up-regulation of this gene would lead to a higher distance to healthy samples. Or, in the other direction: a down-regulation would lead to a smaller distance to healthy samples (which is what we want).
5. We use the mean gradient per gene as a measure of efficacy - under the assumption that all drugs from the benchmark have a down-regulating impact.

This approach is appealing for multiple reasons:
1. It's fast to compute, as we get a score for every gene in one single backpropagation.
2. It doesn't need a decoder, which enable its usage on models that doesn't have a decoder (e.g., contrastive-based or self-distillation models).

Results can be computed for all diseases via the following command line:
```sh
uv run python main.py
```


### 2. Decoder-based backpropagation
