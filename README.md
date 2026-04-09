# Zero-shot in-silico perturbation exploration

In this repository, we explore zero-shot perturbation to infer drugs efficacy based on the EVA-RNA foundation model 🧬.

## Getting started

If not already done, install `uv` (as shown [here](https://docs.astral.sh/uv/getting-started/installation/)), and then the following command at the root of the repository:

```sh
uv sync --dev
```

## Methods

### 1️⃣ Encoder-based backpropagation

As a first approach, I wanted to try a straightforward approach that doesn't even require a decoder:
1. In the latent space, compute the euclidean distance between the centroid of the healthy samples and the diseased samples.
2. Using this distance, we define a loss as the square of this distance.
3. Then, we backpropagate this loss through diseased samples.
4. For each input gene, a positive gradient indicates that an up-regulation of this gene would lead to a higher distance to healthy samples. Or, in the other direction: a down-regulation would lead to a smaller distance to healthy samples (which is what we want).
5. We use the mean gradient per gene as a measure of efficacy - under the assumption that all drugs from the benchmark have a down-regulating impact.

This approach is appealing for multiple reasons:
1. It's fast to compute, as we get a score for every gene in one single backpropagation.
2. It doesn't need a decoder, which enable its usage on models that doesn't have a decoder (e.g., contrastive-based or self-distillation models).

The code logic can be found [here](https://github.com/quentinblampey/zs_perturbation/blob/main/zs_perturbation/encoder_based.py), and results can be computed for all diseases via the following command line:
```sh
uv run python main.py --method encoder
```

Per-disease results are stored in `./results/encoder/<DISEASE>.csv`. Unfortunately, these results are not convincing, hence I also tried the method proposed in the paper (see the next section). Weirdly, using the opposite gradient would lead to a mean AUCROC of 0.62, but I double-checked the sign and believe it should be this way and not the other way around...

### 2️⃣ Decoder-based backpropagation

As a second approach, I wanted to try one approach from the paper. I tried the "intermediate layer" approach, using the second to last layer.

The code logic can be found [here](https://github.com/quentinblampey/zs_perturbation/blob/main/zs_perturbation/decoder_based.py), and results can be computed for all diseases via the following command line:
```sh
uv run python main.py --method decoder
```

Per-disease results are stored in `./results/decoder/<DISEASE>.csv`.

This time, results are much better - it achieves a mean AUCROC of **0.62**, with **three diseases above 0.7**. See the [results plots here](https://github.com/quentinblampey/zs_perturbation/blob/main/results/plots.ipynb).

### 3️⃣ Classification head and counter factual explanation

Another idea would be to train a small classification head on top of the encoder, to predict whether a sample is healthy or not. Based on this classification head, we could use counter-factual approaches to find the smallest perturbation which would make the classication switch from a diseased sample to a healthy sample (i.e., model output switching from 0 to 1).

The rationale behind this is that a classification head would learn more specifically the patterns behind what defines a diseased sample, instead of learning all gene regulatory networks (including mechanisms that are not relevant to discriminate samples).

This idea has not been implemented in this quick exercice to focus on the first two approaches and also to avoid a setup on a HPC to train the classification head.

## Known limitations

For this quick exercice, I made some decisions that could be improved, and made multiple assumptions that might be wrong:
- I assumed all targets where down-regulating the corresponding genes. It seems to be the case, but I would need to go deeper into the mechanisms of action of each drug to better model it.
- For drugs targeting multiple genes, I averaged the score of the genes to get a single scalar for the drug. Instead, I should have modeled the whole drug directly, e.g. bu suming the loss terms into one before backpropagating.
- I'm more familiar with single-cell and spatial data, therefore I'm not sure the preprocessing I did makes sense for RNAseq data. This may significantly change the results.
- Further exploration of the model internal states could help developing new ideas.