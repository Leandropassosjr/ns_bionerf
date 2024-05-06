# BioNeRF: Biologically Plausible Neural Radiance Fields
This is the [NeRFStudio](https://docs.nerf.studio/)-based implementation for [BioNeRF](https://github.com/Leandropassosjr/BioNeRF).
![](https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/gifs.gif)


# Installation
BioNeRF follows the integration guidelines described [here](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html) for custom methods within Nerfstudio. 
### 0. Install Nerfstudio dependencies
[Follow these instructions](https://docs.nerf.studio/en/latest/quickstart/installation.html) up to and including "tinycudann" to install dependencies and create an environment
### 1. Clone this repo
`git clone https://github.com/Leandropassosjr/nf_bionerf`
### 2. Install this repo as a python package
Navigate to this folder and run `python -m pip install -e .`

### 3. Run `ns-install-cli`

### Checking the install
Run `ns-train -h`: you should see a list of "subcommands" with bionerf included among them.

# Using BioNeRF
Now that BioNeRF is installed you can play with it! 

- Launch training with `ns-train bionerf --data <data_folder>`. This specifies a data folder to use. For more details, see [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). 
- Connect to the wand by forwarding the wand port (we use VSCode to do this), and click the link provided in the output of the train script.
- Within the wand, you can check the training progress and several metrics.



# Method

## Overview

[BioNeRF](https://arxiv.org/pdf/2402.07310.pdf) (Biologically Plausible Neural Radiance Fields) extends [NeRF](http://www.matthewtancik.com/nerf) by implementing a cognitive-inspired mechanism that fuses inputs from multiple sources into a memory-like structure, thus improving the storing capacity and extracting more intrinsic and correlated information. BioNeRF also mimics a behavior observed in pyramidal cells concerning contextual information, in which the memory is provided as the context and combined with the inputs of two subsequent blocks of dense layers, one responsible for producing the volumetric densities and the other the colors used to render the novel view. 

# Pipeline
<img src='https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/BioNeRF.png'/>

Here is an overview pipeline for BioNeRF, we will walk through each component in this guide.

## Positional Feature Extraction
The first step consists of feeding two neural models simultaneously, namely $M_{\Delta}$ and $M_c$, with the camera positional information. The output of these models encodes the positional information from the input image. Although the input is the same, the neural models do not share weights and follow a different flow in the next steps.

## Cognitive Filtering
This step performs a series of operations, called \emph{filters}, that work on the embeddings coming from the previous step. There are four filters this step derives: density, color, memory, and modulation.

## Memory Updating
Updating the memory requires the implementation of a mechanism capable of obliterating trivial information, which is performed using the memory filter (Step 3.1 in Figure~\ref{f.bionerf}). Fist, one needs to compute a signal modulation **$\mu$**, for further introducing new experiences in the memory **$\Psi$** through the modulating variable **$\mu$** using a $\textit{tanh}$ function (Step 3.2 in the figure).

## Contextual Inference
This step is responsible for adding contextual information to BioNeRF. Two new embeddings are generate, i.e., **${h}^{\prime}_\Delta$** and **${h}^{\prime}_c$** based on density and color filters, respectively (Step 4 in the figure), which further feed two neural models, i.e., $M^\prime_\Delta$ and $M^{\prime}$. Subsequently, $M^{\prime}_\Delta$ outputs the volume density, while color information is predicted by $M^{\prime}_c$, further used to compute the final predicted pixel information and the loss function.

# Results

For results, view the [paper page](https://arxiv.org/pdf/2402.07310)!
