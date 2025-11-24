# NeuroScript v2: Block Candidates

**Design Philosophy:** Blocks all the way down. Atomic primitives compose into layers, layers into modules, modules into architectures, architectures into models, models into ensembles. Embed your friend's entire GPT as a single block in your mixture-of-experts. Why not?

---

## Hierarchy Levels

```
Level 0: Atomic (primitives)
    ↓
Level 1: Composite (common patterns)
    ↓
Level 2: Architectural (well-known blocks)
    ↓
Level 3: Model (complete models as blocks)
    ↓
Level 4: Meta (ensembles, routers, control flow)
```

---

## Level 0: Atomic Blocks (Primitives)

**The building blocks of building blocks.**

### Core Operations
- `Linear` - dense/fully-connected layer
- `Bias` - additive bias
- `Scale` - multiplicative scaling
- `MatMul` - matrix multiplication
- `Einsum` - Einstein summation (generalized tensor operations)

### Activations
- `ReLU` - rectified linear unit
- `GELU` - Gaussian error linear unit
- `SiLU` / `Swish` - sigmoid linear unit
- `Tanh` - hyperbolic tangent
- `Sigmoid` - logistic function
- `Softmax` - normalized exponential
- `Mish` - self-regularized non-monotonic activation
- `PReLU` - parametric ReLU
- `ELU` - exponential linear unit

### Normalizations
- `LayerNorm` - layer normalization
- `BatchNorm` - batch normalization
- `RMSNorm` - root mean square normalization
- `GroupNorm` - group normalization
- `InstanceNorm` - instance normalization
- `WeightNorm` - weight normalization

### Regularization
- `Dropout` - random neuron dropout
- `DropPath` - stochastic depth
- `Dropblock` - structured dropout for CNNs
- `DropConnect` - connection dropout
- `SpecAugment` - frequency/time masking (audio)

### Convolutions
- `Conv1d` - 1D convolution (sequences)
- `Conv2d` - 2D convolution (images)
- `Conv3d` - 3D convolution (video/volumetric)
- `DepthwiseConv` - channel-wise convolution
- `SeparableConv` - depthwise + pointwise
- `TransposedConv` / `Deconv` - upsampling convolution
- `DilatedConv` - atrous convolution

### Pooling
- `MaxPool` - max pooling
- `AvgPool` - average pooling
- `AdaptiveAvgPool` - output-size-adaptive pooling
- `AdaptiveMaxPool` - output-size-adaptive max pooling
- `GlobalAvgPool` - spatial averaging
- `GlobalMaxPool` - spatial max

### Embeddings
- `Embedding` - discrete token → dense vector
- `PositionalEncoding` - sinusoidal position embeddings
- `LearnedPositionalEmbedding` - trainable positions
- `RotaryEmbedding` (RoPE) - rotary position embeddings
- `ALiBi` - attention with linear biases

### Utility
- `Reshape` - tensor reshaping
- `Transpose` - dimension permutation
- `Concatenate` - tensor concatenation
- `Split` - tensor splitting
- `Slice` - tensor slicing
- `Pad` - tensor padding
- `Crop` - tensor cropping
- `Cast` - dtype conversion
- `Clone` - tensor duplication
- `Identity` - pass-through (useful for routing)

---

## Level 1: Composite Blocks (Common Patterns)

**Built from atomic blocks. Reusable patterns.**

### Attention Mechanisms
- `ScaledDotProductAttention` - core attention operation
- `MultiHeadAttention` - parallel attention heads
- `SelfAttention` - query=key=value
- `CrossAttention` - query ≠ key=value
- `FlashAttention` - memory-efficient attention
- `SparseAttention` - local/strided/block-sparse patterns
- `LinearAttention` - kernel-based O(n) attention
- `GroupedQueryAttention` (GQA) - shared key/value heads
- `MultiQueryAttention` (MQA) - single key/value head

### Feed-Forward Networks
- `FFN` / `MLP` - linear → activation → linear
- `GatedFFN` - gating mechanism (GLU-style)
- `SwiGLU` - SiLU gated FFN
- `GeGLU` - GELU gated FFN
- `Expert` - single expert block (for MoE)

### Residual Connections
- `Residual` - add(x, f(x))
- `PreNormResidual` - norm → f(x) → add
- `PostNormResidual` - f(x) → norm → add
- `HighwayConnection` - gated residual
- `DenseConnection` - concatenate all previous layers

### Gating Mechanisms
- `GLU` - gated linear unit
- `LSTM-Gate` - forget/input/output gates
- `GRU-Gate` - update/reset gates

### Positional Processing
- `RelativePositionBias` - T5-style learned biases
- `ConditionalPositionEncoding` - conditional position embeddings

---

## Level 2: Architectural Blocks (Well-Known Components)

**Famous architectures broken into composable blocks.**

### Transformer Family
- `TransformerBlock` - attention + FFN + residuals
- `TransformerEncoderBlock` - self-attention block
- `TransformerDecoderBlock` - cross-attention block
- `TransformerStack` - N stacked transformer blocks
- `PrefixLM` - prefix language model block
- `EncoderDecoder` - full transformer architecture

### Convolutional Architectures
- `ResNetBlock` - residual block (basic/bottleneck)
- `ResNeXtBlock` - grouped convolutions
- `DenseBlock` - densely connected block
- `InceptionBlock` - multi-scale convolutions
- `SEBlock` - squeeze-and-excitation
- `MBConvBlock` - mobile inverted bottleneck
- `FusedMBConv` - fused mobile conv (EfficientNet)
- `BottleneckBlock` - 1x1 → 3x3 → 1x1 conv

### Recurrent/State Space
- `LSTM` - long short-term memory
- `GRU` - gated recurrent unit
- `BiLSTM` / `BiGRU` - bidirectional variants
- `MambaBlock` - selective state space model
- `S4Block` - structured state space sequence model
- `H3Block` - hybrid H3 layer
- `RetNetBlock` - retentive network
- `RWKVBlock` - receptance weighted key value

### Vision-Specific
- `ConvNeXtBlock` - modernized ResNet block
- `SwinBlock` - shifted window attention
- `ViTBlock` - vision transformer block
- `PatchEmbedding` - image → patch embeddings
- `PatchMerging` - hierarchical patch reduction
- `FPN` - feature pyramid network
- `UNetBlock` - U-Net encoder/decoder block
- `ResUNetBlock` - residual U-Net block

### Generative
- `VAEEncoder` - variational autoencoder encoder
- `VAEDecoder` - VAE decoder
- `GANGenerator` - GAN generator block
- `GANDiscriminator` - GAN discriminator block
- `DiffusionUNet` - diffusion model U-Net
- `DiffusionTimestepEmbed` - timestep conditioning
- `NoiseScheduler` - diffusion noise scheduling

### Graph Neural Networks
- `GCNLayer` - graph convolutional network
- `GATLayer` - graph attention network
- `GraphSAGE` - graph sample and aggregate
- `GINLayer` - graph isomorphism network
- `MessagePassing` - generic message passing
- `EdgeConv` - edge convolution
- `GlobalPooling` - graph-level pooling

### Audio/Speech
- `MelSpectrogram` - mel-scale spectrogram
- `MFCC` - mel-frequency cepstral coefficients
- `WaveNetBlock` - dilated causal convolution
- `Conformer` - convolution-augmented transformer
- `WhisperEncoder` - speech encoder block
- `WhisperDecoder` - speech decoder block

### Normalization Variants
- `AdaptiveLayerNorm` - adaptive LN (e.g., for diffusion)
- `ConditionalLayerNorm` - class-conditional normalization
- `SpectralNorm` - spectral normalization (GANs)

---

## Level 3: Model Blocks (Complete Models as Composable Blocks)

**Entire models packaged as single blocks. Embed GPT-2 in your architecture. Go wild.**

### Language Models
- `GPT` - autoregressive language model (any size)
- `BERT` - bidirectional encoder
- `T5` - encoder-decoder model
- `LLaMA` - LLaMA architecture (any variant)
- `Mistral` - Mistral model
- `Mamba` - full Mamba model
- `RWKV` - full RWKV model
- `Pythia` - Pythia model family
- `Falcon` - Falcon model
- `Phi` - Phi small language models

### Vision Models
- `ResNet` - full ResNet (18/34/50/101/152)
- `EfficientNet` - EfficientNet (B0-B7)
- `ViT` - vision transformer
- `Swin` - Swin transformer
- `ConvNeXt` - ConvNeXt architecture
- `CLIP-VisualEncoder` - CLIP image encoder
- `DINOv2` - self-supervised vision model

### Multimodal
- `CLIP` - contrastive language-image model
- `BLIP` - bootstrapped language-image model
- `Flamingo` - visual language model
- `LLaVA` - large language and vision assistant
- `CoCa` - contrastive captioner

### Generative Models
- `StableDiffusion-UNet` - diffusion model
- `VAE-KL` - Kullback-Leibler VAE
- `VQVAE` - vector quantized VAE
- `StyleGAN` - StyleGAN generator
- `ControlNet` - conditional diffusion control

### Audio Models
- `Whisper` - speech recognition
- `Wav2Vec2` - speech representation
- `HuBERT` - hidden unit BERT
- `MusicGen` - music generation
- `AudioLDM` - audio latent diffusion

### Specialized
- `AlphaFold-Evoformer` - protein folding
- `ProteinMPNN` - protein design
- `ESMFold` - protein language model
- `MolFormer` - molecular transformer

---

## Level 4: Meta Blocks (Control Flow, Routing, Composition)

**Blocks that orchestrate other blocks.**

### Routing
- `Switch` - conditional routing (if-else)
- `Router` - learned routing (multi-path)
- `MixtureOfExperts` (MoE) - sparse expert routing
- `TopKRouter` - route to top-k experts
- `LoadBalancedRouter` - expert load balancing
- `ConditionalCompute` - dynamic depth

### Composition
- `Sequential` - linear chain of blocks
- `Parallel` - multiple blocks in parallel
- `Residual` - skip connection wrapper
- `Ensemble` - average/vote multiple models
- `LoRA` - low-rank adaptation wrapper
- `Adapter` - adapter layer wrapper
- `PrefixTuning` - prefix tuning wrapper

### Recursive/Dynamic
- `RecurrentBlock` - apply block N times
- `DynamicDepth` - variable-depth networks
- `NeuralODE` - neural ordinary differential equations
- `UniversalTransformer` - adaptive computation time

### Multi-Scale
- `Pyramid` - multi-scale processing
- `Cascade` - coarse-to-fine processing
- `HierarchicalMerge` - merge across scales

### Utilities
- `Checkpoint` - gradient checkpointing
- `Quantize` - quantization (INT8/INT4)
- `Prune` - structured/unstructured pruning
- `DistillationHead` - knowledge distillation
- `EMA` - exponential moving average

---

## Level 5: External Model Blocks (Someone Else's Model)

**Import arbitrary models as blocks.**

### Import Formats
- `HuggingFaceModel` - any model from HF Hub
- `TorchHubModel` - PyTorch Hub models
- `ONNXModel` - ONNX models
- `TensorFlowModel` - TF SavedModel
- `JAXModel` - JAX/Flax models
- `SafetensorsModel` - safetensors format

### Example Usage
```yaml
components:
  base_encoder:
    block: HuggingFaceModel
    params:
      repo: "openai/clip-vit-base-patch32"
      component: "vision_model"
      freeze: true

  my_decoder:
    block: TransformerStack
    params:
      layers: 6
      dim: 512

topology:
  image -> base_encoder -> adapter -> my_decoder -> output
```

---

## Domain-Specific Block Collections

### Computer Vision
**Primitives:** Conv2d, MaxPool, BatchNorm, ReLU
**Composite:** ResNetBlock, SEBlock, ViTBlock
**Models:** ResNet50, EfficientNetB0, ViT-B/16
**Meta:** FPN, CascadeRCNN

### Natural Language Processing
**Primitives:** Embedding, Linear, LayerNorm, Dropout
**Composite:** TransformerBlock, MambaBlock
**Models:** GPT-2, BERT-base, LLaMA-7B
**Meta:** MoE, LoRA, PrefixTuning

### Audio/Speech
**Primitives:** Conv1d, MelSpectrogram, LSTM
**Composite:** ConformerBlock, WaveNetBlock
**Models:** Whisper-small, Wav2Vec2
**Meta:** CTC, Attention

### Graphs
**Primitives:** MessagePassing, EdgeConv
**Composite:** GATLayer, GCNLayer
**Models:** GraphSAGE, GIN
**Meta:** GlobalPooling

### Reinforcement Learning
- `PolicyHead` - policy network head
- `ValueHead` - value network head
- `ActorCritic` - actor-critic architecture
- `DQN` - deep Q-network
- `PPONetwork` - PPO architecture

---

## Adapter/Compatibility Blocks

**Make incompatible blocks work together.**

### Shape Adapters
- `DimensionAdapter` - change feature dimensions
- `SequenceLengthAdapter` - resample sequence length
- `BatchAdapter` - handle batch size changes
- `ChannelAdapter` - adapt channel count
- `SpatialAdapter` - resize spatial dimensions

### Type Adapters
- `DTypeAdapter` - convert float32 ↔ float16 ↔ bfloat16
- `QuantizationAdapter` - float ↔ int8/int4
- `DeviceAdapter` - CPU ↔ GPU ↔ TPU

### Format Adapters
- `ImageToPatches` - image → patch sequence
- `PatchesToImage` - patch sequence → image
- `TokensToEmbedding` - discrete → continuous
- `EmbeddingToLogits` - continuous → discrete
- `AudioToSpectrogram` - waveform → frequency
- `SpectrogramToAudio` - frequency → waveform

### Connector Blocks
- `Projection` - arbitrary shape transformation
- `Upsampler` - increase resolution
- `Downsampler` - decrease resolution
- `Interpolate` - smooth resampling
- `BridgeBlock` - generic A → B converter

---

## Specialized/Exotic Blocks

### Memory/Context
- `MemoryBank` - external memory (NTM, DNC)
- `KNNMemory` - k-nearest neighbor lookup
- `VectorDatabase` - embedding search
- `ContextWindow` - sliding window buffer

### Symbolic/Neuro-Symbolic
- `LogicGate` - differentiable logic
- `ProgramSynthesis` - neural program synthesis
- `TreeLSTM` - tree-structured LSTM
- `GraphGrammar` - graph generation rules

### Equivariance/Invariance
- `RotationEquivariant` - rotation equivariance (e.g., E(n) layers)
- `TranslationInvariant` - translation invariance
- `PermutationInvariant` - permutation invariance (Set Transformer)
- `ScaleEquivariant` - scale equivariance

### Energy-Based
- `EnergyFunction` - energy-based model
- `Hopfield` - modern Hopfield network
- `Boltzmann` - restricted Boltzmann machine

### Emerging Research
- `KolmogorovArnold` (KAN) - KAN layers
- `LiquidNeuron` - liquid time-constant networks
- `HyperNetwork` - networks that generate networks
- `MetaLearner` - MAML, Reptile, etc.
- `NeuralTangentKernel` - NTK-based blocks

---

## Composition Examples

### Example 1: Classic CNN
```yaml
Sequential([
  Conv2d(3, 64),
  ReLU,
  MaxPool,
  ResNetBlock(64, 128),
  ResNetBlock(128, 256),
  GlobalAvgPool,
  Linear(256, 10)
])
```

### Example 2: Vision Transformer
```yaml
Sequential([
  PatchEmbedding(patch_size=16),
  PositionalEncoding,
  TransformerStack(layers=12),
  GlobalAvgPool,
  Linear(768, 1000)
])
```

### Example 3: Hybrid (Your Friend's Model Inside Yours)
```yaml
components:
  # Your friend's entire GPT-2
  pretrained_gpt:
    block: HuggingFaceModel(repo="gpt2", freeze=True)

  # Your custom expert
  custom_expert:
    block: MambaBlock(dim=768, layers=6)

  # Router decides which to use
  router:
    block: TopKRouter(k=1, num_experts=2)

topology:
  input -> router -> [pretrained_gpt, custom_expert] -> merge -> output
```

### Example 4: Multi-Modal Madness
```yaml
components:
  # Vision: CLIP encoder
  vision:
    block: HuggingFaceModel(repo="openai/clip-vit-large")

  # Language: LLaMA
  language:
    block: HuggingFaceModel(repo="meta/llama-7b")

  # Fusion: Custom cross-attention
  fusion:
    block: CrossAttention(dim=1024)

topology:
  image -> vision -> vision_embed
  text -> language -> text_embed
  [vision_embed, text_embed] -> fusion -> output
```

---

## Initial Priority List (Phase 2)

**Start with these 15 blocks to prove the system:**

### Tier 0 (Must Have - Week 1)
1. `Linear` - dimension transformation
2. `Embedding` - token → vector
3. `LayerNorm` - normalization
4. `Dropout` - regularization
5. `GELU` - activation

### Tier 1 (Core Patterns - Week 2)
6. `MultiHeadAttention` - attention mechanism
7. `FFN` - feed-forward network
8. `Residual` - skip connections
9. `PositionalEncoding` - position info

### Tier 2 (Architectural - Week 3)
10. `TransformerBlock` - complete transformer layer
11. `TransformerStack` - N stacked blocks

### Tier 3 (Composition - Week 4)
12. `Sequential` - chain blocks
13. `Parallel` - run blocks in parallel
14. `Concatenate` - merge tensors

### Tier 4 (External - Week 5)
15. `HuggingFaceModel` - import external models

**With these 15, you can build:**
- GPT-style models
- BERT-style models
- Hybrid architectures
- Import external models as components

Ship the validator with these, then expand.

---

## Block Development Workflow

### 1. Define Capability Spec
```yaml
# blocks/linear/block.yaml
name: Linear
version: 1.0.0
capabilities:
  inputs:
    x: {shape: [*, in_features], dtype: [float32, float16]}
  outputs:
    y: {shape: [*, out_features], dtype: input.x.dtype}
  params:
    in_features: {type: int, required: true}
    out_features: {type: int, required: true}
    bias: {type: bool, default: true}
```

### 2. Implement Module
```python
# blocks/linear/module.py
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.linear(x)
```

### 3. Write Tests
```python
# blocks/linear/test.py
def test_shape():
    block = Linear(128, 256)
    x = torch.randn(32, 128)
    y = block(x)
    assert y.shape == (32, 256)
```

### 4. Document
```markdown
# blocks/linear/README.md
Linear transformation: y = xW^T + b

Parameters:
- in_features: input dimension
- out_features: output dimension
- bias: whether to add bias (default: true)

Example:
  Linear(in_features=128, out_features=256)
```

### 5. Submit PR
```bash
cd blocks/linear
neuroscript validate-block .
git commit -m "Add Linear block"
gh pr create
```

---

## Future: Auto-Generated Blocks

**Why write blocks when you can generate them?**

```yaml
# Define a block from a paper
block_generator:
  name: CustomAttention
  based_on: MultiHeadAttention
  modifications:
    - replace: ScaledDotProduct -> CosineSimiliarity
    - add: TemperatureScaling(learnable=true)
    - insert_after: Attention -> LayerNorm

# System generates block.yaml + module.py + tests
```

Or even:

```yaml
# Describe in natural language
block_from_paper:
  paper: "arxiv:2304.12345"
  block: "Efficient Multi-Scale Attention (Section 3.2)"

# AI extracts block definition, generates code, you review
```

---

## Summary

**327+ block candidates** organized into:
- 50+ atomic primitives
- 80+ composite patterns
- 100+ architectural components
- 50+ complete models
- 30+ meta/routing blocks
- 17+ adapter/compatibility blocks

**Start with 15, expand to 50 in 3 months, community adds the rest.**

It's blocks all the way down. Ship it.
