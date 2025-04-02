# JEPA: Joint Embedding Predictive Architecture

This repository tracks resources, implementations, and research related to Joint Embedding Predictive Architecture (JEPA), a novel approach to self-supervised learning in AI.

## JEA vs JEPA: Understanding the Difference

### Joint Embedding Architecture (JEA)

Joint Embedding Architectures (JEAs) are self-supervised learning methods that:

- Learn to output similar embeddings for compatible inputs and dissimilar embeddings for incompatible inputs
- Typically rely on hand-crafted data augmentations to create different views of the same data
- Examples include DINO, MoCo, and other contrastive learning approaches
- May suffer from biases associated with invariance-based pretraining

### Joint Embedding Predictive Architecture (JEPA)

Joint Embedding Predictive Architecture (JEPA) is an evolution of self-supervised learning that:

- Predicts the representation of part of an input from the representation of other parts of the same input
- Works at a high level of abstraction rather than predicting pixel values directly
- Avoids the limitations of generative approaches by focusing on semantic features rather than low-level details
- Does not rely on hand-crafted data augmentations or pre-specified invariances
- Aims to learn more semantically meaningful representations

The key difference is that JEPA is predictive in latent space, focusing on high-level semantic features, while avoiding the pitfalls of both contrastive methods (which rely on hand-crafted augmentations) and generative methods (which focus too much on pixel-level details).

## JEPA Variants

1. **I-JEPA (Image-based JEPA)**
   - Focuses on learning representations from images
   - Predicts representations of target blocks from context blocks in the same image
   - Computationally efficient compared to methods requiring multiple augmented views
   - [Original paper (2023)](https://arxiv.org/abs/2301.08243)

2. **V-JEPA (Video-based JEPA)**
   - Extends JEPA to video understanding
   - Makes predictions in latent space for temporal sequences
   - Does not use pretrained image encoders, text, negative examples, or pixel-level reconstruction
   - Captures temporal dynamics essential for understanding motion and activities
   - [Meta AI blog post](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)

3. **D-JEPA (Denoising JEPA)**
   - Integrates JEPA within generative modeling
   - Reinterprets JEPA as a generalized next-token prediction strategy
   - Combines the strengths of generative and predictive approaches

4. **S-JEPA (Skeleton Joint Embedding Predictive Architecture)**
   - Designed specifically for skeletal action recognition
   - Predicts latent representations of missing joints from partial skeleton sequences
   - Enables more effective learning of human motion patterns from limited data
   - [Project website](https://sjepa.github.io/)
   - [Paper (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)

5. **MC-JEPA (Motion-Content Joint-Embedding Predictive Architecture)**
   - Designed to simultaneously interpret both dynamic elements (motion) and static details (content) in video
   - Uses a shared encoder for both aspects, making it more efficient
   - Particularly valuable for applications like autonomous driving, video surveillance, and activity recognition
   - [Paper (2023)](https://arxiv.org/abs/2307.12698)

6. **DMT-JEPA (Discriminative Masked Targets for Joint-Embedding Predictive Architecture)**
   - Addresses JEPA's limitations in understanding local semantics
   - Generates discriminative latent targets from neighboring information
   - Computes feature similarities between masked patches and neighboring patches to select semantically meaningful relations
   - Demonstrates improved performance on tasks like image classification, semantic segmentation, and object detection
   - [Paper (2024)](https://arxiv.org/abs/2405.17995)
   - [Code repository](https://github.com/DMTJEPA/DMTJEPA)

7. **Image World Models (IWM)**
   - Generalizes JEPA to handle different types of corruptions beyond masking
   - Explores both invariant models (maintaining stable features) and equivariant models (adapting to changes while preserving relationships)
   - Enhances model resilience and adaptability
   - [Paper (2024)](https://arxiv.org/abs/2403.00504)

## JEPA Resources

### GitHub Repositories

1. [Official I-JEPA Repository](https://github.com/facebookresearch/ijepa) - Meta AI's implementation of Image-based JEPA
2. [Official V-JEPA Repository](https://github.com/facebookresearch/jepa) - Meta AI's implementation of Video-based JEPA
3. [LumenPallidium/jepa](https://github.com/LumenPallidium/jepa) - Experiments with JEPAs and self-supervised learning

### YouTube Tutorials and Explanations

1. [Overview of Joint Embedding Predictive Architectures](https://www.youtube.com/watch?v=vhDLp2VeVwE) - Breakdown of I-JEPA and V-JEPA papers
2. [Generative AI or Predictive AI? Deep dive on V-JEPA](https://www.youtube.com/watch?v=P3NfkP3eyeo) - Analysis of V-JEPA and its implications for video understanding
3. [JEPA Joint Embedding Predictive Architecture](https://www.youtube.com/watch?v=KZo4ZUWRKc0) - Introduction to JEPA concepts
4. [JEPA: Joint Embedding Predictive Architectures - Yann LeCun](https://www.youtube.com/watch?v=jSdHmImyUjk) - Yann LeCun's presentation on JEPA
5. [Self-supervised Learning, JEPA, and the Future of AI](https://www.youtube.com/watch?v=xIn-Czj1g2Q) - Discussion on JEPA's role in AI's future

### Papers and Articles

1. [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) - Original I-JEPA paper
2. [I-JEPA: The first AI model based on Yann LeCun's vision for more human-like AI](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) - Meta AI blog post about I-JEPA
3. [What is Joint Embedding Predictive Architecture (JEPA)?](https://www.turingpost.com/p/jepa) - Comprehensive explanation of JEPA
4. [NYU Deep Learning Spring 2021 Course](https://atcold.github.io/NYU-DLSP21/) - Course by Yann LeCun and Alfredo Canziani covering energy-based models and self-supervised learning

## Energy-Based Models (EBMs) and JEPA

Energy-Based Models (EBMs) are a fundamental concept in Yann LeCun's vision for AI and are closely related to the development of JEPA. Understanding EBMs provides important context for JEPA's theoretical foundations.

### What are Energy-Based Models?

Energy-Based Models provide a unified framework for many machine learning approaches by associating a scalar energy value to each configuration of the variables of interest. Key characteristics include:

- **Energy Function**: Maps any configuration of variables to a single scalar called energy
- **Learning Process**: Involves shaping the energy function so that desired configurations have low energy, while undesired ones have high energy
- **Inference**: Finding configurations with minimal energy (most likely or desired states)

EBMs offer several advantages:

1. **Flexibility**: Can model complex dependencies between variables without assuming a particular factorization
2. **No Normalization Requirement**: Don't require computing normalized probabilities, which can be computationally expensive
3. **Unified Framework**: Provide a common theoretical foundation for many learning methods

### Connection to JEPA

JEPA can be viewed as an implementation of energy-based learning principles where:

- The energy function is implicitly defined by the distance between predicted and actual representations
- The model learns to minimize this energy by making better predictions in latent space
- The focus on abstract representations rather than pixel-level details aligns with EBM's emphasis on capturing meaningful configurations

Yann LeCun's work on EBMs has directly influenced the development of JEPA as part of his broader vision for self-supervised learning. In his framework, JEPA serves as a practical implementation of energy-based principles for representation learning.

## Free Energy Principle and Brain Function

The Free Energy Principle (FEP), proposed by Karl Friston, offers a universal theory of brain function that shares conceptual similarities with energy-based models and JEPA. This principle provides a neuroscientific perspective that complements the computational approaches of JEPA.

### The Free Energy Principle Explained

The Free Energy Principle posits that biological systems, including the brain, work to minimize a quantity called "free energy" - a measure of the difference between an organism's internal model of the world and the actual sensory inputs it receives. Key aspects include:

- **Active Inference**: Organisms act to confirm their predictions and minimize surprise
- **Hierarchical Predictive Coding**: The brain uses a hierarchical system where higher levels predict the activity of lower levels
- **Variational Free Energy**: A mathematical formulation that approximates the divergence between the brain's model and reality

### Connections to JEPA and Energy-Based Models

The Free Energy Principle aligns with JEPA and EBMs in several ways:

1. **Prediction-Based Learning**: Both FEP and JEPA emphasize learning through prediction rather than reconstruction
2. **Hierarchical Representation**: Both involve hierarchical models that capture increasingly abstract representations
3. **Energy Minimization**: Both frameworks involve minimizing a form of "energy" or divergence between predictions and reality
4. **Focus on Relevant Information**: Both prioritize learning meaningful, predictive features rather than reconstructing every detail

### Resources on Free Energy Principle

1. [The Free Energy Principle Explained](https://www.youtube.com/watch?v=iPj9D9LgK2A) - Clear explanation of the free energy principle and its implications
2. [The Free Energy Principle: Predictive Coding, Active Inference & Consciousness](https://www.youtube.com/watch?v=UkH-7gZnrr4) - Exploration of how the principle relates to consciousness and cognition

The Free Energy Principle provides a biological grounding for many of the computational principles implemented in JEPA, suggesting that these approaches may be capturing fundamental aspects of how biological intelligence works.

## JEPA vs Other Self-Supervised Learning Approaches

To better understand JEPA's significance, it's helpful to compare it with other major self-supervised learning paradigms:

### JEPA vs Generative Models

| Aspect | JEPA | Generative Models (MAE, BEiT, SimMIM) |
|--------|------|----------------------------------------|
| **Learning Target** | Predicts high-level representations in latent space | Reconstructs low-level details (pixels, tokens) |
| **Processing Focus** | Semantic features | Every detail, including irrelevant ones |
| **Computational Efficiency** | More efficient (no pixel-level reconstruction) | Less efficient (decoding back to pixel space) |
| **Limitations** | Requires careful design of prediction targets | May focus too much on pixel-perfect reconstruction |
| **Examples** | I-JEPA, V-JEPA, S-JEPA | BERT, GPT, MAE, SimMIM |

### JEPA vs Contrastive Learning

| Aspect | JEPA | Contrastive Learning (SimCLR, MoCo, DINO) |
|--------|------|--------------------------------------------|
| **Learning Mechanism** | Prediction of representations | Comparison of positive/negative pairs |
| **Data Augmentation** | Minimal reliance on hand-crafted augmentations | Heavy reliance on carefully designed augmentations |
| **Training Signal** | Prediction error in latent space | Similarity/dissimilarity of sample pairs |
| **Collapse Prevention** | Structure of prediction task | Explicit mechanisms (momentum encoders, stop-gradient) |
| **Examples** | I-JEPA, V-JEPA, MC-JEPA | SimCLR, MoCo, DINO, Barlow Twins |

### JEPA vs Joint Embedding Architecture (JEA)

| Aspect | JEPA | JEA |
|--------|------|-----|
| **Core Approach** | Predicts representations of one view from another | Maps similar inputs to similar embeddings |
| **Working Principle** | Prediction in latent space | Similarity in latent space |
| **Handling of Views** | One view predicts another | Multiple views mapped to similar points |
| **Training Objective** | Minimize prediction error | Maximize similarity of positive pairs |
| **Examples** | I-JEPA, V-JEPA, DMT-JEPA | DINO, MoCo, SwAV |

## Why JEPA Matters

JEPA represents a significant advancement in self-supervised learning by addressing limitations of both contrastive and generative approaches. By predicting in representation space rather than pixel space, JEPA learns more semantic features without relying on hand-crafted data augmentations or focusing on irrelevant details.

Yann LeCun, Meta's Chief AI Scientist, has positioned JEPA as a key component in his vision for more human-like AI systems that can learn common sense knowledge about the world through passive observation. The connection to energy-based models provides JEPA with a strong theoretical foundation while offering a practical approach to implementing these principles in modern deep learning architectures.

The growing ecosystem of JEPA variants (I-JEPA, V-JEPA, S-JEPA, DMT-JEPA, etc.) demonstrates the architecture's versatility and potential to address diverse challenges across different data modalities and applications. As research progresses, JEPA's principles may help bridge the gap between current AI capabilities and more human-like understanding of the world.