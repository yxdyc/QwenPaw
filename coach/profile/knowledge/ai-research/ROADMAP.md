# AI/ML Research Learning Roadmap

Target profiles: Senior AI R&D scientist, rigorous reviewer, practitioner across ML/LLM/Agents

## Difficulty Scale

Each topic is rated 1-10. K+1 means advancing by one level from current mastery.

## Learning Path

### Tier 1: Foundations (Difficulty 1-3)

#### T1.1 Linear Algebra for ML
- **Difficulty**: 2
- **Prerequisites**: Undergraduate calculus
- **Key concepts**: Vectors, matrices, eigendecomposition, SVD, matrix calculus, positive definite matrices
- **Assessment criteria**: Can derive gradient of matrix operations; understands why SVD matters for dimensionality reduction
- **Resources**: Strang "Linear Algebra", Goodfellow et al. Ch. 2

#### T1.2 Probability & Information Theory
- **Difficulty**: 3
- **Prerequisites**: Multivariate calculus
- **Key concepts**: Bayes' theorem, distributions, MLE/MAP, entropy, KL divergence, mutual information, exponential family
- **Assessment criteria**: Can derive MLE for common distributions; can explain KL divergence intuitively and compute it
- **Resources**: Bishop "Pattern Recognition" Ch. 1-2, MacKay "Information Theory"

#### T1.3 Classical ML Algorithms
- **Difficulty**: 3
- **Prerequisites**: T1.1, T1.2
- **Key concepts**: Linear/logistic regression, SVM, decision trees, random forests, k-means, PCA, bias-variance tradeoff
- **Assessment criteria**: Can implement from scratch; can explain when to use each; understands regularization
- **Resources**: Murphy "ML: A Probabilistic Perspective", Hastie et al. "Elements of Statistical Learning"

#### T1.4 Optimization for ML
- **Difficulty**: 3
- **Prerequisites**: T1.1, convex optimization basics
- **Key concepts**: Gradient descent variants (SGD, Adam, AdaGrad), convexity, Lagrangian duality, learning rate scheduling
- **Assessment criteria**: Can derive SGD updates; understands why Adam works; can diagnose optimization issues (vanishing gradients, saddle points)
- **Resources**: Boyd & Vandenberghe "Convex Optimization", Nocedal & Wright

### Tier 2: Deep Learning Core (Difficulty 4-6)

#### T2.1 Neural Network Architectures
- **Difficulty**: 4
- **Prerequisites**: T1.1, T1.3, T1.4
- **Key concepts**: MLP, CNN, RNN/LSTM/GRU, attention mechanism, residual connections, normalization (BN, LN, RMSNorm)
- **Assessment criteria**: Can explain each architecture's inductive bias; can derive backprop through attention; understands scaling laws
- **Resources**: Goodfellow et al. "Deep Learning", Vaswani et al. (2017)

#### T2.2 Training Dynamics & Regularization
- **Difficulty**: 5
- **Prerequisites**: T2.1, T1.4
- **Key concepts**: Generalization bounds, weight decay, dropout, data augmentation, learning rate warmup/cosine, label smoothing, mixup
- **Assessment criteria**: Can explain why overparameterized models generalize; can design training recipe for a new task
- **Resources**: Zhang et al. (2017) "Understanding DL requires rethinking generalization"

#### T2.3 Self-Supervised & Contrastive Learning
- **Difficulty**: 5
- **Prerequisites**: T2.1, T2.2
- **Key concepts**: Pretext tasks, contrastive objectives (InfoNCE, SimCLR), masked prediction (BERT, MAE), CLIP, DINO
- **Assessment criteria**: Can explain contrastive loss derivation; understands why SSL works; can compare pretext tasks
- **Resources**: Chen et al. (SimCLR), He et al. (MoCo), Dosovitskiy (ViT/MAE)

#### T2.4 Generative Models
- **Difficulty**: 6
- **Prerequisites**: T2.1, T1.2, T2.2
- **Key concepts**: VAE, GAN, normalizing flows, diffusion models (DDPM, score-based), flow matching
- **Assessment criteria**: Can derive ELBO; can explain diffusion forward/reverse process; understands tradeoffs between generative families
- **Resources**: Kingma (VAE), Ho et al. (DDPM), Song et al. (score-based)

### Tier 3: LLM & Foundation Models (Difficulty 6-8)

#### T3.1 Transformer Architecture Deep Dive
- **Difficulty**: 6
- **Prerequisites**: T2.1
- **Key concepts**: Multi-head attention, positional encoding (absolute, RoPE, ALiBi), KV cache, GQA/MQA, MoE, FlashAttention
- **Assessment criteria**: Can derive attention complexity; can explain RoPE vs absolute PE tradeoffs; understands MoE routing
- **Resources**: Vaswani (2017), Su et al. (RoPE), Shazeer (MoE)

#### T3.2 Pre-training & Scaling Laws
- **Difficulty**: 7
- **Prerequisites**: T3.1, T2.2
- **Key concepts**: Chinchilla scaling, compute-optimal training, data mixture, curriculum learning, training instability, loss prediction
- **Assessment criteria**: Can compute optimal model/data allocation; understands data quality impact; can diagnose training loss spikes
- **Resources**: Hoffmann et al. (Chinchilla), Kaplan et al. (scaling laws)

#### T3.3 Alignment & Fine-tuning
- **Difficulty**: 7
- **Prerequisites**: T3.1, T3.2
- **Key concepts**: SFT, RLHF (PPO), DPO, constitutional AI, reward modeling, safety alignment, instruction following
- **Assessment criteria**: Can explain RLHF pipeline end-to-end; can compare DPO vs PPO; understands alignment tax and jailbreaking
- **Resources**: Ouyang et al. (InstructGPT), Rafailov et al. (DPO), Anthropic (Constitutional AI)

#### T3.4 Reasoning & Chain-of-Thought
- **Difficulty**: 7
- **Prerequisites**: T3.1, T3.3
- **Key concepts**: CoT prompting, self-consistency, tree-of-thought, process reward models, math reasoning, code generation, self-verification
- **Assessment criteria**: Can design CoT strategy for a task; understands when CoT helps vs hurts; can evaluate reasoning quality
- **Resources**: Wei et al. (CoT), Wang et al. (self-consistency), Yao et al. (ToT)

#### T3.5 Evaluation & Benchmarking
- **Difficulty**: 6
- **Prerequisites**: T1.2, T3.1
- **Key concepts**: Benchmark design, contamination detection, statistical significance, Elo rating, human eval, LLM-as-judge, pass@k
- **Assessment criteria**: Can design fair evaluation; can detect benchmark contamination; understands metric limitations
- **Resources**: Liang et al. (HELM), OpenAI eval methodology, Anthropic eval approach

### Tier 4: Agents & Systems (Difficulty 7-9)

#### T4.1 LLM Agent Architectures
- **Difficulty**: 7
- **Prerequisites**: T3.1, T3.4
- **Key concepts**: ReAct, tool use, planning (ToT, HuggingGPT), memory (short/long-term), multi-agent collaboration, function calling
- **Assessment criteria**: Can design agent architecture for a task; understands tool-use tradeoffs; can evaluate agent reliability
- **Resources**: Yao et al. (ReAct), Wang et al. (survey), Park et al. (generative agents)

#### T4.2 MCP & Agent Communication Protocols
- **Difficulty**: 7
- **Prerequisites**: T4.1
- **Key concepts**: Model Context Protocol, A2A protocol, agent-to-agent communication, tool server lifecycle, context window management
- **Assessment criteria**: Can design MCP integration; understands protocol tradeoffs; can manage multi-agent context
- **Resources**: Anthropic MCP spec, Google A2A spec, QwenPaw architecture

#### T4.3 RAG & Knowledge Systems
- **Difficulty**: 7
- **Prerequisites**: T3.1, information retrieval basics
- **Key concepts**: Embedding models, vector databases, chunking strategies, hybrid retrieval, reranking, knowledge graphs, agentic RAG
- **Assessment criteria**: Can design RAG pipeline; understands retrieval-quality tradeoffs; can evaluate RAG end-to-end
- **Resources**: Lewis et al. (RAG), Karpukhin et al. (DPR), Gao et al. (survey)

#### T4.4 Experimental Design & Rigor
- **Difficulty**: 8
- **Prerequisites**: T1.2, T3.5
- **Key concepts**: Ablation studies, statistical significance, confidence intervals, reproducibility, baseline fairness, claim-evidence alignment
- **Assessment criteria**: Can design rigorous experiment; can identify common pitfalls (cherry-picking, p-hacking); can write defensible claims
- **Resources**: Dodge et al. "Show Your Work", Narang et al. "Adversarial evaluation"

#### T4.5 Safety & Guardrails
- **Difficulty**: 8
- **Prerequisites**: T3.3, T4.1
- **Key concepts**: Tool-use safety, prompt injection, sandboxing, output filtering, red-teaming, responsible AI frameworks
- **Assessment criteria**: Can design safety guardrails for agent systems; understands attack vectors; can implement defense-in-depth
- **Resources**: OWASP LLM Top 10, Anthropic RSP, NIST AI RMF

### Tier 5: Frontier Research (Difficulty 9-10)

#### T5.1 Mechanistic Interpretability
- **Difficulty**: 9
- **Prerequisites**: T3.1, T1.1
- **Key concepts**: Circuit discovery, feature visualization, sparse autoencoders, causal tracing, representation engineering
- **Assessment criteria**: Can apply circuit analysis; understands SAE training; can interpret model internals
- **Resources**: Olah et al. (circuits), Cunningham et al. (SAE), Meng et al. (ROME)

#### T5.2 Multi-Agent Systems & Emergence
- **Difficulty**: 9
- **Prerequisites**: T4.1, T4.2
- **Key concepts**: Emergent behavior, agent specialization, social dynamics in agent populations, agent benchmarks, collective intelligence
- **Assessment criteria**: Can design multi-agent system; can identify and measure emergence; understands coordination challenges
- **Resources**: Du et al. (improving factuality), Li et al. (CAMEL), survey papers

#### T5.3 Research Paper Writing & Review
- **Difficulty**: 9
- **Prerequisites**: T4.4, domain expertise in chosen area
- **Key concepts**: Paper structure, related work positioning, ablation design, rebuttal strategy, reviewer perspective, impact assessment
- **Assessment criteria**: Can write a tier-1 paper; can review with constructive rigor; can identify overclaiming
- **Resources**: He & Halpern "Writing tips", Simon Peyton Jones "How to give a good talk"

## Project Anchors

- **Learning-agent prototype**: T4.1, T4.2, T4.4 — building and evaluating a small agent system
- **Cross-model validation experiments**: T4.4, T3.5 — rigorous experimental methodology
- **Tool-execution reliability study**: T4.1, T5.2 — state, side effects, evaluation, and recovery in agent architectures

## K+1 Progression Guide

Typical paths:
- **ML Engineer**: T1.3 → T2.1 → T2.2 → T3.1 → T3.2
- **LLM Researcher**: T3.1 → T3.2 → T3.3 → T3.4 → T5.1
- **Agent Systems**: T3.1 → T4.1 → T4.2 → T4.3 → T5.2
- **Research Scientist**: T4.4 → T3.5 → T5.3 (cross-cutting with any technical track)

Cross-cutting skill: T4.4 (Experimental Design) is critical for all paths — prioritize early.
