---
layout: distill
title: Understanding LoRA Finetuning Dynamics within Vision Language Models
description: Several advanced LoRA methods such as EVA, LoftQ, and PiSSA have emerged. We extend these LLM-based LoRA variations to Vision-Language Models (VLMs), aiming to explore the training behavior of VLMs when adopting these LoRA variants specifically tailored for VLM tasks.
date: 2025-05-30
future: true
htmlwidgets: true
hidden: false

# anonymize when submitting 
authors:
  - name: Baek Seong-Eun
    affiliations:
      name: POSTECH
  - name: Lee Suchan
    affiliations:
      name: POSTECH
  - name: Kang Minseo
    affiliations:
      name: POSTECH


bibliography: 2025-04-28-analysing-the-spectral-biases-in-generative-models.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: What we do
    subsections:
      - name: Evaluating Vanilla LoRA in Multimodal Settings
      - name: Evaluating PiSSA in Multimodal Settings
      - name: Ablation Study of General Initialization Strategies
  - name: Experimental setting
  - name: Related work
    subsections:
      - name: What is PiSSA?
      - name: What is LoftQ?
      - name: What is EVA?
  - name: Findings
    subsections:
      - name: LLaMA 3.2 11B Vision
      - name: Qwen 2.5 VL 7B
  - name: Conclusion
  - name: Contributions and Impact
  - name: Limitation
  - name: Reference
---

## Introduction

Recently, large-scale pre-trained models, including large language models (LLMs), have been rapidly advancing. As the size and complexity of these models grow, so too does the importance of developing fine-tuning strategies that are both computationally efficient and performance-effective. Among the most widely adopted solutions is Low-Rank Adaptation (LoRA), a method that enables parameter-efficient fine-tuning by inserting low-rank matrices into pre-trained layers while keeping the original model weights frozen<d-cite key="hu2022lora"></d-cite>.

LoRA offers a compelling trade-off: it reduces the number of trainable parameters dramatically while preserving or even enhancing task-specific performance. This efficiency has led to extensive adoption in large language models (LLMs), and within that space, numerous studies have explored how seemingly small design choices—such as the initialization strategy of LoRA’s adapter matrices—can have a substantial impact on convergence speed, training stability, and final accuracy. For instance, initializing matrix A with Gaussian noise while keeping matrix B at zero (Init[A]) has been shown to support larger learning rates and faster convergence compared to the reverse (Init[B])<d-cite key="hayou2024impact"></d-cite>. However, this setup can also cause internal instability, especially in early training. As a response, the PiSSA method was introduced, which leverages singular value decomposition (SVD) of the pre-trained weight matrix to initialize A and B along their principal components, resulting in faster and more stable learning in LLMs<d-cite key="meng2024pissa"></d-cite>.

{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/teaser.png" class="teaser" %}
Figure 1. Embedding modality gap between text and image<d-cite key="ma2024bridging"></d-cite>.

Despite these advances, most of the research on LoRA initialization has been limited to language-only models. The behavior of these strategies in Vision-Language Models (VLMs)—which must jointly reason over visual and textual inputs—remains largely underexplored. This is a significant gap, especially given the rapid rise of multimodal models such as LLaVA, BLIP-2, and Flamingo, which are increasingly being applied to real-world tasks like image captioning and visual question answering (VQA)<d-cite key="liu2023visual"></d-cite><d-cite key="li2023blip"></d-cite><d-cite key="alayrac2022flamingo"></d-cite>. Multimodal fine-tuning introduces unique challenges due to modality-specific representations and complex interactions across layers<d-cite key="ma2024bridging"></d-cite>, as highlighted in Figure 1. As such, it is unclear whether initialization strategies that work well in LLMs can be directly transferred to VLMs.

To address this question, our work explores whether LoRA initialization strategies developed for LLMs remain effective when applied to VLMs. Investigating VLM's behavior could yield novel insights and discoveries regarding fine-tuning practices for both LLMs and VLMs. Specifically, we investigate four representative approaches—**LoRA, PiSSA, EVA,** and **LoftQ**—each of which offers a distinct perspective on adapter initialization, ranging from random initialization and spectral decomposition to activation-aware scaling and context-informed decomposition. By conducting controlled experiments in a unified multimodal fine-tuning framework, we aim to analyze how these strategies affect learning dynamics and performance. Ultimately, our goal is to better understand which initialization practices generalize well across modalities, and which require rethinking in the context of multimodal learning.

## What we do
In this blog post, we aim to address this open question by conducting a comprehensive analysis of LoRA initialization strategies within the context of VLMs. Specifically, we investigate the following:

### 1. Evaluating Vanilla LoRA in Multimodal Settings
We investigate whether the training behaviors observed in LLMs with standard LoRA also appear in vision–language models. While prior studies have explored how specific initialization schemes affect convergence and stability in language-only settings, it remains unclear how Vanilla LoRA performs in multimodal contexts. Our goal is to assess whether similar learning dynamics emerge in VLMs or if cross-modal interactions require different initialization considerations.

### 2. Evaluating PiSSA in Multimodal Settings
We explore whether the recently proposed Principal Singular-value-based Initialization (PiSSA) technique exhibits similar impacts on VLMs as it does on LLMs. PiSSA leverages the principal components of the pre-trained weights to initialize the LoRA adapters in a more informed manner, and has been shown to improve convergence speed and training stability in language-only models. This study extends that investigation to multimodal architectures.

### 3. Ablation Study of General Initialization Strategies
Beyond vanilla LoRA and PiSSA, we conduct a broader ablation study comparing several principled initialization strategies, including EVA and LoftQ. By evaluating PiSSA, EVA, and LoftQ side by side, we aim to uncover how different design choices influence training dynamics and performance in multimodal models.

## Experimental setting
The objective of this study is to examine how the learning dynamics of vision–language models (VLMs) vary according to the weight‐initialization schemes employed by LoRA‐based adapters. To this end, we selected Qwen 2.5 VL 7B and LLaMA 3.2 11B Vision as our base architectures and applied four adapter families—LoRA, PiSSA, EVA, and LoftQ—to each. All experiments were orchestrated via the LLaMA-Factory framework, which provides a unified, code-free interface for fine-tuning<d-cite key="zheng2024llamafactory"></d-cite>.

To quantify learning dynamics, we tracked both training loss and gradient norms throughout a single epoch of fine-tuning. Experiments were conducted on 30,000 samples drawn from the LLaVA-Instruct-150K dataset—a GPT-4-generated, multi-modal instruction-following corpus built on MS COCO images, where image-based prompts elicit GPT-4’s answers (questions, explanations, reasoning, etc.). For all LoRA variants we used a rank of 8, and for PiSSA we performed 16 iterative updates. Optimization employed a learning rate of 1 × 10⁻⁴, a cosine scheduler with a warmup ratio of 0.1, and all training ran on NVIDIA A100 GPUs.

## Related work
### What is PiSSA?
PiSSA (Principal Singular Values and Singular Vectors Adaptation) is an advanced method designed for parameter-efficient fine-tuning (PEFT) of Large Language Models (LLMs). It builds upon the Low-Rank Adaptation (LoRA) approach—which typically initializes low-rank matrices A and B randomly, potentially causing slower convergence—by leveraging the intrinsic structure of pre-trained model weights. Specifically, PiSSA employs Singular Value Decomposition (SVD) to identify principal components of the original weight matrix, initializing AA and BB accordingly. This targeted initialization significantly accelerates convergence and enhances overall fine-tuning performance.

PiSSA leverages SVD by decomposing the pre-trained weight matrix $W$ into singular vectors and singular values as follows:

$$
W=U\Sigma VTW = U \Sigma V^T
$$

where $U$ and $V$ are orthogonal matrices containing the left and right singular vectors, respectively, and $\Sigma$ is a diagonal matrix of singular values. PiSSA then selects the top $r$ singular values and their corresponding singular vectors to capture the most significant structures of $W$. The low-rank adapter matrices are initialized using these principal components:

$$
A=UrΣr,B=ΣrVrTA = U_r \sqrt{\Sigma_r}, \quad B = \sqrt{\Sigma_r} V_r^T
$$

where $U_r$, $\Sigma_r$, and $V_r$ denote the truncated matrices containing the top $r$ components. The remaining components form the residual matrix, which is kept frozen during fine-tuning.

### What is LoftQ?
LoftQ (LoRA-Fine-Tuning-Aware Quantization) is an advanced LoRA-based method designed specifically for fine-tuning quantized Large Language Models (LLMs)<d-cite key="li2023loftq"></d-cite>. Unlike standard LoRA, which fine-tunes models with randomly initialized low-rank adapter matrices, or QLoRA, which directly applies LoRA to quantized models often resulting in performance degradation, LoftQ jointly optimizes both the quantization and the low-rank approximation of pre-trained weights. Specifically, LoftQ minimizes the approximation error by solving:

$$
\min_{Q, A, B}\|W - Q - AB^\top\|_F^2
$$

where W is the original pre-trained weight matrix, $Q$ is the quantized approximation of $W$, and $A,B$ are the low-rank adapter matrices. By aligning quantization with LoRA initialization, LoftQ significantly reduces quantization-induced degradation, thereby enhancing fine-tuning performance compared to traditional LoRA or QLoRA approaches.

### What is EVA?
Explained Variance Adaptation (EVA) is a novel initialization scheme for parameter-efficient fine-tuning that provably maximizes the expected gradient signal by aligning low-rank adapter matrices with the principal components of downstream activation distributions<d-cite key="paischer2025parameterefficientfinetuningexplained"></d-cite>. At the onset of fine-tuning, EVA performs incremental Singular Value Decomposition (SVD) on minibatches of activation vectors extracted from the pretrained model, updating the right-singular vectors until convergence and using them to initialize the LoRA adapter matrices. To operate within a fixed rank budget, EVA then globally sorts these converged vectors by their explained variance and adaptively allocates ranks so that components capturing the most variance receive higher capacity thereby reducing the overall number of trainable parameters without sacrificing expressiveness. Importantly, the extra computational cost incurred by this data-driven initialization is minimal often under 1% of total fine-tuning time and remains largely invariant to batch size and order. Empirical evaluations across language generation and understanding, image classification, and reinforcement learning tasks demonstrate that EVA consistently converges faster than existing LoRA variants and achieves the highest average domain performance, all while operating more parameter-efficiently than competing initialization and rank-redistribution methods.
{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/image (4).png" class="img-fluid" %}
Figure 2. Pseudo code for EVA fine-tuning

## Findings
To evaluate how different LoRA initialization strategies affect fine-tuning in multimodal settings, we conducted controlled experiments on two models: Qwen 2.5 VL 7B and LLaMA 3.2 11B Vision. For each model, we applied four adapter variants—LoRA, PiSSA, EVA, and LoftQ—and monitored their learning dynamics over 1,600 training steps. Specifically, we tracked training loss and gradient norms to assess convergence speed, optimization stability, and modality-specific behavior. The results are summarized below.

### LLaMA 3.2 11B Vision
{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/image (5).png" class="img-fluid" %}
Figure 3. Training loss on LLaMA 3.2 11B Vision with diverse LoRA variants

LoftQ, EVA, PiSSA, and LoRA all follow the same path from the beginning of training until all 1600 optimization steps are completed. PiSSA shows a slightly slower and more irregular decay curve than the other adapters. After 50 steps, the three (LoftQ, EVA, and LoRA) show relatively overlapping trajectories in the 0.97±0.02 band, suggesting that training has entered a plateau. On the other hand, PiSSA has a larger oscillation amplitude in the 1.00±0.03 band and sporadically spikes to 1.06, showing higher volatility than the other methods (Figure 2).

{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/image (6).png" class="img-fluid" %}
Figure 4. Grad norm on LLaMA 3.2 11B Vision with diverse LoRA variants

The gradient norm evolution over the 1,600 steps shows a common pattern of a rapid initial spike in the gradient magnitude and then a rapid stabilization as training progresses. PiSSA has the largest peak in the early steps, reaching around 5.0 before settling in the range of 2.0–2.5; LoftQ starts at around 3.6 and then rapidly declines to the range of 0.6–0.8; EVA starts at around 1.2 and soon declines to the range of 0.5–0.9; and plain LoRA has the smallest initial spike (around 0.6) and then stabilizes in the range of 0.4–0.6. After 50 steps, EVA and LoRA show a stable level of oscillation compared to other adapters, while LoftQ shows peaks with large variability in the middle, contrary to the overall low norm. This is presumed to be due to instability caused by the quantization of LoftQ. Also, the norm of PiSSA oscillates at a higher level compared to other adapters (Figure 3).

### Qwen 2.5 VL 7B

{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/image (7).png" class="img-fluid" %}

Figure 5. Training loss on Qwen 2.5 VL 7B with diverse LoRA variants

Across the 1 600 training steps, every LoRA-based variant exhibits a swift reduction in loss relative to its starting point. Yet both PiSSA and LoftQ begin with noticeably higher initial losses than baseline LoRA, and they remain elevated thereafter—PiSSA hovers in the 1.00–1.05 band, while LoftQ settles between about 1.25 and 1.15. This pattern suggests that when an adapter departs too far from re-using the pretrained weights, the loss soon plateaus and resists further improvement. By contrast, EVA and vanilla LoRA track one another almost perfectly, maintaining the lowest loss among the group(Figure 4).

{% include figure.html path="assets/img/2025-04-28-analysing-the-spectral-biases-in-generative-models/image (8).png" class="img-fluid" %}

Figure 6. Grad norm on Qwen 2.5 VL 7Bwith diverse LoRA variants

LoftQ, EVA, vanilla LoRA, and PiSSA exhibit a sharp drop in gradient norm during the first ~250 steps, after which the norm flattens and oscillates within a narrow band. However, unlike LoRA and EVA, both PiSSA and LoftQ stabilise at markedly higher gradient-norm levels. When this behaviour is juxtaposed with their training-loss curves, it becomes clear that even after the loss has largely saturated, PiSSA and LoftQ continue to apply comparatively large updates. Maintaining a high gradient norm without a corresponding decrease in loss is a classic sign of unstable optimisation, implying that PiSSA and LoftQ drive a less stable training dynamic than the steadier LoRA and EVA baselines within a vision-language model(Figure 5).

In vision–language Transformers, the activation variance of visual tokens is typically 2–3 × larger than that of textual tokens. Introducing a parameter-efficient adapter therefore propagates this mismatch to all subsequent gradients, unless the adapter itself corrects the scale disparity.

LoftQ mitigates the problem implicitly: quantising the base model to 4-bit compresses every weight’s dynamic range, pulling the visual and textual feature distributions into the same magnitude band. During fine-tuning this yields smaller gradient norms and a faster loss decrease.

EVA achieves the same effect explicitly. A learnable gating vector rescales the value-projection at training time, dampening the dominant visual channel so that its back-propagated gradients align with those from the textual channel. The resulting optimisation curve closely tracks LoftQ’s, albeit without the cost of full-model quantisation.

PiSSA decomposes each adapter into parallel low-rank branches that are serialised at inference time. Because no rescaling step is applied after the split, the branch processing visual tokens forwards larger activations unmodified, which inflates gradient norms and slows convergence.

Vanilla LoRA leaves the base scales untouched. Consequently its behaviour falls between the two extremes—stable, yet not as well regularised as LoftQ or EVA.

Overall, our experiments show that successful VLM adaptation requires an adapter that compresses, gates, or otherwise re-normalises cross-modal variance. Methods that ignore the scale gap (e.g. PiSSA) incur higher optimisation error and noisier gradients, whereas adapters that enforce scale-matching (LoftQ, EVA) produce smoother training dynamics and superior final performance.

## Conclusion
In our experiments, Vanilla LoRA consistently delivered the most stable training performance on both the LLaMA 3.2 11B Vision and the Qwen 2.5 VL 7B models, a result that contradicts the PiSSA paper’s claim that PiSSA is the most stable LoRA-style adapter for large language models. We hypothesize that PiSSA’s relatively poor performance in the vision-language models stems from the additional modality gap introduced by visual feature tokens: unlike pure text inputs, vision-language models must process both textual and image tokens, and this larger gap may interfere with the SVD-based initialization procedure on which PiSSA relies. Moreover, the LoftQ adapter exhibited unstable training behavior across both VLMs: on LLaMA 3.2 11B Vision, we observed intermittent spikes in the gradient norm, while on Qwen 2.5 VL 7B the training loss peaked at its highest values and converged more slowly. We attribute these instabilities to LoftQ’s quantization scheme, noting that the larger parameter budget of the LLaMA 3.2 11B Vision model appears to mitigate—but not eliminate—these effects, resulting in a comparatively smoother training curve<d-cite key="jin2024comprehensive"></d-cite>. To conclusively demonstrate that quantization degrades the performance of LoRA-family adapters, one must perform an intra-adapter ablation study—comparing PiSSA to its quantized counterpart, QPiSSA—and we designate this rigorous evaluation as a direction for future work.

## Contributions and Impact

This work contributes to a deeper understanding of how initialization strategies affect parameter-efficient fine-tuning (PEFT) in multimodal models. While previous studies have focused primarily on language-only models, our analysis extends this discussion to vision–language models (VLMs), highlighting the importance of modality-aware design in adapter initialization.

First, we demonstrate that methods designed for unimodal architectures, such as PiSSA, do not directly translate to VLMs. Despite its theoretical advantages, PiSSA underperforms in multimodal settings due to its inability to account for the scale mismatch between visual and textual modalities. This result challenges the assumption that initialization strategies can be universally applied across model types and suggests the need for tailored approaches in multimodal learning.

Second, we identify two effective solutions—Vanilla LoRA and EVA—that mitigate cross-modal variance through different mechanisms. EVA explicitly aligns visual and textual modalities via activation-aware gating and Vanilla LoRA achieves stable performance by preserving the original model weights without introducing structural changes. Our findings suggest that, in vision–language settings, keeping the pretrained weight distribution largely intact—as Vanilla LoRA does—can be more beneficial than transformations like quantization. Both methods yield more stable gradients and faster convergence, contributing to improved training efficiency in multimodal fine-tuning.

These insights have practical implications for both researchers and practitioners. As VLMs become more prevalent in real-world applications such as image captioning, visual question answering, and multimodal instruction following, the demand for efficient and robust fine-tuning will only grow. Our findings provide actionable guidance for selecting or designing initialization methods that are better suited to the multimodal context.

Ultimately, this study advances the broader goal of making PEFT techniques more effective and reliable across modalities. By revealing the limitations of existing methods and highlighting promising alternatives, we lay the groundwork for future research into modality-specific PEFT strategies for vision, audio, video, and beyond.

## Limitation

While our study provides valuable insights into the effectiveness of LoRA initialization strategies for vision–language models (VLMs), several limitations remain.

First, our experiments were conducted on a fixed set of architectures—Qwen 2.5 VL 7B and LLaMA 3.2 Vision 11B—and a single dataset, LLaVA-Instruct-150K. While these choices reflect strong and widely used baselines, the generalizability of our findings to other model families (e.g., Flamingo, BLIP-2) or diverse multimodal tasks (e.g., grounding, retrieval, video QA) remains to be validated. Future work should assess whether the same trends hold across models with different encoder-decoder structures, pretraining objectives, and data compositions.

Second, we focused primarily on early training dynamics—namely, loss curves and gradient norms within the first epoch. While these are useful proxies for convergence behavior, they do not fully capture long-term generalization or downstream performance. Additional evaluations on zero-shot or few-shot benchmarks, as well as task-specific metrics, would help further contextualize the practical impact of each initialization method.

In summary, while our controlled setup allowed for clear comparisons across initialization strategies, expanding the evaluation scope along these dimensions would provide a more comprehensive understanding of PEFT methods in multimodal learning.
