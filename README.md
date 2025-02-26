# -DeepSeek-R1-Distill-Qwen-1.5B-PRUNING_Depth_Last_Layers


# Depth Pruning Effects on Transformer Models and an experiment on DeepSeek-R1-Distill-Qwen-1.5B

## Overview
This project investigates the impact of depth pruning on transformer-based language models by comparing performance metrics before and after pruning. The analysis focuses **DeepSeek-R1-Distill-Qwen-1.5B**. Depth pruning, which involves removing the deepest layers of a model, is explored as a means to reduce computational overhead while assessing the trade-offs in task performance.

## Introduction
Transformer models have revolutionized natural language processing (NLP) with their ability to understand and generate human language. However, the significant computational resources required for these models have prompted research into efficient model compression techniques. Depth pruning is one such approach where the deepest layers—often responsible for nuanced contextual understanding—are removed, leading to a simpler and faster model.

## Pruning:
In language models, pruning refers to the process of identifying and removing parts of the network—such as individual neurons, attention heads, or even entire layers—that contribute little to its overall performance. This technique is especially valuable in large transformer-based models where efficiency and speed are paramount.

### How Pruning Works in Language Models
Language models, like GPT or BERT, are composed of multiple transformer layers that process text in a hierarchical manner. Each layer builds on the previous one:

### Early Layers: These layers tend to learn general language patterns, such as syntax and basic semantics. They form the backbone of the model’s ability to understand language.
Later Layers: These layers become increasingly specialized, fine-tuning the model's output to match the training data. They might overfit to the specific distribution or nuances of the training set.
Why Pruning the Last Layers Yields Better Results
Preservation of Core Representations:
Since the early layers capture the fundamental structure and semantics of language, they are crucial for general understanding. By focusing pruning efforts on the later layers, you keep the robust, general-purpose features intact while removing redundant or overly specialized parameters.

### Reduction of Overfitting:
The last layers are more likely to overfit to the training data, capturing noise or specific patterns that don’t generalize well to new inputs. Pruning these layers can mitigate overfitting, potentially improving performance on downstream tasks or when the model is applied in different contexts.

### Efficiency Gains:
Later layers in language models often contain a high number of parameters, which increases the computational cost during inference. Removing unnecessary parameters from these layers not only speeds up processing but also reduces the model’s memory footprint, making it more deployable in resource-constrained environments.

### Task-Specific Specialization vs. Generalization:
In many practical applications, especially when fine-tuning a pre-trained language model for a specific task, it can be beneficial to reduce the over-specialization of the final layers. Pruning helps force the model to rely more on the general, robust features learned in the early layers, leading to better generalization on new tasks.

This project evaluates how depth pruning affects performance across tasks with varying demands:
- **BoolQ:** Focuses on boolean (true/false) reading comprehension.
- **Lambada (OpenAI & Standard):** Measures the ability to predict the final word in a passage, thus requiring deep contextual and long-term coherence.
- **ARC Easy:** Tests reasoning abilities and common-sense inference based on general knowledge.

## Objectives
- **Evaluate Pruning Impact:** Determine how depth pruning affects model performance on tasks requiring different levels of language comprehension.
- **Benchmark Analysis:** Compare metrics such as accuracy and perplexity on BoolQ, Lambada, and ARC Easy.
- **Extract Insights:** Understand the trade-offs between reduced model depth (and thus efficiency) and the ability to perform complex reasoning.

## Experiments & Results

### Llama-3.2 Model
- **BoolQ:**
  - *Base Model Accuracy:* 0.64  
  - *Pruned Model Accuracy:* 0.63  
  - **Insight:** The minimal performance drop indicates that answering simple boolean questions primarily relies on fundamental language understanding, which is mostly preserved despite pruning.

- **Lambada (OpenAI and Standard):**
  - *Lambada OpenAI Base Accuracy:* 0.62 → *Pruned:* 0.46  
  - *Lambada Standard Base Accuracy:* 0.53 → *Pruned:* 0.11  
  - **Insight:** The significant reduction in accuracy for both Lambada variants suggests that the deep contextual and long-term coherence required for these tasks is highly sensitive to the removal of model layers.

- **ARC Easy:**
  - *Base Model Accuracy:* 0.65  
  - *Pruned Model Accuracy:* 0.46  
  - **Insight:** The performance decline on ARC Easy reflects the model’s reduced capacity for reasoning and maintaining logical connections, which are critical for answering general knowledge questions.

### DeepSeek-R1-Distill-Qwen-1.5B Model
- **BoolQ:**
  - *Base Model Accuracy:* 0.68  
  - *Pruned Model Accuracy:* 0.44  
  - **Insight:** Although there is a noticeable drop, the retained performance indicates that the fundamental language understanding for boolean questions is still somewhat functional despite the pruning.

- **Lambada (OpenAI and Standard):**
  - *Lambada OpenAI Base Accuracy:* 0.34 → *Pruned:* 0.00  
  - *Lambada Standard Base Accuracy:* 0.29 → *Pruned:* 0.00  
  - **Insight:** The complete collapse in performance (accuracy dropping to 0.00) and the dramatic increase in perplexity illustrate that the pruned model loses its ability to maintain the necessary contextual understanding for these tasks.

- **ARC Easy:**
  - *Base Model Accuracy:* 0.62  
  - *Pruned Model Accuracy:* 0.26  
  - **Insight:** The steep drop in ARC Easy performance further underscores how depth pruning compromises the model’s reasoning capabilities and common-sense inference.

## Insights & Conclusions
- **Task Sensitivity:**  
  Tasks requiring deep contextual reasoning and long-term dependency tracking—such as Lambada and ARC Easy—are more adversely affected by depth pruning. In contrast, BoolQ, which relies more on basic language comprehension, exhibits relatively less degradation.

- **Implications of Depth Pruning:**  
  The experiments demonstrate that while depth pruning can simplify model architecture and potentially improve efficiency, it does so at the cost of crucial layers that underpin complex language understanding. This aligns with the findings in *"What Matters in Transformers? Not All Attention is Needed"* ([arXiv:2406.15786](https://arxiv.org/abs/2406.15786)), which suggest that the deepest layers are critical for achieving optimal performance on challenging tasks.

- **Trade-off Between Efficiency and Performance:**  
  The study highlights the inherent trade-offs in model pruning: reducing computational cost and speeding up inference may lead to significant performance drops on tasks that require nuanced comprehension and long-term coherence. The choice of pruning strategy must therefore balance efficiency with the desired level of task performance.

This project provides valuable insights into transformer model pruning strategies, showcasing their advantages and limitations in real-world NLP applications.


