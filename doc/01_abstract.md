# Abstract

SignVerse-2M is a large-scale multilingual pose-native dataset for sign language pose modeling and evaluation. Starting from publicly available multilingual sign language videos, it applies a unified DWPose-based preprocessing pipeline to convert raw videos into 2D pose sequences that can be consumed directly by downstream models. The current release consolidates roughly **two million clips** spanning **25+ sign languages**, while preserving the variability of open-world shooting conditions, camera setups, and signer identities that are often filtered out in laboratory datasets.

Unlike video-text corpora that are optimized primarily for recognition or translation, SignVerse-2M is designed as a **pose-first interface** for multilingual sign generation, cross-lingual transfer, and evaluation. The release includes the data construction pipeline, the stored pose schema, and an initial SignDW Transformer baseline. Our goal is not only to scale sign language data, but to provide a reusable representation layer that connects sign language research more naturally to modern pose-driven generation systems.

## Contributions

- We construct **SignVerse-2M**, a multilingual pose-native corpus that systematically turns public sign language videos into DWPose sequences suitable for generative modeling.
- We position the dataset as a **sign language interface for modern pose-driven generation**, rather than as a side product of raw-video preprocessing.
- We provide **task definitions and a simple SignDW Transformer baseline** to validate practical usability under a shared multilingual evaluation setup.
- We document both the **strengths** of the representation for open-world robustness and its **current limitations** in fine-grained hand shape, non-manual signals, and linguistic completeness.
