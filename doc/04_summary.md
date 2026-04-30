## Why Pose-Native

Existing large-scale sign language resources usually keep the supervision at the level of raw video and text. That format is valuable for semantic tasks, but it is a poor fit for open-world sign generation, where nuisance factors such as clothing, background, framing, and recording conditions can dominate the input space. SignVerse-2M addresses this by shifting the interface into a unified DWPose representation, so that multilingual modeling and comparison can happen in a more controlled motion space.

The project is motivated by a practical observation from the broader generation community: many recent pose-guided image and video systems already converge on standardized keypoint interfaces such as DWPose. A sign-language dataset that is released directly in this space is therefore easier to reuse, easier to compare across methods, and easier to connect to downstream rendering systems than another raw video-text corpus with custom preprocessing.

## Contributions

- We construct **SignVerse-2M**, a multilingual pose-native corpus that systematically turns public sign language videos into DWPose sequences suitable for generative modeling.
- We position the dataset as a **sign language interface for modern pose-driven generation**, rather than as a side product of raw-video preprocessing.
- We provide **task definitions and a simple SignDW Transformer baseline** to validate practical usability under a shared multilingual evaluation setup.
- We document both the **strengths** of the representation for open-world robustness and its **current limitations** in fine-grained hand shape, non-manual signals, and linguistic completeness.

<!--
## Current Limitations

The current release also has clear boundaries, and these are part of the intended interpretation of the benchmark:

- DWPose is a strong and practical representation, but it is still weaker than raw video for very fine-grained hand shape, finger articulation, and subtle non-manual features.
- The dataset is designed for motion modeling and evaluation, not as a complete linguistic representation of all sign content.
- Language coverage is broad but long-tailed, so transfer and benchmark difficulty vary substantially across languages.
- The qualitative renderer interface demonstrates compatibility, but it is not yet a complete end-to-end human evaluation protocol.

In that sense, SignVerse-2M should be read as infrastructure for future work: a large-scale multilingual pose-native base that makes later modeling, evaluation, and visualization more comparable.
-->
