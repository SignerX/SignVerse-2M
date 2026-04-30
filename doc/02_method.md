## Data Pipeline

SignVerse-2M is not defined by a single export script. It is built as a staged processing system with manifest-driven sample management, subtitle structuring, video acquisition, pose extraction, status tracking, and sharded publication. The current repository metadata covers **39,196 videos** and approximately **2M segments**.

![Overview of the SignVerse-2M data processing pipeline](./static/images/pipeline.png)

The current implementation decodes videos at **24 FPS**, extracts body, hand, and face keypoints with DWPose, and stores the result in a per-video `poses.npz` package. This design supports resumable processing, partial reruns, and large-scale corpus maintenance instead of assuming a one-shot offline export.

## Corpus Structure

- The release preserves the heterogeneity of open-world sign language videos instead of normalizing everything into a studio-style distribution.
- Each `video_id` is associated with subtitles, structured caption JSON, pose outputs, and completion markers.
- A global lightweight state index tracks acquisition, subtitle, download, processing, and upload status across the corpus.
- The long-tail language distribution inherited from YouTube-SL-25 is treated as a core benchmark property rather than an incidental artifact.

![Distribution of content hours across sign languages in YouTube-SL-25](./static/images/language_hours.png)

## Released Pose Schema

The released `poses.npz` format is **person-centric**. Each frame payload records the number of detected signers and groups body, face, left-hand, and right-hand keypoints together with confidence scores under each detected person. This makes the stored representation easier to serialize, inspect, and reuse downstream than a flattened OpenPose-style array.

<img src="./static/images/poses_schema.png" alt="Schema of the released poses.npz payload" style="width:50%; max-width:100%; height:auto; display:block; margin:0 auto;">
