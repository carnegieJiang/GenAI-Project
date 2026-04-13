# 5. Baseline Results

## Baseline setup

The baseline method is the fine-tuned InstructPix2Pix model. This is the correct baseline for our project because our task is prompt-guided style transfer: the input is a source image plus a style instruction, and the output is a stylized image that should preserve the original content. InstructPix2Pix directly supports this same format:

```text
source image + text instruction -> edited image
```

We evaluated the trained baseline model saved on AWS at:

```text
/home/ec2-user/GenAI-Project/model/instructp2p
```

The evaluation used 8 examples from the StyleBooth subset. For each example, we used the source image and style prompt as model input, generated a baseline output image, and compared it against the target stylized image provided by StyleBooth.

The qualitative figures are saved as side-by-side grids in this format:

```text
source image | baseline output | target stylized image
```

## Summary results

| Method | Dataset | Samples | Avg. time / image | Avg. CLIP prompt alignment | Avg. CLIP source-output similarity |
|---|---|---:|---:|---:|---:|
| InstructPix2Pix baseline | StyleBooth subset | 8 | 2.329 sec | 26.2144 | 0.7842 |

## Per-sample results

| Sample / style | Time / image | CLIP prompt alignment | CLIP source-output similarity |
|---|---:|---:|---:|
| Color Field Painting | 2.593 sec | 28.0758 | 0.7184 |
| futuristic-retro futurism | 2.300 sec | 26.6149 | 0.8331 |
| futuristic-retro futurism | 2.289 sec | 27.0916 | 0.8453 |
| Impressionism | 2.289 sec | 24.7200 | 0.8000 |
| sai-comic book | 2.289 sec | 22.1476 | 0.8173 |
| sai-lowpoly | 2.289 sec | 27.6712 | 0.6790 |
| papercraft-papercut collage | 2.290 sec | 25.9982 | 0.8580 |
| futuristic-vaporwave | 2.290 sec | 27.3960 | 0.7227 |

## What the metrics mean

CLIP prompt alignment measures how well the generated output matches the text style prompt. Higher is better. The average CLIP prompt alignment was 26.2144, which suggests the baseline is generally responding to the requested style prompts.

CLIP source-output similarity measures how similar the generated output remains to the original source image in CLIP image-feature space. Higher means more content is preserved. The average source-output similarity was 0.7842, which suggests the generated results usually keep recognizable content from the source image while still changing style.

Timing was also recorded because the rubric asks for meaningful analysis beyond just image examples. The baseline averaged 2.329 seconds per image on the AWS A10G GPU, so it is practical to use as a baseline for a small validation subset.

## Interpretation

These are the expected results for the baseline. InstructPix2Pix is a general instruction-guided image editing model, so it should be able to make visible style changes when given prompts like "restyle this image as Impressionism" or "restyle this image as futuristic-vaporwave." At the same time, because it edits an existing source image rather than generating from scratch, it should preserve much of the source content and layout.

The qualitative grids show this behavior: the baseline generally changes visual style, texture, and color palette while keeping the main structure of the source image recognizable. This matches the project goal of studying the tradeoff between style alignment and content preservation.

The results also show why this is only a baseline. Some outputs preserve content well but may not fully match the target style. Other outputs show stronger stylization but alter details from the original image. This is the expected style-versus-content tradeoff, and it motivates comparing the baseline against our Method 1 diffusion model and later improvements.

## Figures to include

Use the grid images below in the Google Doc/report. Each grid already shows source, baseline output, and target:

```text
results/baseline/grids/000_ColorFieldPainting_00409_grid.png
results/baseline/grids/001_futuristic-retrofuturism_01679_grid.png
results/baseline/grids/002_futuristic-retrofuturism_01824_grid.png
results/baseline/grids/003_Impressionism_02286_grid.png
results/baseline/grids/004_sai-comicbook_03657_grid.png
results/baseline/grids/005_sai-lowpoly_04012_grid.png
results/baseline/grids/006_papercraft-papercutcollage_04506_grid.png
results/baseline/grids/007_futuristic-vaporwave_10476_grid.png
```

## Result files

```text
results/baseline/baseline_results_section.md
results/baseline/baseline_summary.csv
results/baseline/baseline_results.csv
results/baseline/grids/
results/baseline/outputs/
```
