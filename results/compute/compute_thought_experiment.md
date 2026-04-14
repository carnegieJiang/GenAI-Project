# 7. Thought-Experiment on Compute

## Hardware used so far

All main GPU work was run on AWS using a `g5.2xlarge` EC2 instance. This instance has:

| Instance | GPU | GPU memory | Notes |
|---|---|---:|---|
| AWS EC2 `g5.2xlarge` | 1x NVIDIA A10G | about 24 GB | Used for baseline training, Method 1 training, and baseline inference/evaluation |

For cost conversion, I use the us-east-1 Linux on-demand estimate of **$1.212 per g5.2xlarge instance-hour**. AWS on-demand pricing charges by instance time, and recent pricing references list `g5.2xlarge` at about **$1.212/hour** in us-east-1.

## Estimated GPU hours so far

These are estimates based on the model checkpoints, saved sample timestamps, and the runs we know were completed. They should be treated as approximate GPU-hours, not exact billing records.

| Task | What was run | Estimated GPU hours | Estimated cost |
|---|---|---:|---:|
| Dataset setup / StyleBooth prep | Downloading/unpacking dataset and preparing the subset metadata | ~0.5 hr | ~$0.61 |
| Baseline training | Fine-tuned InstructPix2Pix baseline to 10,000 steps | ~4.0 hr | ~$4.85 |
| Baseline evaluation | Ran trained baseline on 8 StyleBooth examples and computed CLIP/timing results | ~0.1 hr | ~$0.12 |
| Method 1 training | Latent diffusion Method 1 training runs/checkpoints and sample generation | ~10.4 hr | ~$12.60 |
| **Total so far** |  | **~15.0 GPU hr** | **~$18.18** |

The largest compute cost so far was Method 1 training. Baseline evaluation was small by comparison: the final baseline evaluation averaged **2.329 seconds per image** for 8 examples, so the inference cost itself was negligible compared with training.

## Why these compute results make sense

This compute usage matches the experimental plan. The project first needed a working dataset pipeline and baseline model, then needed baseline results, then used the same AWS GPU setup for Method 1 training. The baseline training and Method 1 training were the main GPU consumers because they require backpropagation and checkpointing. In contrast, dataset preparation and baseline inference mostly involved file I/O or forward passes, so they used much less compute.

The `g5.2xlarge` / A10G setup is a reasonable choice for our project because it is strong enough to run Stable Diffusion / InstructPix2Pix-style models while being much cheaper than larger multi-GPU machines. It is also aligned with our current project scale: we are comparing baselines and early methods, not training a large foundation model from scratch.

## If we had an additional $450 in AWS credits

At about **$1.212/hour**, an extra **$450** would buy roughly:

```text
$450 / $1.212 per hour = about 371 g5.2xlarge GPU-hours
```

I would spend the credits to improve the research goals rather than just run longer training blindly.

| Use of extra credits | Approx. budget | Approx. GPU hours | Why this helps |
|---|---:|---:|---|
| Fuller Method 1 training and checkpoints | ~$170 | ~140 hr | Train Method 1 longer and evaluate whether performance improves or plateaus. |
| Baseline and Method 1 evaluation on a larger validation subset | ~$60 | ~50 hr | Move from 8 qualitative examples to a larger, more reliable validation set with CLIP/content metrics. |
| Ablations on guidance strength / prompt wording | ~$80 | ~66 hr | Test the style-content tradeoff by varying guidance and prompt templates. |
| Additional comparison runs / debugging budget | ~$90 | ~74 hr | Rerun failed jobs, test alternative hyperparameters, and fix implementation issues without risking the whole plan. |
| Final qualitative result generation | ~$50 | ~41 hr | Generate polished final figures across multiple styles for the report/poster. |

The most important use would be evaluating the style-content tradeoff more thoroughly. For example, we could run the baseline and Method 1 on a larger StyleBooth subset and report CLIP prompt alignment against content-preservation metrics. This directly supports the project goal of comparing how different methods balance style alignment with content preservation.

I would not spend the extra credits on training a completely new large model from scratch. That would be outside our project scope and would probably use the budget inefficiently. A better plan is to use the credits for longer controlled training, more validation examples, more ablations, and stronger qualitative figures.

## Sources / assumptions

- Hardware was checked on the AWS machine: `g5.2xlarge`, 1x NVIDIA A10G GPU.
- Cost estimate uses about `$1.212/hour` for AWS EC2 `g5.2xlarge` Linux on-demand pricing in us-east-1.
- GPU-hour estimates are approximate and based on checkpoint/sample timestamps, not exact AWS billing exports.
