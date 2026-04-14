# 7. Thought-Experiment on Compute

## Hardware used so far

Our GPU work so far has been run on AWS using a `g5.2xlarge` EC2 instance. The machine has:

| Instance | GPU | GPU memory | Used for |
|---|---|---:|---|
| AWS EC2 `g5.2xlarge` | 1x NVIDIA A10G | about 24 GB | StyleBooth setup, InstructPix2Pix baseline training/evaluation, and partial Method 1 training |

For cost conversion, we use an approximate on-demand price of **$1.212 per g5.2xlarge instance-hour** in us-east-1. This means each GPU-hour on this instance costs about **$1.21**. These estimates are not exact AWS billing records, but they are reasonable for reporting compute usage.

## What has actually been run

So far, the completed baseline work is stronger than the Method 1 work:

| Component | Status | Notes |
|---|---|---|
| StyleBooth dataset setup | Completed | Dataset was downloaded/unpacked on AWS and subset metadata was prepared. |
| Baseline training | Completed | InstructPix2Pix was fine-tuned using `data/stylebooth_dataset` with `max_train_steps=10000`. This corresponds to the full StyleBooth dataset path in `src/baseline/run.sh`, not just the 8-example report subset. |
| Baseline evaluation | Completed | The trained baseline was evaluated on 8 examples from the StyleBooth subset for the baseline-results section. |
| Method 1 training | Partial / paused | Method 1 latent diffusion training was started and produced checkpoints/samples, but the run was stopped/paused because it was taking too long. We are not claiming this is a fully completed full-dataset Method 1 training run yet. |
| Method 1 evaluation | Not finalized yet | The group still needs to decide whether to evaluate the partial checkpoint, train Method 1 on a smaller subset, or resume longer training. |

## Estimated GPU hours so far

These estimates include training, evaluation, dataset setup, and basic overhead such as model loading/debugging. They should be treated as approximate GPU-hours.

| Task | What was run | Estimated GPU hours | Estimated cost |
|---|---|---:|---:|
| Dataset setup / StyleBooth prep | Downloading/unpacking StyleBooth, converting/preparing metadata, and moving files around | ~0.5 hr | ~$0.61 |
| Baseline training | Fine-tuned InstructPix2Pix on `data/stylebooth_dataset` to 10,000 steps | ~4.0 hr | ~$4.85 |
| Baseline evaluation | Ran trained InstructPix2Pix on 8 StyleBooth subset examples and computed timing/CLIP metrics | ~0.1 hr | ~$0.12 |
| Method 1 partial training | Started latent diffusion Method 1 training; saved intermediate checkpoints/samples, but did not complete the intended full training plan | ~10.4 hr | ~$12.60 |
| Misc. setup/debugging overhead | Environment setup, model loading tests, small failed/partial runs, monitoring, and data/model path debugging | ~1.0 hr | ~$1.21 |
| **Total so far** |  | **~16.0 GPU hr** | **~$19.39** |

The baseline evaluation itself was cheap compared with training. For the 8-image baseline evaluation, the trained InstructPix2Pix model averaged **2.329 seconds per image** on the AWS A10G GPU. Most of the compute cost came from the baseline training and the partial Method 1 training.

## Why these compute results make sense

This compute pattern matches our experimental plan. The baseline had to be trained first so we could produce baseline results for comparison. Then Method 1 training was started as the next research method, but it was paused because full training was taking too long. That is expected for diffusion-based training: it is much more expensive than running inference on a few validation examples.

It also makes sense that the baseline evaluation was run on a subset rather than the full dataset for the report checkpoint. The goal of the baseline-results section is to show an easy-to-read table/figure with timing, metrics, and interpretation. Running on a small subset gives us enough evidence for a checkpoint/report section without spending unnecessary compute before the final evaluation plan is settled.

## What we should do next

For the report, we should be honest that Method 1 is not fully finished yet. The group has three realistic options:

| Option | Compute implication | Report implication |
|---|---|---|
| Use the partial Method 1 checkpoint | Lowest extra compute | Fastest way to compare against the baseline, but results should be labeled as preliminary. |
| Train Method 1 on a smaller subset | Moderate extra compute | Cleaner experimental story because both training and evaluation scope are controlled. |
| Resume Method 1 full training | Highest extra compute | Stronger final result if time/budget allows, but riskier because it may keep taking too long. |

The safest plan for the report is probably to evaluate the best available Method 1 checkpoint on the same subset used for baseline evaluation, then optionally run a smaller controlled Method 1 training experiment if time allows. That would let us compare baseline vs. Method 1 under the same evaluation setup instead of waiting for an expensive full-dataset Method 1 run.

## If we had an additional $450 in AWS credits

At about **$1.212/hour**, an extra **$450** would buy roughly:

```text
$450 / $1.212 per hour = about 371 g5.2xlarge GPU-hours
```

I would spend the extra credits on controlled experiments that directly support the style-vs-content research question, not just on training longer without a plan.

| Use of extra credits | Approx. budget | Approx. GPU hours | Why this helps |
|---|---:|---:|---|
| Finish or extend Method 1 training | ~$160 | ~132 hr | Either resume the full Method 1 run or train a smaller controlled version until results stabilize. |
| Evaluate baseline and Method 1 on a larger validation subset | ~$70 | ~58 hr | Move beyond 8 examples and produce a stronger metrics table for CLIP alignment and content preservation. |
| Guidance/prompt ablations | ~$80 | ~66 hr | Test how prompt wording and guidance strength affect the style-content tradeoff. |
| Checkpoint comparison for Method 1 | ~$60 | ~50 hr | Compare partial checkpoints to see whether longer training actually improves results. |
| Debugging and rerun buffer | ~$50 | ~41 hr | Leave budget for failed jobs, environment issues, reruns, and final figure generation. |
| Final qualitative result generation | ~$30 | ~25 hr | Generate polished examples across several styles for the final report/poster. |

The highest-value use of extra credits would be to make the Method 1 comparison more reliable. Specifically, we should compare the baseline and Method 1 on the same evaluation subset, then scale up to more examples if the method is working. This directly supports the project goal of analyzing the tradeoff between style alignment and content preservation.

We should not use the extra credits to train a brand-new large model from scratch. That would be outside the project scope and would likely waste the budget. The better plan is to use the credits for controlled Method 1 training, checkpoint comparisons, more validation examples, and clearer final figures.

## Sources / assumptions

- Hardware was checked on the AWS machine: `g5.2xlarge`, 1x NVIDIA A10G GPU.
- Cost estimate uses about `$1.212/hour` for AWS EC2 `g5.2xlarge` Linux on-demand pricing in us-east-1.
- Baseline training used the full dataset path shown in `src/baseline/run.sh`: `/home/ec2-user/GenAI-Project/data/stylebooth_dataset`.
- Baseline evaluation used 8 examples from the StyleBooth subset, as shown in `results/baseline/baseline_results.csv`.
- Method 1 training is described as partial/paused because the run did not finish the intended full training plan.
- GPU-hour estimates are approximate and based on observed checkpoints, sample timestamps, known runs, and setup/debugging activity, not exact AWS billing exports.
