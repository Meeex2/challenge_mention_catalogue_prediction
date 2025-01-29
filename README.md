# challenge_mention_catalogue_prediction

## Wedn.21 Jan

### ViT: No training

Evaluation Results:
Hit@1: 0.1875 (15/80)
Hit@3: 0.3250 (26/80)
Hit@5: 0.4125 (33/80)
Hit@10: 0.5250 (42/80)

### ViT: Removed backgrounds

Evaluation Results:
Hit@1: 0.2250 (18/80)
Hit@3: 0.4500 (36/80)
Hit@5: 0.5375 (43/80)
Hit@10: 0.6250 (50/80)

### ViT: Removed backgrounds - google/vit-base-patch16-224-in21k

Evaluation Results:
Hit@1: 0.2875 (23/80)
Hit@3: 0.4625 (37/80)
Hit@5: 0.4875 (39/80)
Hit@10: 0.5750 (46/80)

### CLIP-ViT - openai/clip-vit-base-patch32

Evaluation Results:
Hit@1: 0.1750 (14/80)
Hit@3: 0.3125 (25/80)
Hit@5: 0.4375 (35/80)
Hit@10: 0.5250 (42/80)

### CLIP-ViT - backgrounds removed openai/clip-vit-base-patch32

Evaluation Results:
Hit@1: 0.2125 (17/80)
Hit@3: 0.3375 (27/80)
Hit@5: 0.4625 (37/80)
Hit@10: 0.5250 (42/80)

## Thu.23 Jan

### model: google/vit-base-patch16-224-in21k

Evaluating retrieval performance on original test images...
Evaluation Results:
Hit@1: 0.1125 (9/80)
Hit@3: 0.2750 (22/80)
Hit@5: 0.3625 (29/80)
Hit@10: 0.4625 (37/80)

Evaluating retrieval performance on test images with backgrounds removed...
Evaluation Results:
Hit@1: 0.2875 (23/80)
Hit@3: 0.4625 (37/80)
Hit@5: 0.4875 (39/80)
Hit@10: 0.5625 (45/80)

After contrastive learning:

Evaluating retrieval performance on original test images...
Euation Results:
Hit@1: 0.0000 (0/80)
Hit@3: 0.0000 (0/80)
Hit@5: 0.0000 (0/80)
Hit@10: 0.0000 (0/80)

Evaluating retrieval performance on test images with backgrounds removed...
Evaluation Results:
Hit@1: 0.1625 (13/80)
Hit@3: 0.3125 (25/80)
Hit@5: 0.3500 (28/80)
Hit@10: 0.4375 (35/80)

### model: nomic_embed

Evaluation Results:
Hit@1: 0.1125 (9/80)
Hit@3: 0.2625 (21/80)
Hit@5: 0.3375 (27/80)
Hit@10: 0.4875 (39/80)

### model: resnet-50

Evaluation Results:
Hit@1: 0.1375 (11/80)
Hit@3: 0.2125 (17/80)
Hit@5: 0.2625 (21/80)
Hit@10: 0.3375 (27/80)

## Mon.27 Jan

`resnet_mistral.py`:

- accuracy: 0.00%

`resnet_chatgpt.py`:

- accuracy: 21.25%

`resnet_claude.py`:

- accuracy: 21.25%

`resnet_deepseek.py`:

- accuracy: 18.75%

After removing duplicates from DAM (based on our labeling):
Evaluation Results:
Hit@1: 0.4125 (33/80)
Hit@3: 0.5375 (43/80)
Hit@5: 0.6250 (50/80)
Hit@10: 0.6875 (55/80)

After removing backgrounds from DAM (dark backgrounds) and using test time augmentations:
Evaluation Results:
Hit@1: 0.3625 (29/80)
Hit@3: 0.5250 (42/80)
Hit@5: 0.6000 (48/80)
Hit@10: 0.6750 (54/80)

After removing duplicates from DAM (based on our labeling), test time augmentations and colors hists:
Evaluation Results:
Hit@1: 0.4125 (33/80)
Hit@3: 0.5500 (44/80)
Hit@5: 0.6500 (52/80)
Hit@10: 0.7000 (56/80)

After removing duplicates from DAM (based on our labeling), test time augmentations and colors hists
and using DAM_bg_removed:
Evaluation Results:
Hit@1: 0.3500 (28/80)
Hit@3: 0.5375 (43/80)
Hit@5: 0.5875 (47/80)
Hit@10: 0.6750 (54/80)

## Wed.29 Jan

Evaluating ViT classification capability; it is better than ResNEt the we trained => no need to filter
Classification Accuracy: 0.9556
