# DETR Training Container — Components

This page documents the main components of the DETR training codebase and how they interact.

At a high level, the training flow is:

```mermaid
%%{init: {
  "theme": "dark",
  "themeVariables": {
    "lineColor": "#ffffff",
    "textColor": "#f5f5f5",
    "primaryTextColor": "#f5f5f5",
    "secondaryTextColor": "#f5f5f5",
    "tertiaryTextColor": "#f5f5f5",
    "noteTextColor": "#f5f5f5"
  },
  "flowchart": {
    "curve": "linear"
  }
}}%%
flowchart LR
  %% Force white arrow/link strokes, while keeping labels readable in dark theme.
  linkStyle default stroke:#ffffff,stroke-width:2px,color:#f5f5f5

  subgraph CLI["CLI entrypoints"]
    mainpy["main.py\n(train / eval CLI)"]
    submitit["run_with_submitit.py\n(Slurm/submitit launcher)"]
  end

  subgraph Engine["Training engine"]
    enginepy["engine.py\n(train_one_epoch / evaluate)"]
  end

  subgraph Models["Model definitions"]
    detr["models/detr.py\n(DETR model + criterion)"]
    backbone["models/backbone.py\n(ResNet backbone wrapper)"]
    transformer["models/transformer.py\n(Encoder/decoder)"]
    segm["models/segmentation.py\n(mask/panoptic heads)"]
    matcher["models/matcher.py\n(Hungarian matcher)"]
    posenc["models/position_encoding.py\n(positional encodings)"]
  end

  subgraph Data["Datasets & transforms"]
    coco["datasets/coco.py\n(COCO detection)"]
    pan["datasets/coco_panoptic.py\n(COCO panoptic)"]
    tfs["datasets/transforms.py\n(augmentations)"]
    evalc["datasets/coco_eval.py\n(COCO eval)"]
    paneval["datasets/panoptic_eval.py\n(panoptic eval)"]
  end

  subgraph Utils["Utilities"]
    misc["util/misc.py\n(distributed utils, logging)"]
    boxops["util/box_ops.py\n(box geometry)"]
    plot["util/plot_utils.py\n(visualization helpers)"]
  end

  mainpy --> enginepy
  submitit --> mainpy

  enginepy --> detr
  enginepy --> coco
  enginepy --> pan
  enginepy --> evalc
  enginepy --> paneval
  enginepy --> misc

  detr --> backbone
  detr --> transformer
  detr --> matcher
  detr --> segm
  transformer --> posenc

  coco --> tfs
  pan --> tfs
  evalc --> boxops
  segm --> boxops
  plot --> detr
```

> Dark-theme note: the Mermaid init block above forces **arrow/link stroke color to white** and uses light text colors so edges and labels remain readable.

## Notes

- If your docs renderer applies additional CSS to Mermaid SVGs, this diagram still forces edge strokes via `linkStyle default`.
- If you add more edges and want per-edge overrides, you can append additional `linkStyle <index> ...` lines.
