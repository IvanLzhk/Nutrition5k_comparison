# Nutrition5k comparison architecture

The diagram below shows the training and evaluation flow starting at
`main.py`. It reflects the current single-split default (`FOLDS = None`);
when folds are configured, the split branch repeats once per fold.

```mermaid
flowchart TD
    entry["main.py<br/>experiment configuration"] --> experiment["CNNExperiment"]
    config["setup.py<br/>dataset and checkpoint paths"] --> experiment

    experiment --> ids["Read train/test dish IDs<br/>validate no overlap"]
    ids --> split["Build split<br/>single split or K-fold"]

    data["Nutrition5k data<br/>metadata CSV + imagery"] --> dataset["Nutrition5kDataset"]
    cache["PNG transform cache<br/>data/cache/nutrition5k"] <--> dataset
    split --> trainData["Training dataset<br/>augmentation enabled"]
    split --> validationData["Validation dataset"]
    ids --> testData["Test dataset"]
    trainData --> trainLoader["DataLoader + weighted sampler"]
    validationData --> validationLoader["DataLoader"]
    testData --> testLoader["DataLoader"]
    dataset --> trainData
    dataset --> validationData
    dataset --> testData

    trainLoader --> trainer["BaseExperiment / train_model"]
    validationLoader --> trainer
    trainer --> model["SimpleCNN"]
    model --> loss["MSELoss on standardized targets"]
    loss --> optimizer["AdamW + cosine scheduler"]
    optimizer --> trainer
    trainer --> stopping["Early stopping<br/>gradient clipping / AMP"]
    trainer --> checkpoints["Checkpoints<br/>.pt weights + .json metadata"]
    resume["Optional resume checkpoint"] --> trainer

    checkpoints --> reload["Load best checkpoint"]
    reload --> evaluator["test_model"]
    testLoader --> evaluator
    evaluator --> metrics["test_metrics.json<br/>MSE, MAE, MAPE, R2, Acc@k"]

    classDef source fill:#e8f1ff,stroke:#3973ac,color:#102a43
    classDef process fill:#eef8ee,stroke:#4d8b57,color:#17351b
    classDef storage fill:#fff4df,stroke:#b7791f,color:#4a2c00
    classDef model fill:#f5eafd,stroke:#805ad5,color:#322153

    class entry,config,data,resume source
    class experiment,ids,split,dataset,trainData,validationData,testData,trainLoader,validationLoader,testLoader,trainer,loss,optimizer,stopping,reload,evaluator process
    class cache,checkpoints,metrics storage
    class model model
```

## Model data path

Each dish contributes one overhead image and zero or more side-angle images.
The custom collator flattens side images and records the owning dish index so
the model can aggregate them back into per-dish features.

```mermaid
flowchart LR
    overhead["Overhead RGB<br/>(B, 3, H, W)"] --> shared["Shared CNN feature extractor<br/>Conv2d + ReLU + pooling"]
    side["Side views<br/>(N, 3, H, W)"] --> shared
    sideIndex["side_dish_indices<br/>(N)"] --> aggregate["Per-dish mean aggregation"]
    shared --> overheadFeature["Overhead feature<br/>(B, C)"]
    shared --> sideFeature["Side-view features<br/>(N, C)"]
    sideFeature --> aggregate
    sideIndex --> aggregate
    aggregate --> dishFeature["Per-dish side feature<br/>(B, C)"]
    overheadFeature --> concat["Concatenate<br/>(B, 2C)"]
    dishFeature --> concat
    concat --> regressor["Linear regressor"]
    regressor --> output["Five predictions<br/>calories, mass, fat, carbs, protein"]
```

## Persistent outputs

- `data/cache/nutrition5k/*.png`: transformed image cache entries.
- `checkpoints/simple_cnn/<timestamp>/<split>/simple_cnn_latest.pt`: latest
  training state.
- `checkpoints/simple_cnn/<timestamp>/<split>/simple_cnn_best.pt`: best
  validation checkpoint.
- Sibling `.json` files: readable checkpoint metadata and history.
- `test_metrics.json`: final test metrics for the selected checkpoint.
