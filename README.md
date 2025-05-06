# A Comparative Study of Controlled Text Generation Systems Using Level-Playing-Field Evaluation Principles
This repository contains the code, datasets, and results for a comparative study of Controlled Text Generation (CTG) systems, conducted using Level-Playing-Field (LPF) evaluation principles. 

The study addresses the lack of standardization in evaluating Controlled Text Generation (CTG) techniques, which has made it difficult to fairly compare different techniques. Variations in datasets, evaluation methods, and metrics across studies have led to inconsistent and sometimes misleading conclusions about model effectiveness.

## Key Contributions

1. **Systematic Selection of Techniques**  
   The paper identifies a broad set of CTG techniques by searching relevant literature, filtering down to methods that focus on key control attributes like sentiment, topic, and keyword use. Only open-source, accessible systems were included, ensuring replicability and transparency.

2. **Diverse Evaluation Metrics**  
   To provide a holistic performance picture, the study employs multiple evaluation metrics: diversity (*Distinct-n*), fluency (*SLOR*), and control effectiveness (*CE*), using three classifiers to avoid single-metric bias.

3. **Bias Mitigation**  
   The LPF approach addresses biases in system, metric, and data selection, and standardises output generation and post-processing to avoid spurious variations in results.

---

This repository includes the data, selected models, and evaluation scripts needed to reproduce the LPF assessment framework, facilitating fair and comprehensive evaluation of CTG techniques across the research community.

 
## Folder structure
```
├─ bash             # all bash files to execute the experiments
├─ data             # all datasets used
├─ plots
├─ requirements     # requirements files for the environment required by the CTG techniques
├─ results          # contains all generated texts, processed texts and evaluation results, 
│   │               # for each technique for each control attribute
│   ├─ keywords
│   ├─ multiple
│   ├─ sentiment
│   └─ topic
├─ scripts
│   └─ src
│       ├─ config
│       └─ models
│           ├─ bolt
│           ├─ cat_paw
│           ├─ ctg_bart
│           ├─ ctrl
│           ├─ dexperts
│           ├─ discup
│           ├─ gedi
│           ├─ multi_ctg
│           └─ prior_control
└─ tables

```

## Envirnment creation
To create the required conda environments for running CTG systems:

```
sh bash/environment_creation.sh
```


## Text generation
To generate text using a specific CTG model:

```
sh bash/<model_name>_generation.sh
```

For example, ```sh bash/pplm_generation.sh```.

## Texts evaluation
To evaluate the outputs:

```
sh bash/<model_name>_evaluation.sh
```

For example, ```sh bash/gedi_evaluation.sh```.

## Contact

For questions or contributions, please open an issue or contact the corresponding author [michela.lorandi2@mail.dcu.ie].


