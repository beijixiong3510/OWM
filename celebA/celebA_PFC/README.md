
## How to Run the Code
System sequentially learn all 40 different, context-specific mapping rules with a single classifier (91.43%).

```
python train_cdp_OWM.py
```

## Context
The contextual information was the embedding vectors of the corresponding task names trained by [gensim](https://radimrehurek.com/gensim/)  ([wordvet.mat](https://github.com/beijixiong3510/OWM/blob/master/celebA/celebA_PFC/wordvet.mat)).

