# Turkish Sentiment Analysis with BERT

Classifies Turkish product reviews as negative, neutral or positive, using a Turkish BERT model as a feature extractor and a small scikit-learn neural network on top. It is a compact example for anyone exploring sentiment analysis in Turkish.

> **Status:** archived. Built in 2021. Kept for reference, not maintained.

## What it does

- Reads labeled product reviews from `csvjson.json` (review text in `Görüş`, label in `Durum`) and maps the labels `Olumsuz`, `Tarafsız` and `Olumlu` to 0, 1 and 2 (negative, neutral, positive).
- Encodes each review with [`dbmdz/bert-base-turkish-128k-uncased`](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased) and uses the first-token (`[CLS]`) vector of the last hidden layer as its features. BERT itself is not fine-tuned.
- Trains a scikit-learn `MLPClassifier` (hidden layers of 128 and 64, tanh, Adam, early stopping) on a random train/test split and prints a classification report.
- Saves the trained classifier to `finalized_model.sav` with pickle. The file is committed.
- `test_model.py` reloads the saved classifier, scores it on a split of the first 1,000 reviews and prints the run time.

The original README reports an accuracy of 0.71. The run output itself is not committed.

## Tech stack

- Python
- Hugging Face Transformers and PyTorch (BERT feature extraction)
- scikit-learn (`MLPClassifier`, metrics, train/test split)

## Running it locally

There is no requirements file. The imports need:

```bash
pip install torch transformers scikit-learn
```

Before running `python main.py` (train) or `python test_model.py` (evaluate):

- Both scripts open the dataset from an absolute path on the original machine. Change it to `csvjson.json`.
- The code unpacks the BERT output as a tuple, which matches transformers 3.x. On transformers 4 or later, load the model with `AutoModel.from_pretrained(..., return_dict=False)`, otherwise feature extraction fails for every review.
- `finalized_model.sav` is a pickle file. Loading a pickle can run arbitrary code, so only load one you trust.

## Credits

- Dataset: [Duygu Analizi Veri Seti (Olumlu/Olumsuz/Tarafsız)](https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari), Turkish reviews collected from electronics stores, on Kaggle
- Model: [dbmdz/bert-base-turkish-128k-uncased](https://huggingface.co/dbmdz/bert-base-turkish-128k-uncased)
- Approach based on the [BERT Features notebook](https://github.com/akoksal/BERT-Sentiment-Analysis-Turkish/blob/master/PyIstanbul%20Notebooks/BERT%20Features.ipynb) from BERT-Sentiment-Analysis-Turkish

## License

MIT, see [LICENSE](LICENSE). The review data in `csvjson.json` comes from the Kaggle dataset above, which is published under CC0.
