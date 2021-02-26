# Unior-NLP @PAN2020_Profiling-Fake-News-Spreaders

UniOR NLP Group code repository for the task at PAN-CLEF 2020 "[Profiling Fake News Spreaders](https://pan.webis.de/clef20/pan20-web/author-profiling.html)".

In that workshop, UniOR NLP Group participated in both English and Spanish subtasks.
We used different machine learning algorithms combined with strictly stylometric features, categories of emojis and a
bunch of lexical features related to the fake news headlines vocabulary.

Here you can find the code used and the submitted models.

## Data

For a complete description of both data and task, please refer to the [Task Website](https://pan.webis.de/clef20/pan20-web/author-profiling.html) or [Zenodo](https://zenodo.org/record/4039435#.YDlBUHVKjeQ).

## How to use

The commands below show how to use the scripts.

The train_fnsp.py script trains the English and Spanish models using the data located at ```input data directory``` and saves the trained models to ```output directory path```.

```
python3 train_fnsp.py -i "input data directory" -o "output directory path"
```

The predict_fnsp.py scripts read the data from an ```input data directory``` and write to an ```output directory``` the predicted labels for each document of the dataset.

```
python3 predict_fnsp.py -i "input directory path" -o "output directory path"
```


## Citing

```
@InProceedings{manna:2020,
  author =              {Raffaele Manna and Antonio Pascucci and Johanna Monti},
  booktitle =           {{CLEF 2020 Labs and Workshops, Notebook Papers}},
  crossref =            {pan:2020},
  editor =              {Linda Cappellato and Carsten Eickhoff and Nicola Ferro and Aur{\'e}lie N{\'e}v{\'e}ol},
  month =               sep,
  publisher =           {CEUR-WS.org},
  title =               {{Profiling Fake News Spreaders Through Stylometry and Lexical Features: UniOR NLP @PAN2020---Notebook for PAN at CLEF 2020}},
  url =                 {http://ceur-ws.org/Vol-2696/},
  year =                2020
}

```
