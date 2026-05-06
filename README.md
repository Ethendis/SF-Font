# SF-Font

The GB2312_6763 component table is on the prepare dictionary

#Dataset Preparation

1.Build the reference font library

```bash
python character_select.py
```


2. Match reference characters for the character.

```bash
python character_map.py
```

The training data files tree should be (The data examples are shown in directory data_examples/train/)：
```
├──datas
│   ├── data
│   │    ├── font0             # Source font
│   │    │    ├── char0.png
│   │    │    ├── char1.png
│   │    │    ├── char2.png
│   │    │    └── ...
│   │    └── font1             # Target font
│   │         ├── char0.png
│   │         ├── char1.png
│   │         ├── char2.png
│   │         └── ...
│   └── data1
│        ├── font0             # Source font
│        │    ├── char0.png
│        │    ├── char1.png
│        │    ├── char2.png
│        │    └── ...
│        └── font1             # Reference character set of target font
│             ├── char0.png
│             ├── char1.png
│             ├── char2.png
│             └── ...
```
