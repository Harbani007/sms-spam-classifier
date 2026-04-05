# data/

## Dataset: SMS Spam Collection

| Property    | Value                                    |
|-------------|------------------------------------------|
| Name        | SMS Spam Collection v.1                  |
| Source      | UCI Machine Learning Repository          |
| Size        | 5,574 messages (4,827 ham / 747 spam)    |
| Licence     | Creative Commons Attribution 4.0 (CC BY) |

### Download instructions

**Option A — Kaggle**

1. Go to <https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset>
2. Click **Download** → you will get `spam.csv`
3. Place the file here as **`data/spam.csv`**

**Option B — UCI directly**

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
mv SMSSpamCollection data/spam.csv
```

> Note: The UCI file is TSV with no header.  The loader in `src/train.py`
> handles both the Kaggle CSV variant (columns `v1`, `v2`) and the
> headerless TSV variant automatically.

### Expected format after placement

```
data/
└── spam.csv   ← your dataset file lives here
```

The loader in `src/train.py` will auto-detect the separator and normalise
column names.  It handles all of these common variants:

| Columns present       | Dataset variant          |
|-----------------------|--------------------------|
| `v1`, `v2`            | Kaggle default           |
| `label`, `text`       | Pre-cleaned variant      |
| `label`, `message`    | Alternative CSV          |
| `Category`, `Message` | Another common variant   |
