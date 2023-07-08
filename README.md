## Download database

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
```

## Setup directories

```bash
mkdir -p ./trained_model
mkdir -p ./report
```

## Run model training

```bash
python dupr.py
```