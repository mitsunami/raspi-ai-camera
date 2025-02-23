### Setup

```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install pip --upgrade

pip install tensorflow~=2.14.0 model-compression-toolkit~=2.1.0
pip install tensorflow_datasets
pip install imx500-converter[tf]
```

### Re-train model

```bash
python .\train.py
```

### Quantisation and compression

```bash
python .\quantize.py
```

### Conversion for AI Camera

```bash
imxconv-tf -i .\models\mobilenet-quant-rps.keras -o .\models\mobilenet-quant-rps-conv # this needs Java 17
scp .\models\mobilenet-quant-rps-conv\packerOut.zip pi@raspizero2w.local:/home/pi/
```
