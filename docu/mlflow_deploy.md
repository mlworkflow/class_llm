[mlflow deploy docu](https://www.restack.io/docs/mlflow-knowledge-install-mlflow-remote-server)



verbindung zu azure machine
`ssh -i ~/.ssh/mlflow_key.pem azureuser@172.201.218.136`

ram erhöhen (optional)
```
   32  sudo fallocate -l 2G /swapfile
   33  sudo chmod 600 /swapfile
   34  sudo mkswap /swapfile
   35  sudo swapon /swapfile
   36  sudo bash Miniconda3-latest-Linux-x86_64.sh
```

conda installieren
```
    4  sudo apt update && sudo apt upgrade -y
    9  sudo apt install software-properties-common -y
   11  sudo wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   12  sha256sum Miniconda3-latest-Linux-x86_64.sh
   15  bash Miniconda3-latest-Linux-x86_64.sh
   16  exec bash
   17  conda --version
```

env einrichten
```
conda create -n py311_env python=3.11 -y
conda activate py311_env
pip install --upgrade mlflow
# needed for mlflow to work
pip install protobuf==3.20.*
# posgresql (optional)
conda install -c conda-forge psycopg2
```

mlflow starten
```
# make sure you have write permission in ./ dir
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

ui öffnen
```
http://172.201.218.136:5000
```




