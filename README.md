# CS643-PA2

Link to Container: https://hub.docker.com/r/dea1013/cs643-pa2 

## Files
- app.py: Python program for running model
- evaluate.py: Contains code for evaluating models
- preprocess.py: Contains code for preprocessing the dataset
- train.py: Python program to train model
- model: Directory for loading model
- Dockerfile: Docker file used to create docker image

## Setup Instructions

### Training

#### S3 Bucket
- Create S3 bucket with the following name: dilip-anand-cs643-pa2
- Upload following files:
  - TrainingDataset.csv
  - train.py
  - preprocess.py
- Create folder “logs”

#### EMR Cluster Specs
- Create EMR cluster with following steps (if not specified, assume default):
  - Amazon EMR Release: emr-6.10.0
  - Application Bundle: spark
  - Instance Groups: Primary (m5.xlarge), Core (m5.xlarge), Task (m5.xlarge)
  - Provisioning Configuration: Core size: 1 instance, Task size: 4 instances
  - Logs Location: s3://dilip-anand-cs643-pa2/logs/
  - Amazon EC2 Key Pair: vockey
  - Service Role: EMR_DefaultRole
  - Instance Profile: EMR_EC2_DefaultRole

#### EMR Launch Step
- Add a step to the EMR cluster (if not specified, assume default):
  - Type: Spark Application
  - Deploy Mode: Cluster Mode
  - Application Location: s3://dilip-anand-cs643-pa2/trainEMR.py
  - Arguments:
    - --py-files s3://dilip-anand-cs643-pa2/preprocess.py

#### Downloading the Model
- Create and SSH into Ubuntu Linux EC2 instance
- Ensure that AWS credentials are up to date
- Run the following:
  `aws s3 cp s3://dilip-anand-cs643-pa2/model model --recursive`

#### Cleaning Up
- Delete EMR cluster, S3 bucket, and EC2 instance (if no longer needed)

### Running Application Without Docker

#### Set Up Directory
- Clone repository: `git clone https://github.com/dea1013/CS643-PA2.git`
- Create a directory and download the following files into the directory:
  - app.py
  - preprocess.py
  - evaluate.py
  - TestDataset.csv
  - model
- cd into the directory

#### Set Up Environment
- Create and activate virtual environment: \
  `sudo apt update -y` \
  `sudo apt upgrade -y` \
  `sudo apt install python3.10-venv` \
  `python3 -m venv venv` \
  `source venv/bin/activate` \
  `pip install numpy` \
  `pip install pyspark` \
  `sudo apt-get install default-jdk -y` \
  `export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`

#### Run App
- Run: `python3 app.py TestDataset.csv`
  - The following was used for the validation set: `python3 app.py ValidationDataset.csv`
- The application prints out the F1 score of the model

### Building Docker

#### Set Up Environment
- Run: \
  `sudo apt update -y` \
  `sudo apt upgrade -y` \
  `sudo apt install docker.io` \
  `sudo apt install python3.10-venv`

#### Build Docker
- Clone repository: `git clone https://github.com/dea1013/CS643-PA2.git`
- Create and move files to new directory:
  - app.py
  - preprocess.py
  - evaluate.py
  - model
  - Dockerfile
- cd into directory
- Run: \
  `python3 -m venv venv` \
  `source venv/bin/activate` \
  `pip install numpy` \
  `pip install pyspark` \
  `python3 -m pip freeze > requirements.txt` \
  `deactivate` \
  `rm -rf venv` \
  `sudo docker build --tag dea1013/cs643-pa2 .`

#### Push Docker
- Run: \
  `sudo docker login` \
  `sudo docker push dea1013/cs643-pa2:latest`
  
### Running Application With Docker

#### Set Up Environment
- Run: \
  `sudo apt update -y` \
  `sudo apt upgrade -y` \
  `sudo apt install docker.io`

#### Pull Docker
- Run: `sudo docker pull dea1013/cs643-pa2:latest`

#### Run Docker
- Run: `sudo docker run -v $(pwd)/TestDataset.csv:/app/TestDataset.csv dea1013/cs643-pa2:latest TestDataset.csv`
  - The following was used for the validation set: `sudo docker run -v $(pwd)/ValidationDataset.csv:/app/ValidationDataset.csv dea1013/cs643-pa2:latest ValidationDataset.csv`
