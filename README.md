# ML Course Notes and implementations

This repo contains notes taken from the online class on coursera Machine Learning taught by Dr. Andrew Ng. This repo also contains
basic implementations of ML algorithms though this is still a work in progress, apologies for bad project structure and programming syntax.

## Running the scripts
Your python scripts should run in an virtualenv, and use a requirements.txt file to indicate the required dependencies. Example requirements.txt:
```
boto==2.39.0
boto3==1.3.0
troposphere==1.5.0
requests==2.9.1
pysocks==1.5.6
```

```
virtualenv --no-site-packages venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

deactivate
```
Once in the virtualenv, suppose you want to execute the script `LinearRegression.py`
```
for Python 2.7
./Linearregression.py

for Python > 2.7
TBD
```

## References:
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
