# coeva2

## Create the environment on HPC


```
module purge
module load swenv/default-env/v1.1-20180716-production lang/Python/3.6.4-foss-2018a math/Gurobi/8.1.1-intel-2018a-Python-3.6.4
virtualenv adv
chmod +x adv/bin/activate
. ./adv/bin/activate
pip install --update -r requirements.txt
```

## On macOS

Run before installing the other requirements

```
pip3 install xgboost
```

