# coeva2

## Create the environment on HPC


```
module purge
module restore python3
virtualenv adv
chmod +x adv/bin/activate
. ./adv/bin/activate
pip install -r requirements.txt
```