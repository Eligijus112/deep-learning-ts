# Activating the virtual environment

For the creation of virtual environments (VE) anaconda is used. You can download anaconda here: 
https://www.anaconda.com/distribution/

After installing anaconda into your system follow the steps bellow to work in a virtual environment.

Creation of the VE:
```
conda create python=3.7 --name deep-ts
```

Activating the VE:
```
conda activate deep-ts
```

Installing all the packages from the **requirements.txt** file to the virtual environment:
```
pip install -r requirements.txt
```

If you are using Microsoft Visual Studio code there may be some additional pop ups indiciating that some packages should be installed (linter or ikernel). 

# Time series data

The data is taken from: https://www.kaggle.com/robikscube/hourly-energy-consumption. The data is an hourly time series regarding the power consumption (in MW) in the Dayton region. The data spans from 2004-10-01 to 2018-08-03 (**n=121271**)