# Covid-19 R-0 parameter estimation using the SIR model

Compartmental models aims to model the evolution of infectious diseases separating the population in compartments, sections of the population with certain characteristics.
SIR is the simplest compartmental model that asigns the population in three categories: S(Susceptibles), I(Infectious) and R(Recovered). This proyects tries to determine
the R-0 parameter for some countries whose infectious evolution fits the SIR model.

![italy](</Results/Italy.png>)

# Usage

```python
python SIR_model.py
```
Options:
```python
usage: Visualize your data statistics comparing multiple configurations
       [-h] [-c COUNTRY] [-s SUSCEPTIBLES] [-i INFECTED] [-r RECOVERED]
       [-a A_ESTIMATE] [-b B_ESTIMATE] [--days DAYS]

optional arguments:
  -h, --help            show this help message and exit
  -c COUNTRY, --country COUNTRY
                        Country to analize
  -s SUSCEPTIBLES, --susceptibles SUSCEPTIBLES
                        Number of susceptible people to get infected. Depends
                        on each country's population size
  -i INFECTED, --infected INFECTED
                        Number of infected of day 1
  -r RECOVERED, --recovered RECOVERED
                        Number of recovered people on day 1 (defaults to 0)
  -a A_ESTIMATE, --a_estimate A_ESTIMATE
                        "a" parameter first estimation
  -b B_ESTIMATE, --b_estimate B_ESTIMATE
                        'b' parameter first estimation
  --days DAYS           Number of days to see the infected evolution
```

# Data Source

https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases

