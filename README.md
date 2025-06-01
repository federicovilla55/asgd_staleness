# Optimization for Machine Learning: Mini-Project (EPFL CS-439)

Project Description: [Read Here](https://github.com/epfml/OptML_course/blob/d61132781e50c3528249cb6325ff9aacd55a26d3/labs/mini-project/miniproject_description.pdf).
Project Report : See below

In order to run the main ASGD variants script, you have to be placed at the root of the project. Then, you can run the following command in the command shell:
```
python -m ASGD.run_tests <asgd_variant_name> <overparam_number>
```

where
- `<asgd_variant_name>` can be `dasgd`, `saasgd`, or `asap_sgd`
- `<overparam_number>` can be `110`, `150`, or `200`

Below is the project structure along with explanations of the key components to help you understand it.

```
.
│
├── ASGD                # Main folder with code files and saved results
│   ├── config.py
│   ├── core            # Code related to Asynchronous framework
│   ├── data            # Code related to the data set used
│   ├── experiments
│   │   ├── ckpt        # Saved results folder
│   │   ├── asap_sgd.py # ASAP model running script
│   │   ├── dasgd.py    # DASGD model running script 
│   │   └── saasgd.py   # SAASGD model running script
│   ├── models          # Code related to other models used
│   └── run_tests.py    # File running the main script
│ 
└── Generalization of Asynchronous SGD Variants.pdf # Project report
```
