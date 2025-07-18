# Optimization for Machine Learning: Mini-Project (EPFL CS-439)

Project Description: [Read Here](https://github.com/epfml/OptML_course/blob/d61132781e50c3528249cb6325ff9aacd55a26d3/labs/mini-project/miniproject_description.pdf).

Project Report : [Read Here](https://github.com/federicovilla55/optML_mini_project/blob/main/Generalization%20of%20Asynchronous%20SGD%20Variants.pdf)

In order to run the main ASGD variants script, you have to be placed at the root of the project. Then, you can run the following command in the command shell:
```
python -m src.run_tests <asgd_variant_name> <overparam_number>
```

where
- `<asgd_variant_name>` can be `dasgd`, `saasgd`, or `asap_sgd`
- `<overparam_number>` can be `110`, `150`, or `200`

Below is the project structure along with explanations of the key components to help you understand it.

```
.
│
├── src                 # Main folder with code files and saved results
│   ├── config.py
│   ├── core            # Code related to Asynchronous framework
│   ├── data            # Code related to the data set used
│   ├── experiments
│   │   ├── ckpt        # Saved results folder
│   │   ├── asap_sgd.py # ASAP model running script
│   │   ├── dsrc.py    # DASGD model running script 
│   │   └── sasrc.py   # SAASGD model running script
│   ├── models
│   └── run_tests.py    # File running the main script
│ 
└── Generalization of Asynchronous SGD Variants.pdf # Project report
```
