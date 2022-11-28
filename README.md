# The Compact and Extended Emission Calibration Artefacts Tool (CEECAT).

This tool allows the user to generate artefact maps of sky models that they choose. All experiment conditions can be added in the experiment.yaml file in the same format as the example file. To run the tool in experiment mode use the ***--experimentConditions*** parameter combined with the experiement yaml file.


The tool also allows the user to generate validation experiments. Simply run the tool with the ***-validate*** argument first, once that has completed, which may take up to a few days. Run the tool with the ***--validationImages*** argument in order to generate the graphs needed for validation.

Arguments:
 * --validate Re-run all baselines.
 * --validationImages Run the imaging from existing files.
 * --justG Run only the linear experiments.
 * --processes How many cores to run the validation on. The more cores used, the faster it will run but it will consume more computer resources. We reccomend running on a server if you intend to use most of the cores. The default value of this parameter is ***all cores***.
 * --experimentConditions The YAML file where all the experiment conditions are stored.

 The ***crashReport.txt*** file should contain any information on program crashes that may have occured.

 ### Requirements
* matplotlib==3.5.1
* numpy==1.22.3
* PyYAML==6.0
* tqdm==4.64.0
* argparse==1.4.0

 For any questions or comments please direct them to [Jason Jackson](mailto:ajsnpjackson@gmail.com).
