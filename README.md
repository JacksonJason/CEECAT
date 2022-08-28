# The Extended Emission Calibration Artefacts Tool (EECAT).

This tool allows the user to generate artefact maps of sky models that they choose. All experiment conditions can be added in the experiment.yaml file in the same format as the example file.


The tool also allows the user to generate validation experiments. Simply run the tool with the validate argument first, once that has completed, which may take up to a few days. Run the tool with the validationImages argument in order to generate the graphs needed for validation.

Arguments:
 * --validate Re-run all baselines.
 * --validationImages Run the imaging from existing files.
 * --justG Run only the linear experiments.
 * --processes How many cores to run on.
 * --experimentConditions The YAML file where all the experiment conditions are.
