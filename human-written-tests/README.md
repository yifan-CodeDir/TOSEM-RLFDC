# Human-written Test Selection (RQ1, RQ4)

## Precomputed Simulation Results

The precomputed results are available in the directory `output/`.
- `tests`: The index of the test case (following the order in the coverage matrix) selected at each iteration
- `ranks`: The rank of faulty methods at each iteration
- `fitness_history`: The fitness of the selected test cases at each iteration
- `full_ranks`: The rank of faulty methods when using all available test cases

## Model checkpoints
We provide the trained RL model under the **model** directory
- `./model/cross-version`: Models trained in cross-version scenario (five fold for each subject)
- `./model/cross-subject`: Models trained in cross-subject scenario (models trained on five subjects, respectively)

## Getting Started

### **Step 1**. Extract coverage files
Our simulation needs the coverage matrix of faulty programs.
Due to the storage limit, we have included the coverage matrix of the `Lang`, `Chart` and `Time` project. Before running the simulation, you need to extract the files:

```shell
cd data/coverage_matrix 
sh extract.sh Lang
cd ../../
```

### **Step 2**. Train the network model from scratch

Use the following command to train the model for `<project>` in cross-version scenario:
```shell
python simulate_network.py --metric TfD_network --pid <project> --output ./output.json --train 
```
The results will be save to `<path_to_output>`. Note that as soon as the model get trained, the script will run evaluation automatically.

Use the following command to train the model for `<project>` in cross-subject scenario:
```shell
python simulate_network_cross.py --metric TfD_network --pid <project> --train 
```

### **Step 3**. Run the evaluation

You can directly do evalution with our trained model in cross-version scenario. Use the following command to evaluate the model for `<project>`:
```shell
python simulate_network.py --metric TfD_network --pid <project> --output ./output.json
```

You can also directly do evalution with our trained model in cross-subject scenario. Use the following command to evaluate the model:
```shell
python simulate_network_cross.py --metric TfD_network --pid <project> 
```
**Using the script [`plot-RQ1.ipynb`](./plot-RQ1.ipynb), you can reproduce the `mAP` and `acc@n` results in Figure 3 and Table 2.**


### **Step 4**. Run the simulation for other metrics (e.g., TfD, EntBug, FDG)

Use the following command to run the simulation for bugs `<project>-<start>b` ~ `<project>-<end>b`:
```shell
python simulate_other.py --pid <project> --start <start> --end <end> --output <path_to_output> --formula <ochiai|wong2|dstar|op2|tarantula> --metric <fdc_metric> --iter <num_iterations>
```