# Human-written Test Selection (RQ1)

## Precomputed Simulation Results

The precomputed results are available in the directory `output/`.
- `tests`: the index of the test case (following the order in the coverage matrix) selected at each iteration
- `ranks`: the rank of faulty methods at each iteration
- `fitness_history`: the fitness of the selected test cases at each iteration
- `full_ranks`: the rank of faulty methods when using all available test cases


## Getting Started

### **Step 1**. Extract coverage files
Our simulation needs the coverage matrix of faulty programs.
Due to the storage limit, we have included the coverage matrix of only the `Lang` project. Before running the simulation, you need to extract the files:

```shell
cd data/coverage_matrix 
sh extract.sh Lang
cd ../../
```

### **Step 2**. Train the network model from scratch

Use the following command to train the model for `<project>`:
```shell
python simulate_network.py --metric TfD_network --pid <project> --output ./output.json --train 
```
The results will be save to `<path_to_output>`. Note that as soon as the model get trained, the script will run evaluation automatically.


### **Step 3**. Run the evaluation

You can directly do evalution with our trained model. Use the following command to evaluate the model for `<project>`:
```shell
python simulate_network.py --metric TfD_network --pid <project> --output ./output.json
```
**Using the script [`plot-RQ1.ipynb`](./plot-RQ1.ipynb), you can reproduce the `mAP` and `acc@n` results in Figure 3 and Table 2.**


### **Step 4**. Run the simulation for other metrics

Use the following command to run the simulation for bugs `<project>-<start>b` ~ `<project>-<end>b`:
```shell
python simulate_other.py --pid <project> --start <start> --end <end> --output <path_to_output> --formula <ochiai|wong2|dstar|op2|tarantula> --metric <diagnosability_metric> --iter <num_iterations>
```