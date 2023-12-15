# Test Generation with EVOSUITE 
## Statistics of generated tests for RQ3
| Metric    | EVO-line   |EVO-DDU   |EVO-EntBug   |EVO-RLFDC      |
| :-------- | :--------- | :------- | :---------- | :-------- | 
| # Total tests      | 42,482 | 24,347 | 13,026 | 24,271 |
| # Failing tests    | 1,413  | 1,209  | 1,004  | 1,282  | 


## Prerequisite
- docker

## Getting Started

- 🚨 **Please make sure that the docker daemon is running on your machine.** If not, you will encounter the error: `Cannot connect to the Docker daemon at unix:///var/run/docker.sock`
- 🚨 The docker image `agb94/fdg` only supports the  `linux/amd64` architecture. If the architecture of your processor is `arm64`, e.g. Apple sillicion, please try with another machine.


### **Step 1**. Setup Docker

```shell
cd docker
docker pull agb94/fdg
sh make_dirs.sh
docker run -dt --volume $(pwd)/resources/d4j_expr:/root/workspace --volume $(pwd)/results:/root/results --name fdg agb94/fdg
docker exec -it fdg /bin/bash
```

This will
- load the docker image `fdg:amd64`,
- create a docker container named `fdg`,
- and create a new Bash session in the container `fdg`

Now, go to **Step 2**.

### **Step 2**. Change EVOSUITE config

- `cp evosuite-config /root/evosuite-config`

### **Step 3**. Generate test cases and do fault localization with RLFDC

The script `invocation_xxx.py` will run the corresponding baseline on the specific project. 
-  run `nohup python3 invocation_xxx.py ><project>_<seed>.out 2>&1 &`
-  You can run several projects in parellel, i.e., run Chart, Time, Lang, Math, Closure simutaneously by change the `invocation_xxx.py`.
-  Closure may take much more time. So you may try to run different versions in parellel. 

### **Step 4**. (Optional) Do fault localization with other metrics

- `cd /root/workspace`
- `python3 tfd_main.py Lang 1 --tool evosuite --id <test_suite_id> --budget 10 --selection FDG:0.5 --noise 0.0`


### **Step 4**. Summarize results
- run `python3 summarize_tfd_FL_results.py <baseline_name>_TS_<seed> 120 TfD_network -q 10 -o ./output_analysis/<entbug/ddu>_b120_q10_TfDnetwork_output.pkl`
- run `python3 cal_acc@n.py --file ./output_analysis/<entbug/ddu>_b120_q10_TfDnetwork_output.pkl`
