# Graph Topology and Oversmoothing: A Study on GCNs

## Requirements

This project requires Python 3.10 and torch 2.5.1 and uses `conda` for environment management. The necessary dependencies are listed in `requirements.txt`.

Create a new conda environment:
```bash
conda create -n name_env python=3.10.16
```
Activate environment:
```bash
conda activate name_env
````

Install torch 2.5.1:
```bash
pip install torch==2.5.1
```

Install requirements:
```bash
conda install freetype
```

```bash
pip install -r requirements.txt
```
## Running examples
Compute Dirichlet energy of graph with 30 nodes and sparsity 0.5. Save results in output.txt
```bash
python main.py --num_nodes 30 --sparsity 0.5 --type_graph Sparsity --output_file output.txt
```
Compute Dirichlet energy of cycle graph with 30 nodes. Save results in output.txt
```bash
python main.py --num_nodes 30 --num_hubs 0 --type_graph Hubs --output_file output.txt 
```
Compute Dirichlet energy of graph with 30 nodes and 2 hubs. Save results in output.txt
```bash
python main.py --num_nodes 30 --num_hubs 2 --type_graph Hubs --output_file output.txt 
```
Compute Dirichlet energy of regular graph with 30 nodes and 120 edges. Save results in output.txt
```bash
python main.py --num_nodes of --num_hubs 0 --num_edges 120 --type_graph Regular --output_file output.txt
```
Compute Dirichlet energy of regular graph with 30 nodes and 120 edges. Save results in output.txt
```bash
python main.py --num_nodes 30 --num_hubs 0 --num_edges 120 --type_graph Regular --output_file output.txt
```
Compute Dirichlet energy of imbalanced graph with 1 hub, 30 nodes and 120 edges. Save results in output.txt
```bash
python main.py --num_nodes 30 --num_hubs 1 --num_edges 120 --type_graph Regular --output_file output.txt
```
Compute Dirichlet energy of graph with 30 nodes, starting with sparsity 1 and iteratively changing the diameter. Save results in output.txt
```bash
python main.py --num_nodes 30 --sparsity 1 --type_graph Diameter --output_file output.txt
```
Compute Dirichlet energy of graph with 30 nodes, starting with sparsity 1 and iteratively changing the average shortest path length. Save results in output.txt
```bash
python main.py --num_nodes 30 --sparsity 1 --type_graph ASPL --output_file output.txt
```

