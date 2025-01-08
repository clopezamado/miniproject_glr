import os
import typing
import torch_geometric
from GNNmodels import GCN
import torch
import datasets
import torch.nn as nn
import numpy as np
import random
import measureOversmoothing
import matplotlib.pyplot as plt
import argparse
import networkx as nx


def train(
    params: typing.Dict,data
) -> torch.nn.Module:
  """
    This function trains a node classification model and returns the trained model object.
  """
    
  # Set a model
  if params['model_name'] == 'GCN':
      model = GCN(
        params["input_dim"],
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
        ).to(device)
  else:
      raise NotImplementedError

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
    model.parameters(),  # Parameters to optimize
    lr=params["lr"],  # Learning rate
    weight_decay=params["weight_decay"]  # Weight decay for regularization
  )
  curr_val_acc=0
  conseq_val_decrease=0
  for epoch in range(params["epochs"]):
    if conseq_val_decrease<params["max_patience"]:
      model.train()
      optimizer.zero_grad()

      logits_train=model(data.x,data.edge_index)
      loss = loss_fn(logits_train[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()

      #Obtain validation accuracy
      val_accuracy=evaluate(model,data,data.val_mask)
      if val_accuracy<curr_val_acc:
        conseq_val_decrease+=1
      else:
        conseq_val_decrease=0
      curr_val_acc = max(curr_val_acc, val_accuracy)
      print("epoch={}, loss={}, validation accuracy={}".format(epoch, loss.item(), val_accuracy))
    else:
      return model
  return model

def evaluate(
    model,
    data,
    mask
):
    model.eval()

    logits = model(data.x, data.edge_index)
    predictions = logits[mask].argmax(dim=1)
    correct = (predictions == data.y[mask]).sum().item()
    total_num = mask.sum().item()
    val_accuracy = correct / total_num if total_num > 0 else 0.0
    return val_accuracy


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--sparsity")
    parser.add_argument("--num_hubs")
    parser.add_argument("--type_graph")
    parser.add_argument("--num_edges")
    parser.add_argument("--output_file")
    args = parser.parse_args()
    if args.num_nodes:
        n_nodes=int(args.num_nodes)
    if args.sparsity:
        sparsity=float(args.sparsity)
    if args.num_hubs:
        n_hubs=int(args.num_hubs)
    if args.num_edges:
        n_edges=int(args.num_edges)
    else:
        n_edges=n_nodes
    type_graph=args.type_graph
    output_file=args.output_file
    
    training_params = {
    "lr": 0.05,  # learning rate
    "weight_decay": 0.0005,  # weight_decay
    "epochs": 0,  # number of total training epochs
    "max_patience": 5, # number of k for early stopping
    "hid_dim": 128, # size of hidden features
    "n_layers": 128, # number of layers
    "model_name": "GCN",
    "num_nodes": 2000,
    "sparsityLevel":1,
    "n_classes": 4,
    "input_dim":128
    }
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    # load dataset
    if type_graph=="Sparsity":
        data = datasets.obtainSparseGraph(n_nodes,sparsity,training_params["n_classes"])
        data = data.to(device)
        model=train(training_params,data)
        energy=measureOversmoothing.get_dirichlet_energy_model(model,data)
        with open(output_file, "a") as file:
            # Write the list with brackets and comma-separated elements
            file.write(f"({n_nodes},{sparsity})\n")
            file.write(f"[{', '.join(map(str, energy))}]\n")
    elif type_graph=="Hubs":
        data = datasets.obtainUniformGraphWithHubs(n_nodes,n_edges,n_hubs,training_params["n_classes"])
        data = data.to(device)
        model=train(training_params,data)
        energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
        with open(output_file, "a") as file:
            # Write the list with brackets and comma-separated elements
            file.write(f"({n_nodes},{n_hubs},{n_edges})\n")
            file.write(f"[{', '.join(map(str, energy))}]\n")
    elif type_graph=="Regular":
        data = datasets.obtainUniformGraphWithHubs_regular(n_nodes,n_edges,n_hubs,training_params["n_classes"])
        data = data.to(device)
        model=train(training_params,data)
        energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
        with open(output_file, "a") as file:
            # Write the list with brackets and comma-separated elements
            file.write(f"({n_nodes},{n_hubs},{n_edges})\n")
            file.write(f"[{', '.join(map(str, energy))}]\n")
    elif type_graph=="Diameter":
        data = datasets.obtainSparseGraph(n_nodes,sparsity,training_params["n_classes"])
        model=train(training_params,data)
        changedDiamGraphs,diameters_changedDiam,noChangedDiamGraphs=datasets.changeDiameter(data,10)
        with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"Change diameter\n")
        for step, (data, diam) in enumerate(zip(changedDiamGraphs, diameters_changedDiam)):           
            data = data.to(device)
            energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
            with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"({n_nodes},{sparsity},{step})\n")
                file.write(f"{diam}\n")
                file.write(f"[{', '.join(map(str, energy))}]\n")
        with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"No change diameter\n")
        for step,data in enumerate(noChangedDiamGraphs):
            data = data.to(device)
            energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
            with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"({n_nodes},{sparsity},{step})\n")
                file.write(f"[{', '.join(map(str, energy))}]\n")
    elif type_graph=="ASPL":
        data = datasets.obtainSparseGraph(n_nodes,sparsity,training_params["n_classes"])
        model=train(training_params,data)
        changedGraphs,aspl_changed,noChangedGraphs,aspl_noChanged=datasets.changeASPL(data,30,5)
        with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"Change ASPL\n")
        for step, (data,aspl) in enumerate(zip(changedGraphs,aspl_changed)):           
            data = data.to(device)
            energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
            with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"({n_nodes},{sparsity},{step})\n")
                file.write(f"{aspl}\n")
                file.write(f"[{', '.join(map(str, energy))}]\n")
        with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"No change ASPL\n")
        for step,(data,aspl) in enumerate(zip(noChangedGraphs,aspl_noChanged)):
            data = data.to(device)
            energy=measureOversmoothing.get_dirichlet_energy_model(model,data)   
            with open(output_file, "a") as file:
                # Write the list with brackets and comma-separated elements
                file.write(f"({n_nodes},{sparsity},{step})\n")
                file.write(f"{aspl}\n")
                file.write(f"[{', '.join(map(str, energy))}]\n")

