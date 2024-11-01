import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost as xgb

from src.utils.preprocessing import process
from src.utils.evaluation import recenter_data



## DIAGRAM COMPARISONS OF DATA LEAKAGE VS NO DATA LEAKAGE

from graphviz import Digraph

def create_flowchart():
    dot = Digraph(comment='Approach Comparison')

    # Subgraph for Approach with Data Leakage
    with dot.subgraph(name='cluster_leakage') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        c.edges([('A1', 'B1'), ('B1', 'C1'), ('B1', 'D1'), ('C1', 'E1'), ('E1', 'F1'), ('F1', 'G1'), ('G1', 'H1'), ('H1', 'I1'), ('I1', 'J1')])
        c.attr(label='Approach with Data Leakage')

    # Subgraph for Approach without Data Leakage
    with dot.subgraph(name='cluster_no_leakage') as c:
        c.attr(color='blue')
        c.node_attr['style'] = 'filled'
        c.edges([('A2', 'B2'), ('B2', 'C2'), ('B2', 'D2'), ('C2', 'E2'), ('E2', 'F2'), ('F2', 'G2'), ('G2', 'H2'), ('H2', 'I2')])
        c.attr(label='Approach without Data Leakage')

    # Custom styling for specific nodes
    dot.node('A1', 'Original Dataset', fillcolor='#f9f')
    dot.node('A2', 'Original Dataset', fillcolor='#f9f')
    dot.node('J1', 'Evaluate on Test', fillcolor='#f66')
    dot.node('I2', 'Average Results', fillcolor='#6f6')

    dot.render('flowchart.gv', view=True)  # This will create and immediately open the diagram

create_flowchart()
