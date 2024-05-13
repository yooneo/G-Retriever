import re
import os
import torch
import pandas as pd

from tqdm import tqdm
from torch_geometric.data.data import Data

from src.dataset.preprocess.generate_split import generate_split
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = '/home/neo/PycharmProjects/G-Retriever/dataset/irs'
dataset = pd.read_csv(f'{path}/source/graph.csv', sep='|')


def textualize_graph(graph_series):
    nodes = {}
    edges = []
    src, edge_attr, dst = graph_series.node_1, graph_series.edge, graph_series.node_2
    src = str(src).lower().strip()
    dst = str(dst).lower().strip()
    if src not in nodes:
        nodes[src] = len(nodes)
    if dst not in nodes:
        nodes[dst] = len(nodes)
    edges.append({'src': nodes[src], 'edge_attr': str(edge_attr).lower().strip(), 'dst': nodes[dst], })

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes, edges


def step_zero():
    # drop nan
    dataset.dropna(subset=["node_1", "node_2"], inplace=True)
    # create nodes file with unique node_id, and edges file with source, relation, target
    all_nodes = pd.concat([dataset["node_1"], dataset["node_2"]]).unique()

    # Create a mapping from nodes to unique IDs
    node_to_id = {node: i for i, node in enumerate(all_nodes)}

    # Save edges with node_id
    edges_df = dataset.copy(deep=True)
    edges_df.replace({"node_1": node_to_id, "node_2": node_to_id}, inplace=True)
    # Rename columns
    edges_df.rename(columns={'node_1': 'src', 'node_2': 'dst', 'edge': 'edge_attr'}, inplace=True)
    edges_df.to_csv(f"{path}/edges.csv", index=False, columns=['src', 'dst', 'edge_attr', 'chunk_id'])

    # Save nodes
    node_id_df = pd.DataFrame.from_dict(node_to_id, orient='index', columns=['node_id'])
    node_id_df.reset_index(level=0, inplace=True)
    node_id_df.columns = ["node_attr", "node_id"]
    node_id_df.to_csv(f"{path}/nodes.csv", index=False, columns=['node_attr', 'node_id'])


def step_one():
    # assign unique node_id to nodes, including node_1 and node_2
    # generate textual graphs
    os.makedirs(f'{path}/nodes', exist_ok=True)
    os.makedirs(f'{path}/edges', exist_ok=True)

    all_nodes = pd.DataFrame()
    all_edges = pd.DataFrame()

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        nodes, edges = textualize_graph(row)
        all_nodes = pd.concat([all_nodes, nodes])
        all_edges = pd.concat([all_edges, edges])

    all_nodes.to_csv(f'{path}/nodes.csv', index=False, columns=['node_id', 'node_attr'])
    all_edges.to_csv(f'{path}/edges.csv', index=False, columns=['src', 'edge_attr', 'dst'])


def step_two():
    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)
        for i in tqdm(range(len(dataset))):
            nodes = pd.read_csv(f'{path}/nodes/{i}.csv')
            edges = pd.read_csv(f'{path}/edges/{i}.csv')
            x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
            e = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
            edge_index = torch.LongTensor([edges.src, edges.dst])
            data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(nodes))
            torch.save(data, f'{path}/graphs/{i}.pt')

    def _encode_graph2():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs', exist_ok=True)

        nodes = pd.read_csv(f'{path}/nodes.csv')
        edges = pd.read_csv(f'{path}/edges.csv')

        num_nodes = nodes.node_id.nunique()
        x = text2embedding(model, tokenizer, device, nodes.node_attr.tolist())
        edge_index = torch.LongTensor([edges.src, edges.dst])
        edge_attr = text2embedding(model, tokenizer, device, edges.edge_attr.tolist())
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

        torch.save(data, f"{path}/graphs.pt")

    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    _encode_graph2()


if __name__ == '__main__':
    step_zero()
    # step_one()
    step_two()
    # generate_split(len(dataset), f'{path}/split')
