import os
from typing import io
import io as os_io

import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst
from src.utils.seed import seed_everything
from src.config import parse_args_llama
from src.model import load_model as load_gen_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.utils.collate import collate_fn
import json

from src.utils.lm_modeling import load_model as load_embedding_model, load_text2embedding

embedding_model_name = "sbert"
generate_model_name = "llama3-8b-4bit"
path = "/home/neo/PycharmProjects/G-Retriever/dataset/irs"
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

emb_model, emb_tokenizer, device = load_embedding_model[embedding_model_name]()
text2embedding = load_text2embedding[embedding_model_name]
graph = torch.load(f"{path}/graphs.pt")

args = parse_args_llama()
args.llm_model_path = llama_model_path[args.llm_model_name]
args.model_name = 'single_inference_llm'
gen_model = load_gen_model['single_inference_llm'](graph=graph, graph_type='Knowledge Graph', args=args)
gen_model = gen_model.to(device)


def retrieve_subgraph(question: str):
    nodes = pd.read_csv(f"{path}/nodes.csv")
    edges = pd.read_csv(f"{path}/edges.csv")

    q_emb = text2embedding(emb_model, emb_tokenizer, device, question)
    subgraph, desc = retrieval_via_pcst(graph, q_emb, nodes, edges, topk=3, topk_e=5, cost_e=0.5)

    return subgraph, desc


def generate(subgraph, desc, question):

    gen_input = {
        'id': '1',
        'question': question,
        'desc': desc,
        'label': [],
    }

    gen_output = gen_model.inference(gen_input)

    return gen_output


if __name__ == "__main__":
    args = parse_args_llama()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    question = pd.read_csv(f"{path}/questions.csv")
    retrieval_output = f"{path}/retrieval_output.csv"
    output_df = pd.DataFrame(columns=['question', 'nodes', 'edges', 'pred'])

    for q in question['question']:
        subgraph, desc = retrieve_subgraph(question=q)
        gen = generate(subgraph, desc, question=q)

        new_row = {}

        try:
            if gen.get('pred'):
                data = pd.read_csv(os_io.StringIO(gen['pred']), dtype=str)
                # convert your data to single string if it's not already
                new_row['pred'] = data.to_string()

            if gen.get('desc'):
                # Split 'desc' field into two parts
                node_attr, edge_attr = gen['desc'].split('\n\n')
                # Convert each part into DataFrame
                node_attr_df = pd.read_csv(os_io.StringIO(node_attr), dtype=str)
                edge_attr_df = pd.read_csv(os_io.StringIO(edge_attr), dtype=str)

                new_row['nodes'] = node_attr_df.to_string()
                new_row['edges'] = edge_attr_df.to_string()

            new_row['question'] = q

            # Append new_row to existing output_df
            output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            print(f"Error: {e}, row: {q}")

    output_df.to_csv(retrieval_output, index=False)
