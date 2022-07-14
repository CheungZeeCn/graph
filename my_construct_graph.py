import pandas as pd
import pickle
import argparse
from tqdm import tqdm, trange
from torch_geometric.data import Data
import torch
import os
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import numpy as np

g_batch_size = 500
all_embs = None


def node_id_mapping_from_txt(path, df_frac=1):
    """
    Function to map keywords and titles to node ids
    and save the mapping dictionary to node_id_map.pkl
    Args:
    path:str - Path to the dataset tsv
    df_frac:float - Percentage of data to use (for large files)
    Returns:
    Dictionary of {keyword/title: id}
    """
    nodes = []
    with open(path) as f:
        for node in f:
            node = node.strip()
            if node != "":
                nodes.append(node)
    node_id_map = dict(zip(nodes, range(len(nodes))))
    pickle.dump(node_id_map, open("node_id_map.pkl", "wb"))
    return node_id_map, nodes


def make_semantic_node_embeddings_cuda(path, df_frac=1):
    """
    Function to save semantic embeddings of keywords and titles
    in batches of g_batch_size to the directory node_embeds/
    Args:
    path:str - Path to the dataset tsv
    df_frac:float - Percentage of data to use (for large files)
    """
    if os.path.isfile('./node_id_map.pkl'):
        node_id_map = pickle.load(open("node_id_map.pkl", "rb"))
        id_node_map = {v:k for k, v in node_id_map.items()}
        nodes = [id_node_map[i] for i in range(len(id_node_map))]
    else:
        node_id_map, nodes = node_id_mapping_from_txt(path, df_frac)
    print("Loading Tokenizer")
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = XLMRobertaModel.from_pretrained(model_name)
    model = model.to('cuda')

    print("Initialize node_embeds")
    #node_keys = list(node_id_map.keys())
    node_keys = nodes
    print("Number of nodes", len(node_keys))
    batch_size = g_batch_size
    print("Making semantic node embeddings")
    for ind in trange(1 + len(node_keys) // batch_size):
        raw_inputs = node_keys[ind * batch_size:(ind + 1) * batch_size]
        if len(raw_inputs) == 0:
            break
        inputs = tokenizer(raw_inputs, add_special_tokens=True, padding='max_length',
                           max_length=128, truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to('cuda')

        with torch.no_grad():
            outputs = torch.mean(model(**inputs).last_hidden_state, dim=1)
        np.save(open(f"node_embeds/node_embed_{ind}", "wb"), outputs.cpu().detach().numpy())
    return


def init_all_embedding(base_data="./node_embeds/"):
    all_embs = np.load("node_embeds/node_embed_all.np")
    # all_embs = []
    # for i in range(0, 2895):
    #     # if i % 100 == 0:
    #     #     print(i)
    #     i_f = f"{base_data}node_embed_{i}"
    #     all_embs.append(np.load(i_f))
    # all_embs = np.vstack(all_embs)
    # np.save(open(f"node_embeds/node_embed_all.np", "wb"), all_embs)
    print(f"all_embs.shape {all_embs.shape}, save to node_embed_all.np")
    return all_embs


def get_embedding(text, node_id_map, root_node_path="./node_embeds/"):
    """
    Function to retrieve the semantic embedding of a
    text keyword/title using the node_id_map and saved
    semantic embeddings.
    Args:
    text:str - Text of the title/keyword
    node_id_map:dict - Map of the text to the node id
    root_node_path:str - Directory where the semantic embeddings are saved
    Returns:
    feature_embed:np.array - semantic embedding for text
    """
    global  all_embs
    node_id = node_id_map.get(text, 0)
    #feature_embed = np.load(root_node_path + f"node_embed_{node_id // g_batch_size}")[node_id % g_batch_size]
    feature_embed = all_embs[node_id]
    return feature_embed


def construct_graph_df(path, df_frac=1, threshold=100):
    """
    Function to get the 2-hop neighborhood of all keywords and titles.
    The neighborhood is saved in a pickle file: two_hop_ngbrs.pkl
    The function gives priority to the 1-hop neighborhood.
    Args:
    path:str - Path to the dataset tsv
    df_frac:float - Percentage of data to use (for large files)
    threshold:int - Maximum size of the neighborhood to be considered
    """
    # HEADER: query	product_id	product_title	query_locale	esci_label	gain
    df = pd.read_pickle(path)
    df = df.sample(frac=df_frac, random_state=42)
    # 不进一步提取keyword了直接query
    # query->asin list
    qa_df = df.groupby('query')['product_title'].apply(list).reset_index(name="neighbours")
    query_asin_map = dict(zip(qa_df["query"], qa_df.get("neighbours", [])))
    # asin->query list
    aq_df = df.groupby('product_title')['query'].apply(list).reset_index(name="neighbours")
    asin_query_map = dict(zip(aq_df["product_title"], aq_df.get("neighbours", [])))
    del (df)
    two_hop_ngbrs = {}
    node_id_map = pickle.load(open("node_id_map.pkl", "rb"))
    root_node_path = "./node_embeds/"
    print("Constructing query neighborhood graph")
    # for 每一个 query->1nb list  q -> a
    for _, row in tqdm(qa_df.iterrows(), total=len(qa_df)):
        r, c, feature = [], [], []
        neighbours = set(list(filter(lambda i: type(i) is str, list(row["neighbours"]))))
        # query的emb
        feature.append(get_embedding(row["query"], node_id_map, root_node_path))
        for ngbr in list(neighbours)[:threshold]:
            # 每一个 1阶邻居 的 emb
            # feature.append(get_embedding(ngbr, node_id_map, root_node_path))
            # 1 如果有邻居到query(title)
            if ngbr in asin_query_map:
                # 并集 加入 2阶邻居 (query)
                neighbours.update(set(filter(lambda i: type(i) is str, list(aq_df.get(ngbr, [])))))

        # 这里面没有区分是1/2阶 做法也好奇怪
        neighborhood = list(neighbours)[:threshold]
        for _, neighbor in enumerate(neighborhood):
            feature.append(get_embedding(neighbor, node_id_map, root_node_path))
            r.append(0)
            c.append(_ + 1)
        # 奇怪的构图 也没有注释
        two_hop_ngbrs[row["query"]] = Data(x=torch.tensor(feature, dtype=float),
                                             edge_index=torch.tensor([r, c], dtype=int))
    # pickle.dump(two_hop_ngbrs, open("q-p_two_hop_ngbrs.pkl", "wb"))

    # a->query
    print("Constructing asin neighborhood graph")

    # i = 0
    # def dump_tmp_two_hop_ngbrs(tmp_dict, i):
    #     base = "./node_embeds/2hops/"
    #     file = f"{base}two_hop_ngbrs_{i}.pkl"
    #     print(f"dump {len(tmp_dict)} 2hop to file {file}")
    #     pickle.dump(tmp_dict, open(file, "wb"))

    # tmp_two_hop_ngbrs = {}
    for _, row in tqdm(aq_df.iterrows(), total=len(aq_df)):
        r, c, feature = [], [], []
        neighbours = set(list(filter(lambda i: type(i) is str, list(row["neighbours"]))))
        feature.append(get_embedding(row["product_title"], node_id_map, root_node_path))
        for ngbr in list(neighbours)[:threshold]:
            if ngbr in query_asin_map:
                neighbours.update(set(filter(lambda i: type(i) is str, list(aq_df.get(ngbr, [])))))
        neighborhood = list(neighbours)[:threshold]
        for _, neighbor in enumerate(neighborhood):
            feature.append(get_embedding(neighbor, node_id_map, root_node_path))
            r.append(0)
            c.append(_ + 1)
        two_hop_ngbrs[row["product_title"]] = Data(x=torch.tensor(feature, dtype=float),
                                           edge_index=torch.tensor([r, c], dtype=int))
    #     if len(tmp_two_hop_ngbrs) == 500000:
    #         dump_tmp_two_hop_ngbrs(tmp_two_hop_ngbrs, i)
    #         tmp_two_hop_ngbrs = {}
    #         i += 1
    # if len(tmp_two_hop_ngbrs) != 0:
    #     dump_tmp_two_hop_ngbrs(tmp_two_hop_ngbrs, i)

    pickle.dump(two_hop_ngbrs, open("two_hop_ngbrs.pkl", "wb"))


if __name__ == "__main__":
    """
    Main script to be called for the graph neighborhood construction for the SMLM model.
    make_semantic_node_embeddings - Creates the semantic embeddings for the nodes.
    construct_graph - Constructs the graph neighborhood using the 'Data' class of the torch-geometric library.
    Files created:
    node_id_map.pkl: map of the titles/keywords to unique ids.
    node_embeds/: directory with semantic embeddings for the titles/keywords.
    two_hop_ngbrs.pkl: final graph with the two hop neighborhood for all titles and keywords
    """
    parser = argparse.ArgumentParser(description='Construct graph from query-asin dataset')
    parser.add_argument('--node_path', metavar='P', type=str, help='path to the query-asin dataset', default="/home/ana/data2/projects/esci-code/data/node_embeds/texts.txt")
    # t1_subt2 only
    parser.add_argument('--path', metavar='P', type=str, help='path to the query-asin dataset', default="/home/ana/data2/projects/esci-code/data/node_embeds/t12_graph_esc.pkl")

    parser.add_argument('--frac', metavar='F', type=float,
                        help='incase we need to only process a fraction of the dataset', default=1)
    parser.add_argument('--threshold', metavar='T', type=int, help='threshold on the size of neighborhood', default=100)

    args = parser.parse_args()
    # make_semantic_node_embeddings_cuda(args.node_path, df_frac=1)
    # print("make_semantic_node_embeddings_cuda DONE")
    # hardcode here!!!!
    all_embs = init_all_embedding()
    construct_graph_df(args.path, df_frac=args.frac, threshold=args.threshold)
