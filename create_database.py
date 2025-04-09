import argparse
import os
import pandas as pd

from ete3 import Tree
from pathlib import Path


def get_sequences_from_fasta(header_list, fasta_path):

    f = open(fasta_path, 'r')

    fasta_txt = f.read()
    data = [(seq.split()[0],"".join(seq.split('\n')[1:])) for seq in fasta_txt.split('>')[1:]]

    f.close()

    founded_seqs = ([],[])
    for head_in_file, seq in data:

        if head_in_file in header_list:
            founded_seqs[0].append(head_in_file)
            founded_seqs[1].append(seq.replace('\n',''))

    return founded_seqs
            
def get_nodes_at_depth(node, level, node_list=[], curr_level=0):

    if(curr_level == level):
        node_list.append(node)
    else:
        for child in node.get_children():
            get_nodes_at_depth(child, level, node_list, curr_level+1)
    return node_list

def define_classes_at_depth(fasta_path, tree: Tree, depth):

    node_list = get_nodes_at_depth(tree, depth)

    len_nodes = len(node_list)

    class_ids = [node.name for node in node_list]

    node2seqs = {class_ids[i]:node.get_leaf_names() for i,node in enumerate(node_list)}
    class2seqs = {}

    for id in class_ids:

        class2seqs[id] = get_sequences_from_fasta(node2seqs[id], fasta_path)

    return class2seqs

def define_classes_from_list(fasta_path, tree: Tree, node_ids_list, class_ids=None):

    node_list = [tree.search_nodes(name=id)[0] for id in node_ids_list]

    len_nodes = len(node_list)
    
    if not class_ids:
        class_ids = [node.name for node in node_list]

    node2seqs = {class_ids[i]:node.get_leaf_names() for i,node in enumerate(node_list)}
    class2seqs = {}

    for id in class_ids:

        class2seqs[id] = get_sequences_from_fasta(node2seqs[id], fasta_path)

    return class2seqs
    
def save_to_csv(class2ids, ofpath):
        
    seq_ids = []
    sequences = []
    labels = []
    
    for class_id, head_seqs in class2ids.items():

       seq_ids.extend(head_seqs[0])
       sequences.extend(head_seqs[1])

       labels.extend([class_id] * len(head_seqs[0]))

    
    dataset_dict = {'id_seq': seq_ids, 'sequence': sequences, 'group': labels}

    path = Path(ofpath)

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame.from_dict(dataset_dict).to_csv(path, index=True)

def main(args):

    depth = args.depth
    node_list = args.node_list

    if depth is None and len(node_list) == 0:
        
        raise Exception("Please one between depth and node list is required")

    elif depth and len(node_list) > 0:

        raise Exception("Only one between depth and node list is required")
    
    tree = Tree(args.tree)


    if depth:

        class2seqs = define_classes_at_depth(args.ifpath, tree, depth)

    elif len(node_list) > 0:
        
        class2seqs = define_classes_from_list(args.ifpath, tree, depth, args.node_list, args.class_ids)
    
    save_to_csv(class2seqs, args.ofpath)

if __name__ == "__main__":
    # options for using from the command line

    parser = argparse.ArgumentParser(description="Only one between depth and node list is required")

    parser.add_argument(
        "-f", "--fpath", dest="ifpath", type=str, default=None, required=True, help="File path fasta"
        )
    parser.add_argument(
        "-t", "--tree", dest="tree", type=str, default=None, required=True, help="Tree file format newick"
        )
    parser.add_argument(
        "-o", "--ofpath", dest="ofpath", type=str, default=".", help="Output file path"
        )
    parser.add_argument(
        "--depth", dest="depth", type=int, default=None, help="Depth where cutting subtree (clusters) - defining class"
        )
    parser.add_argument(
        "--nodes", dest="node_list", type=str, default=[], help="Internal Node IDs list of subtrees"
        )
    parser.add_argument(
        "--class", dest="class_ids", type=str, default=None, help="Class ids associated at same order with nodes (only if passing a node list)"
        )
    
    options = parser.parse_args()

    main(options)