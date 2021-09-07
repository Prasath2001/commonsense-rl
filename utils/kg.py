import sys
import networkx as nx
import logging
import json
import requests
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.generic import escape_entities

# Logging formatting
FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level='INFO', stream=sys.stdout)
kg = {}
source_paths= defaultdict(dict)


def shortest_path_subgraph(kg_graph, prev_graph, nodes, inventory_entities=None, command_entities=None, path_len=2, add_all_path=False):
    if inventory_entities is None:
        inventory_entities = []
    if command_entities is None:
        command_entities = []
    # Get non-neighbor nodes: nodes without edges between them
    world_graph = kg_graph.subgraph(list(prev_graph.nodes)+nodes).copy()
    world_graph = nx.compose(prev_graph,world_graph)
    world_graph.remove_edges_from(nx.selfloop_edges(world_graph))

    if path_len < 2:
        return world_graph
    triplets = []
    # Add command related relations
    pruned_entities = list(set(command_entities)-set(inventory_entities))
    if pruned_entities:
        for src_et in inventory_entities:
            for tgt_et in pruned_entities:
                if src_et != tgt_et:
                    try:
                        pair_dist = nx.shortest_path_length(kg_graph, source=src_et, target=tgt_et)
                    except nx.NetworkXNoPath:
                        pair_dist = 0
                    if pair_dist >= 1 and pair_dist <= path_len:
                        triplets.append([src_et, tgt_et, 'relatedTo'])
    else: # no items in the pruned entities, won't happen
        for entities in command_entities:
            for src_et in entities:
                for tgt_et in entities:
                    if src_et != tgt_et:
                        try:
                            pair_dist = nx.shortest_path_length(kg_graph, source=src_et, target=tgt_et)
                        except nx.NetworkXNoPath:
                            pair_dist=0
                        if pair_dist >= 1 and pair_dist <= path_len:
                            triplets.append([src_et, tgt_et, 'relatedTo'])
    world_graph, _= add_triplets_to_graph(world_graph, triplets)
    return world_graph


def construct_graph(triplets): # Gets triplets like (e1,r,e2) and returns a directed graph and entities=concepts in subgraph.
    graph = nx.DiGraph()
    entities = {}
    for [e1, e2, r] in triplets:
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r not in graph.edges[e1, e2]['relation']:
                graph.edges[e1, e2]['relation'] += ' ' + r
        else:
            graph.add_edge(e1, e2, relation=r)
    return graph, entities


def add_triplets_to_graph(graph, triplets): # Adding edges between entities with relationship type and returning graph and entities.
    entities = dict(graph.nodes.data())
    for [e1, e2, r] in triplets:
        e1 = e1.lower().strip()
        e2 = e2.lower().strip()
        r = r.lower().strip()
        if e1 not in entities:
            graph.add_node(e1)
            entities[e1] = e1
        if e2 not in entities:
            graph.add_node(e2)
            entities[e2] = e2
        # Add Edge information
        if graph.has_edge(e1, e2):
            if r not in graph.edges[e1, e2]['relation']:
                graph.edges[e1, e2]['relation'] += ' ' + r
        else:
            graph.add_edge(e1, e2, relation=r)
    return graph, entities


def draw_graph(graph, title="cleanup", show_relation=True, weights=None, pos=None): # Plots the graph passed with relations.
    if not pos:
        pos = nx.spring_layout(graph, k=0.95)
    if weights:
        nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000, node_color=weights.tolist(),
                vmin=np.min(weights), vmax=np.max(weights), node_shape='o', alpha=0.9, font_size=8, with_labels=True,
                label=title,cmap='Blues')
    else:
        nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000, node_color='pink',
                node_shape='o', alpha=0.9, font_size=8, with_labels=True, label=title)
    if show_relation:
        p_edge = nx.draw_networkx_edge_labels(graph, pos, font_size=6, font_color='red',
                                          edge_labels=nx.get_edge_attributes(graph, 'relation'))


def draw_graph_colormap(graph,node_weights, showbar=False, cmap='YlGnBu'):
    # node_weights: maps node id/name to attention weights
    pos = nx.spring_layout(graph, k=0.95)
    weights = []
    for node in graph.nodes:
        weights.append(node_weights[node])
    # cmap = plt.cm.YlGnBu#RdBu
    cmap = plt.get_cmap(cmap)
    vmin = np.min(weights)
    vmax = np.max(weights)
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1, node_size=1000,
            node_color=weights, vmin=vmin, vmax=vmax, cmap=cmap,
            node_shape='o', alpha=0.9, font_size=8, with_labels=True, label='Attention')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    if showbar:
        plt.colorbar(sm)
    plt.show()


def construct_kg(filename: str, print_every=1e6, cache_load=True, logger=logging.getLogger(__name__)) -> (nx.DiGraph, list, set):
    ### This functions takes conceptnet_subgraph.txt as input and returns a undirected graph, triplets, entities to play function in train_agent.py
    # access edges with graph.edges.data('relation')
    if 'graph' in kg and cache_load:
        return kg['graph'], kg['triplets'], kg['entities']

    path = Path(filename)
    if not path.exists():
        filename = './kg/conceptnet/kg.txt'

    triplets = []
    with open(filename, 'r') as fp:
        for idx, line in enumerate(fp):
            e1, r, e2  = line.rstrip("\n").rsplit()
            triplets.append([e1.lower().strip(), e2.lower().strip(), r.lower().strip()])
            if idx % print_every == 0:
                print("*",end='')
    [graph, entities] = construct_graph(triplets)
    graph = graph.to_undirected(as_view=True) # Since a->b ==> b->a
    if cache_load:
        kg['graph'] = graph
        kg['triplets'] = triplets
        kg['entities'] = entities
    return graph, triplets, entities


class RelationExtractor:
    def __init__(self, tokenizer, openie_url="http://localhost:9000/"):
        """
        :param tokenizer:
        :param openie_url: server url for Stanford Core NLPOpen IE
        """
        self.tokenizer = tokenizer
        self.openie_url = openie_url
        self.kg_vocab = {}
        self.agent_loc = ''

    def call_stanford_openie(self,sentence): # Used to retrieve Open Information Extraction as triplets. Example : born-in (Barack Obama, Hawaii) 
        """ To know about OpenIE visit https://nlp.stanford.edu/software/openie.html """
        querystring = {
            "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
            "pipelineLanguage": "en"}
        response = requests.request("POST", self.openie_url, data=sentence, params=querystring) # Accessing data using API request for Standford Core NLPOpen IE Server.
        response = json.JSONDecoder().decode(response.text)
        return response

    def fetch_triplets(self,text, current_graph, prev_action=None): # This function gets observation, inventory, description , hints as a combined string = text and returns triplets.
        triplets = []
        remove = []
        prev_remove = []
        link = []
        c_id = len(self.kg_vocab.keys()) # Number of keys in vocabulary of knowledge graph.
        obs = self.tokenizer.clean_string(text, preprocess=True) # Preprocess and Cleans the string.
        dirs = ['north', 'south', 'east', 'west'] # Directions
        obs = str(obs) # Observation as string
        doc = self.tokenizer.nlp_eval(obs) # Obs is passed to Spacy NLP object , returns doc.
        sents = {} # Stores result of OpenIE
        try:
            sents = self.call_stanford_openie(doc.text)['sentences'] # doc.text contains obs as string.
        except:
            print("Error in connecting to Stanford CoreNLP OpenIE Server")
        for ov in sents:
            tokens = ov["tokens"] # Contains tokens of a sentence.
            triple = ov['openie'] # Contains triplet of a sentence.
            for tr in triple:
                h, r, t = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower() # triplet is changed to lower case.
                if h == 'we':
                    h = 'you'
                    if r == 'are in':
                        r = "'ve entered"

                if h == 'it': # Don't consider relationship between objects. Consider only between subject and object.
                    break
                triplets.append((h, r, t)) # Store triplets of all sentences.

        room = ""
        room_set = False
        for rule in triplets:
            h, r, t = rule
            if 'entered' in r or 'are in' in r or 'walked' in r: # Checks for a room in object of triplet.
                prev_remove.append(r)
                if not room_set:
                    room = t
                    room_set = True # A room setting is found from obs.
            if 'should' in r:
                prev_remove.append(r)
            if 'see' in r or 'make out' in r: 
                link.append((r, t))
                remove.append(r)
            # else:
            #    link.append((r, t))

        prev_room = self.agent_loc # Previous agent's location is stored in previous room.
        self.agent_loc = room # Update the current location of agent from room.
        add_rules = [] # Add rules
        if prev_action is not None:
            for d in dirs:
                if d in prev_action and room != "": # If there is a action performed with a direction involved and room is not None, then append it to rules.
                    add_rules.append((prev_room, d + ' of', room)) # Example rule : Kitchen south of Livingroom.
        prev_room_subgraph = None
        prev_you_subgraph = None

        for sent in doc.sents:
            sent = sent.text
            if sent == ',' or sent == 'hm .':
                continue
            if 'exit' in sent or 'entranceway' in sent: # If there is exit or entrance to a room
                for d in dirs:
                    if d in sent:
                        triplets.append((room, 'has', 'exit to ' + d)) # Storing it in triplet.
        if prev_room != "":
            graph_copy = current_graph.copy() # current graph is from infos['local/world graph']
            graph_copy.remove_edge('you', prev_room) # Removing the Current location of agent in the Graph to update.
            con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)] # Connected components.

            for con_c in con_cs: # Iteration each component of graph
                if prev_room in con_c.nodes: # Room is one of the nodes
                    prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes) # Returns a subgraph of nodes in component from local/world graph.
                if 'you' in con_c.nodes: # prev_you_subgraph keeps track of agent's information.
                    prev_you_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)

        for l in link:
            add_rules.append((room, l[0], l[1])) # Rules to link objects in room. Example : Kitchen see stove etc.

        for rule in triplets:
            h, r, t = rule
            if r == 'is in':
                if t == 'room':
                    t = room
            if r not in remove: # If r is not 'see' or 'make out', append to rules.
                add_rules.append((h, r, t)) 
        edges = list(current_graph.edges)
        for edge in edges:
            r = 'relatedTo'
            if 'relation' in current_graph[edge[0]][edge[1]]:
                r = current_graph[edge[0]][edge[1]]['relation']
            if r in prev_remove:
                current_graph.remove_edge(*edge)

        if prev_you_subgraph is not None: # If agent's graph is none, then remove edges of agent from current graph.
            current_graph.remove_edges_from(prev_you_subgraph.edges) 

        for rule in add_rules:
            u = '_'.join(str(rule[0]).split()) # Gets node u from rule.
            v = '_'.join(str(rule[2]).split()) # Gets node v from rule.
            if u != 'it' and u not in self.kg_vocab: # Node is added in vocabulary of KG.
                self.kg_vocab[u] = c_id
                c_id += 1 # Incrementing length of vocab of KG.
            if v != 'it' and v not in self.kg_vocab: # Same as node u.
                self.kg_vocab[v] = c_id
                c_id += 1
            skip_flag = False
            for skip_token in self.tokenizer.ignore_list: # Skipping tokens in ignore list
                if skip_token in u or skip_token in v:
                    skip_flag = True
            if u != 'it' and v != 'it' and not skip_flag:
                r = str(rule[1]).lower() # Relation between node u and v is stored in r.
                if not rule[1] or rule[1] == '': # If there is no relation, then r is set to relatedTo.
                    r = 'relatedTo'
                current_graph.add_edge(str(rule[0]).lower(), str(rule[2]).lower(), relation=r) # Add edges between node u and v with relation r.
        prev_edges = current_graph.edges
        if prev_room_subgraph is not None:
            current_graph.add_edges_from(prev_room_subgraph.edges) # Add graph from current room.
        current_edges = current_graph.edges # Edges list is updated.
        return current_graph, add_rules # Returns current graph and rules.


def khop_neighbor_graph(graph, entities, cutoff=1, max_khop_degree=None): # Returns a graph with all k-hop neighborhood nodes of the given entities.
    all_entities = []
    for et in entities:
        candidates = nx.single_source_shortest_path(graph, et, cutoff=cutoff).keys() # Finds shortest path candidates between entities with cutoff.
        if not max_khop_degree or len(candidates)<=max_khop_degree: # Filters eligible candidates from set of candidates available within k-hops .
            all_entities.extend(list(candidates))
    return graph.subgraph(set(entities)|set(all_entities)) # Returning the graph


def ego_graph_seed_expansion(graph, seed, radius, undirected=True, max_degree=None):
    working_graph = graph
    if undirected:
        working_graph = graph.to_undirected()
    marked = set(seed)
    nodes = set(seed)

    for _ in range(radius):
        border = set()
        for node in marked:
            neighbors = {n for n in working_graph[node]}
            if max_degree is None or len(neighbors) <= max_degree:
                border |= neighbors
        nodes |= border
        marked = border

    return graph.subgraph(nodes)


def shortest_path_seed_expansion(graph, seed, cutoff=None, undirected=True, keep_all=True):
    nodes = set(seed)
    seed = list(seed)

    working_graph = graph
    if undirected:
        working_graph = graph.to_undirected()
    for i in range(len(seed)):
        start = i + 1 if undirected else 0
        for j in range(start, len(seed)):
            try:
                if not keep_all:
                    path = nx.shortest_path(working_graph, seed[i], seed[j])
                    if cutoff is None or len(path) <= cutoff:
                        nodes |= set(path)
                else:
                    paths = nx.all_shortest_paths(working_graph, seed[i], seed[j])
                    for p in paths:
                        if cutoff is None or len(p) <= cutoff:
                            nodes |= set(p)
            except nx.NetworkXNoPath:
                continue
    return graph.subgraph(nodes)


def load_manual_graphs(path):  # This function loads conceptnet manual subgraph for each game and returns it as dictionary.
    path = Path(path)
    manual_world_graphs = {}
    if not path.exists():
        print('None Found.')
        return manual_world_graphs

    files = path.rglob("conceptnet_manual_subgraph-*.tsv")
    for file in files:
        game_id = str(file).split('-')[-1].split('.')[0]
        graph, triplets, entities = construct_kg(file, cache_load=False)
        manual_world_graphs[game_id]={}
        manual_world_graphs[game_id]['graph'] = graph
        manual_world_graphs[game_id]['triplets'] = triplets
        manual_world_graphs[game_id]['entities'] = entities
    print(' DONE')
    return manual_world_graphs




def kg_match(extractor, target_entities, kg_entities):
    result = set()
    kg_entities = escape_entities(kg_entities)
    for e in target_entities:
        e = e.lower().strip()
        result |= extractor(e, kg_entities)
    return result


def save_graph_tsv(graph, path): # Saving graph as TSV file. Each line contains u,relation,v triplet.
    relation_map = nx.get_edge_attributes(graph, 'relation')
    lines = []
    for n1, n2 in graph.edges:
        relations = relation_map[n1, n2].split()
        for r in relations:
            lines.append(f'{n1}\t{r}\t{n2}\n')
    with open(path, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    from utils import extractor
    from utils.nlp import Tokenizer

    tk_extractor = extractor.get_extractor('max')
    tokenizer = Tokenizer(extractor=tk_extractor)
    rel_extract = RelationExtractor(tokenizer,openie_url='http://iqa962.sl.cloud9.ibm.com:9000/')
    # text = 'On the table, you see an apple, a hat, a key and an umbrella. '
    text = "You've just walked into a Living Room. You try to gain information on your " \
           "surroundings by using a technique you call looking. You can see a closet. " \
           "You idly wonder how they came up with the name TextWorld for this place. " \
           "It's pretty fitting. A closed standard looking antique trunk is in the room. " \
           "You can see a table. The table is usual. On the table you see an apple, a mug, " \
           "a newspaper, a note, a hat and a pencil. You smell a sickening smell, and follow " \
           "it to a couch. The couch is standard. But the thing is empty. Hm. Oh well You see a " \
           "gleam over in a corner, where you can see a tv stand. The tv stand is ordinary. " \
           "On the tv stand you can make out a tv. You don't like doors? Why not try going east, " \
           "that entranceway is unguarded. You are carrying nothing."
    sents = text
    # clauses = clausie.clausie(text)
    # propositions = clausie.extract_propositions(clauses)
    # sents = ''
    # for prop in propositions:
    #     sent = clausie.proposition_text_str(prop)
    #     sents += sent
    #     print(sent)
    graph, add_rules = rel_extract.fetch_triplets(sents, nx.DiGraph())
    print(add_rules)

