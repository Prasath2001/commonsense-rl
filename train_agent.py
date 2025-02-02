import argparse
import numpy as np
import random
from time import time
import torch
import pickle
import agent
from utils import extractor
from utils.generic import getUniqueFileHandler
from utils.kg import construct_kg, load_manual_graphs, RelationExtractor
from utils.textworld_utils import get_goal_graph
from utils.nlp import Tokenizer
from games import dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(agent, opt, random_action=False):  #opt - command line argument parser
    filter_examine_cmd = False
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.
    game_path = opt.game_dir + "/" + (
        str(opt.difficulty_level) + "/" + opt.mode  if opt.difficulty_level != '' else opt.game_dir + "/" + opt.mode ) # opt.mode = train,test,valid mode.
    manual_world_graphs = {}  # Contains subgraph related to the game.
    if opt.graph_emb_type and 'world' in opt.graph_type:  # graph_emb_type = (numberbatch,complex) , graph_type = (world,local)
        print("Loading Knowledge Graph ... ", end='')
        agent.kg_graph, _, _= construct_kg(game_path + '/conceptnet_subgraph.txt')  # construct_kg returns the undirected graph(from conceptnet_subgraph.txt),triplets,entities 
        # construct_kg found in utils/kg.py
        print(' DONE')
        # optional: Use complete or brief manually extracted conceptnet subgraph for the agent
        print("Loading Manual World Graphs ... ", end='')
        manual_world_graphs = load_manual_graphs(game_path + '/manual_subgraph_brief')  # Returning the graph file as a dictionary with graph, triplets and entities.
        # load_manual_graphs found in utils/kg.py

    if opt.game_name: # command line argument for game name - files with extension .ulx
        game_path = game_path + "/"+ opt.game_name

    env, game_file_names = dataset.get_game_env(game_path, infos_to_request, opt.max_step_per_episode, opt.batch_size,
                                                opt.mode, opt.verbose)  # Returns the Environment and game file names.
    # Get Goals as graphs
    goal_graphs = {}  # This dictionary contains goal graph for each game.
    for game_file in env.gamefiles:
        goal_graph = get_goal_graph(game_file)  # Goal graphs contains objects,locations to cleanup. Example - apple,fridge to cleanup (goal) for each game.
        # Function found in utils/textworld_utils.py
        if goal_graph:
            game_id = game_file.split('-')[-1].split('.')[0]
            goal_graphs[game_id] = goal_graph # Each game_id has a goal graph

    # Collect some statistics: nb_steps, final reward.
    total_games_count = len(game_file_names) # Total number of games
    game_identifiers, avg_moves, avg_scores, avg_norm_scores, max_poss_scores = [], [], [], [], [] # Collect these stats

    for no_episode in (range(opt.nepisodes)): #nepisodes = number of episodes is a command line argument
        if not random_action: # random action is a choice, function parameter in play function. Default set to False.
            random.seed(no_episode)
            np.random.seed(no_episode)
            torch.manual_seed(no_episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(no_episode)
            env.seed(no_episode)

        agent.start_episode(opt.batch_size) # batch size is a command line argument - denotes number of games per batch (default batch_size=1)
        # start episode is found in agent.py - this function resets the parameters - mode, no_of_episodes, transitions, stats for every episode.
        avg_eps_moves, avg_eps_scores, avg_eps_norm_scores = [], [], []
        num_games = total_games_count
        game_max_scores = []
        game_names = [] # This contains game id from game.metadata
        while num_games > 0:
            obs, infos = env.reset()  # Start new episode.
            if filter_examine_cmd: # This helps to remove examine and look command from admissible commands.
                for commands_ in infos["admissible_commands"]: # [open refri, take apple from refrigeration]
                    for cmd_ in [cmd for cmd in commands_ if cmd.split()[0] in ["examine", "look"]]:
                        commands_.remove(cmd_) # Removing examine and look command from admissible commands.

            batch_size = len(obs)  # obs = [ 'apple in found on the table....','dirty shoes is on the floor...',...], obs[0] is observation of 1st game.
            num_games -= len(obs)  # updating number of games left.
            game_goal_graphs = [None] * batch_size # Game goal graphs for each game of the batch is initialized as None.
            max_scores = [] # Max scores possible for each game in the batch is taken from the game file - infos['game'] - game.max_score is used below.
            game_ids = [] # ID for each game in the batch  is taken from game.metadata 
            game_manual_world_graph = [None] * batch_size # Manual world graph for each game of the batch is initialized as None.
            for b, game in enumerate(infos["game"]):
                max_scores.append(game.max_score) # Max scores appended from game file.
                if "uuid" in game.metadata:
                    game_id = game.metadata["uuid"].split("-")[-1] # Game id is taken from game file.
                    game_ids.append(game_id) # Collection of games id from the batch.
                    game_names.append(game_id)  # Collection of games id from the batch is appended to game_names.
                    game_max_scores.append(game.max_score) # Appending maximum possible score for each game of the batch.
                    if len(goal_graphs):
                        game_goal_graphs[b] = goal_graphs[game_id] # goal graphs is a dictionary with key - game id, value - goal graph.
                    if len(manual_world_graphs):
                        game_manual_world_graph[b] = manual_world_graphs[game_id] # manual world graph is a dictionary with key - game id, value - manual world graph.

            if not game_ids:
                game_ids = range(num_games,num_games+batch_size)
                game_names.extend(game_ids)

            commands = ["restart"]*len(obs) # All commands are first initialized to 'restart'. If command = restart, then previous action is None.
            scored_commands = [[] for b in range(batch_size)] # scored_commands = [ [1,0,2,..],[1,1,0..],...,[1,1,1..] ] - scored_commands[0] contains scores for 1st game of the batch.
            last_scores = [0.0]*len(obs) # last_scores contains last score for each game of the batch.
            scores = [0.0]*len(obs)
            dones = [False]*len(obs)
            nb_moves = [0]*len(obs)
            infos["goal_graph"] = game_goal_graphs # dictionary is set to infos['goal graph'] - (game_id,goal graph) - (key,value) pair. 
            infos["manual_world_graph"] = game_manual_world_graph # dictionary is set to infos['manual world graph'] - (game_id,manual world graph) - (key,value) pair.
            agent.reset_parameters(opt.batch_size)
            for step_no in range(opt.max_step_per_episode): # maximum steps per episode is a command line argument with default value = 50.
                nb_moves = [step + int(not done) for step, done in zip(nb_moves, dones)] # calculate number of moves till done = True.

                if agent.graph_emb_type and ('local' in agent.graph_type or 'world' in agent.graph_type): #graph_emb_type = 'numberbatch' or 'complex'.
                    # prune_nodes = opt.prune_nodes if no_episode >= opt.prune_episode and no_episode % 25 ==0 and step_no % 10 == 0 else False
                    prune_nodes = opt.prune_nodes # Prune low probability nodes in world graph. prune_nodes = 'true' or 'false'.
                    agent.update_current_graph(obs, commands, scored_commands, infos, opt.graph_mode, prune_nodes) # This updates the current local/world graph with current facts.

                commands = agent.act(obs, scores, dones, infos, scored_commands, random_action)  # Action function returns a command/action to perform.
                obs, scores, dones, infos = env.step(commands) # Action is performed.
                infos["goal_graph"] = game_goal_graphs # updating the goal graph after action is taken place.
                infos["manual_world_graph"] = game_manual_world_graph # updating the manual world graph after action is taken place.

                for b in range(batch_size):  # For each game of the batch
                    if scores[b] - last_scores[b] > 0: # Checking for improvement in score, if yes scores are updated with last scores.
                        last_scores[b] = scores[b]  # Scores are updated after rewards.
                        scored_commands[b].append(commands[b]) # scored_commands[0] = [put apple in fridge, put dirty shirt in washing machine,...]

                if all(dones): # If all games are done, break
                    break
                if step_no == opt.max_step_per_episode - 1: # No. of maximum steps allowed in a episode.
                    dones = [True for _ in dones]  # Making all the dones = True after reaching the maximum episodes limit.
            agent.act(obs, scores, dones, infos, scored_commands, random_action)  # Let the agent know the game is done.

            if opt.verbose:
                print(".", end="")
            avg_eps_moves.extend(nb_moves) # appending number of moves.
            avg_eps_scores.extend(scores) # appending scores.
            avg_eps_norm_scores.extend([score/max_score for score, max_score in zip(scores, max_scores)]) # appending normalized scores.
        if opt.verbose:
            print("*", end="")
        agent.end_episode()  # End of episode resets parameters like episode_has_started = False etc... - found in agent.py
        game_identifiers.append(game_names)
        avg_moves.append(avg_eps_moves) # episode x # games  Aggregating avg. moves across episodes.
        avg_scores.append(avg_eps_scores) # Aggregating avg. scores across episodes.
        avg_norm_scores.append(avg_eps_norm_scores) # Aggregating avg. normalized scores across episodes.
        max_poss_scores.append(game_max_scores) # Aggregating maximum possible scores across episodes.
    env.close() # Closing the environment.
    game_identifiers = np.array(game_identifiers) # Converting list to numpy array.
    avg_moves = np.array(avg_moves)
    avg_scores = np.array(avg_scores)
    avg_norm_scores = np.array(avg_norm_scores)
    max_poss_scores = np.array(max_poss_scores)
    if opt.verbose: # Printing statistics for each iteration with a for loop inside.
        idx = np.apply_along_axis(np.argsort, axis=1, arr=game_identifiers)
        game_avg_moves = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_moves))), axis=0)
        game_norm_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_norm_scores))), axis=0)
        game_avg_scores = np.mean(np.array(list(map(lambda x, y: y[x], idx, avg_scores))), axis=0)

        msg = "\nGame Stats:\n-----------\n" + "\n".join(
            "  Game_#{} = Score: {:5.2f} Norm_Score: {:5.2f} Moves: {:5.2f}/{}".format(game_no,avg_score,
                                                                                            norm_score, avg_move,
                                                                                            opt.max_step_per_episode)
            for game_no, (norm_score, avg_score, avg_move) in
            enumerate(zip(game_norm_scores, game_avg_scores, game_avg_moves)))

        print(msg)

        total_avg_moves = np.mean(game_avg_moves)
        total_avg_scores = np.mean(game_avg_scores)
        total_norm_scores = np.mean(game_norm_scores)
        msg = opt.mode+" stats: avg. score: {:4.2f}; norm. avg. score: {:4.2f}; avg. steps: {:5.2f}; \n"
        print(msg.format(total_avg_scores, total_norm_scores,total_avg_moves))

        ## Dump log files ......
        str_result = {opt.mode + 'game_ids': game_identifiers, opt.mode + 'max_scores': max_poss_scores,
                      opt.mode + 'scores_runs': avg_scores, opt.mode + 'norm_score_runs': avg_norm_scores,
                      opt.mode + 'moves_runs': avg_moves}

        results_ofile = getUniqueFileHandler(opt.results_filename + '_' +opt.mode+'_results')
        pickle.dump(str_result, results_ofile)
    return avg_scores, avg_norm_scores, avg_moves  # play function returns avg. scores, avg. normalized scores, avg. moves


if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser(add_help=False)

    # game files and other directories
    parser.add_argument('--game_dir', default='./games/twc', help='Location of the game e.g ./games/testbed')
    parser.add_argument('--game_name', help='Name of the game file e.g., kitchen_cleanup_10quest_1.ulx, *.ulx, *.z8')
    parser.add_argument('--results_dir', default='./results', help='Path to the results files')
    parser.add_argument('--logs_dir', default='./logs', help='Path to the logs files')

    # optional arguments (if game_name is given) for game files
    parser.add_argument('--batch_size', type=int, default='1', help='Number of the games per batch')
    parser.add_argument('--difficulty_level', default='easy', choices=['easy','medium', 'hard'],
                        help='difficulty level of the games')

    # Experiments
    parser.add_argument('--initial_seed', type=int, default=42)
    parser.add_argument('--nruns', type=int, default=5)
    parser.add_argument('--runid', type=int, default=0)
    parser.add_argument('--no_train_episodes', type=int, default=100)
    parser.add_argument('--no_eval_episodes', type=int, default=5)
    parser.add_argument('--train_max_step_per_episode', type=int, default=50)
    parser.add_argument('--eval_max_step_per_episode', type=int, default=50)
    parser.add_argument('--verbose', action='store_true', default=True)

    parser.add_argument('--hidden_size', type=int, default=300, help='num of hidden units for embeddings')
    parser.add_argument('--hist_scmds_size', type=int, default=3,
                help='Number of recent scored command history to use. Useful when the game has intermediate reward.')
    parser.add_argument('--ngram', type=int, default=3)
    parser.add_argument('--token_extractor', default='max', help='token extractor: (any or max)')
    parser.add_argument('--corenlp_url', default='http://localhost:9000/',
                        help='URL for Stanford CoreNLP OpenIE Server for the relation extraction for the local graph')

    parser.add_argument('--noun_only_tokens', action='store_true', default=False,
                        help=' Allow only noun for the token extractor')
    parser.add_argument('--use_stopword', action='store_true', default=False,
                        help=' Use stopwords for the token extractor')
    parser.add_argument('--agent_type', default='knowledgeaware', choices=['random','simple', 'knowledgeaware'],
                        help='Agent type for the text world: (random, simple, knowledgeable)')
    parser.add_argument('--graph_type', default='', choices=['', 'local','world'],
                        help='What type of graphs to be generated')
    parser.add_argument('--graph_mode', default='evolve', choices=['full', 'evolve'],
                        help='Give Full ground truth graph or evolving knowledge graph: (full, evolve)')
    parser.add_argument('--local_evolve_type', default='direct', choices=['direct', 'ground'],
                        help='Type of the generated/evolving strategy for local graph')
    parser.add_argument('--world_evolve_type', default='cdc',
                        choices=['DC','CDC', 'NG','NG+prune','manual'],
                        help='Type of the generated/evolving strategy for world graph')
    parser.add_argument('--prune_nodes', action='store_true', default=False,
                        help=' Allow pruning of low-probability nodes in the world-graph')
    parser.add_argument('--prune_start_episode', type=int, default=1, help='Starting the pruning from this episode')

    # Embeddings
    parser.add_argument('--emb_loc', default='embeddings/', help='Path to the embedding location')
    parser.add_argument('--word_emb_type', default='glove',
                        help='Embedding type for the observation and the actions: ...'
                             '(random, glove, numberbatch, fasttext). Use utils.generic.load_embedings ...'
                             ' to take car of the custom embedding locations')
    parser.add_argument('--graph_emb_type', help='Knowledge Graph Embedding type for actions: (numberbatch, complex)')
    parser.add_argument('--egreedy_epsilon', type=float, default=0.0, help="Epsilon for the e-greedy exploration")

    opt = parser.parse_args() # Command line argument parser.
    print(opt)
    random.seed(opt.initial_seed)
    np.random.seed(opt.initial_seed)
    torch.manual_seed(opt.initial_seed)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.initial_seed)
        torch.backends.cudnn.deterministic = True
    # yappi.start()

    scores_runs = []
    norm_score_runs = []
    moves_runs = []
    test_scores_runs = []
    test_norm_score_runs = []
    test_moves_runs = []

    random_action = False
    if opt.agent_type == 'random':
        random_action = True
        opt.graph_emb_type = None
    if opt.agent_type == 'simple':
        opt.graph_type = ''
        opt.graph_emb_type = None

    # Reset prune start episodes if pruning is not selected
    if not opt.prune_nodes:
        opt.prune_start_episode = opt.no_train_episodes

    tk_extractor = extractor.get_extractor(opt.token_extractor)

    results_filename = opt.results_dir + '/' + opt.agent_type + '_' + opt.game_dir.split('/')[-1] + '_' + (
        opt.graph_mode + '_' + opt.graph_type + '_' if opt.graph_type else '') + (
                           str(opt.word_emb_type) + '_' if opt.word_emb_type else '') + (
                           str(opt.graph_emb_type) + '-' if opt.graph_emb_type else '') + str(
        opt.nruns) + 'runs_' + str(opt.no_train_episodes) + 'episodes_' + str(opt.hist_scmds_size) + 'hsize_' + str(
        opt.egreedy_epsilon) + 'eps_' + opt.difficulty_level+'_' +  opt.local_evolve_type+'_' +  opt.world_evolve_type + '_' + str(opt.runid) + 'runId'
    opt.results_filename = results_filename
    graph = None
    seeds = [random.randint(1, 100) for _ in range(opt.nruns)]
    for n in range(0, opt.nruns):
        opt.run_no = n
        opt.seed = seeds[n]
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)  # For reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opt.seed)

        tokenizer = Tokenizer(noun_only_tokens=opt.noun_only_tokens, use_stopword=opt.use_stopword, ngram=opt.ngram,
                              extractor=tk_extractor)
        rel_extractor = RelationExtractor(tokenizer, openie_url=opt.corenlp_url) # tokenizer and rel_extractor are objects passed to agent.
        myagent = agent.KnowledgeAwareAgent(graph, opt, tokenizer,rel_extractor, device)
        myagent.type = opt.agent_type # Agent type is random, simple (Only Text), knowledge-aware (Text + Commonsense).

        print("Training ...")
        myagent.train(opt.batch_size)  # Tell the agent it should update its parameters.
        opt.mode = "train"
        opt.nepisodes = opt.no_train_episodes  # for training
        opt.max_step_per_episode=opt.train_max_step_per_episode
        starttime = time()
        print("\n RUN ", n, "\n")
        scores, norm_scores, moves = play(myagent, opt, random_action=random_action)
        print("Trained in {:.2f} secs".format(time() - starttime))

        # Save train model
        torch.save(myagent.model.state_dict(), getUniqueFileHandler(results_filename, ext='.pt'))

