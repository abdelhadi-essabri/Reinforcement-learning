# Imports des bibliothèques nécessaires
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math

# Constantes
T = (12, 12)  # Position de l'état cible/terminal
K = 12  # Taille de la grille (KxK)

# Fonction qui génère les transitions possibles pour chaque état
def dict_transition(K, T, trap):
    """
    Renvoie un dictionnaire contenant les transitions possibles à partir de chaque position (i, j) de la grille.
    
    Args:
    - K (int): Dimension de la grille.
    - T (tuple): Position de la case terminale.
    - trap (bool): Si True, certains états deviennent des "pièges" avec des transitions limitées.
    
    Returns:
    - transitions (dict): Dictionnaire où chaque clé est une position (i, j) et chaque valeur un sous-dictionnaire des transitions possibles.
    """
    transitions = {}
    
    for i in range(1, K+1):
        for j in range(1, K+1):
            # Ignore les transitions si la case est un piège ou si elle est terminale
            if (i == K and j == K) or (trap and ((1 <= i <= 8 and j == 4) or (5 <= i <= 12 and j == 8))):
                continue
            
            # Initialisation des probabilités de déplacement à partir de la case (i, j)
            transitions[(i, j)] = {}
            if i != 1:   # Mouvement vers la gauche
                transitions[(i, j)]["left"] = (i - 1, j)
            if i != K:   # Mouvement vers la droite
                transitions[(i, j)]["right"] = (i + 1, j)
            if j != 1:   # Mouvement vers le bas
                transitions[(i, j)]["down"] = (i, j - 1)
            if j != K:   # Mouvement vers le haut
                transitions[(i, j)]["up"] = (i, j + 1)
                
    return transitions

# Création des transitions pour une grille avec ou sans pièges
transitions = dict_transition(12, (12, 12), False)
transitions_trap = dict_transition(12, (12, 12), True)

# Fonction qui génère les récompenses pour chaque état
def dict_rewards(K, T, trap):
    """
    Renvoie un dictionnaire contenant les récompenses associées à chaque position de la grille.
    
    Args:
    - K (int): Dimension de la grille.
    - T (tuple): Position de la case terminale.
    - trap (bool): Si True, certains états deviennent des "pièges" avec des récompenses négatives.
    
    Returns:
    - rewards (dict): Dictionnaire où chaque clé est une position (i, j) et chaque valeur une récompense.
    """
    rewards = {}
    
    for i in range(1, K+1):
        for j in range(1, K+1):
            if trap and ((1 <= i <= 8 and j == 4) or (5 <= i <= 12 and j == 8)):
                rewards[(i, j)] = -2 * (K - 1)
            elif (i, j) != T:
                rewards[(i, j)] = -1
            else:
                rewards[(i, j)] = 2 * (K - 1)
                
    return rewards

# Création des récompenses pour une grille avec ou sans pièges
rewards = dict_rewards(12, (12, 12), False)
rewards_trap = dict_rewards(12, (12, 12), True)

# Fonction pour récupérer les états et les actions possibles
def state_action(K, trap):
    """
    Renvoie les états, actions, et paires (état, action) valides pour une grille donnée.
    
    Args:
    - K (int): Dimension de la grille.
    - trap (bool): Si True, certains états deviennent des "pièges" sans action possible.
    
    Returns:
    - states (list): Liste des états de la grille.
    - actions (list): Liste des actions possibles.
    - states_actions (list): Liste des paires (état, action) possibles.
    """
    states = [(i, j) for i in range(1, K+1) for j in range(1, K+1)]
    actions = ["up", "down", "left", "right"]
    
    # Définir les paires état-action en fonction des contraintes de la grille
    states_actions = []
    for state in states:
        i, j = state
        if state == T or (trap and ((1 <= i <= 8 and j == 4) or (5 <= i <= 12 and j == 8))):
            states_actions.append((state, None))
            continue
        if i != 1:
            states_actions.append((state, "left"))
        if i != K:
            states_actions.append((state, "right"))
        if j != 1:
            states_actions.append((state, "down"))
        if j != K:
            states_actions.append((state, "up"))
    
    return states, actions, states_actions

# Initialiser une fonction de valeur pour chaque état
def dict_function(states, T):
    """
    Initialise une fonction de valeur pour chaque état, assignant une valeur aléatoire.
    
    Args:
    - states (list): Liste des états de la grille.
    - T (tuple): Position de la case terminale.
    
    Returns:
    - value_function (dict): Dictionnaire où chaque clé est un état et chaque valeur la valeur associée.
    """
    value_function = {state: np.random.rand() * 10 for state in states}
    value_function[T] = 0  # La valeur de la case terminale est fixée à 0
    return value_function

# Fonction pour choisir une action selon les probabilités d'une politique
def choice(state, dict_pi, states_actions):
    """
    Choisit une action pour un état donné, basée sur les probabilités de la politique dict_pi.
    
    Args:
    - state (tuple): État actuel.
    - dict_pi (dict): Politique associant une probabilité à chaque paire état-action.
    - states_actions (list): Liste des paires état-action valides.
    
    Returns:
    - action (str): Action choisie pour l'état actuel.
    """
    if (state, None) in states_actions:
        return None
    e = random.uniform(0, 1)
    filtered_keys = [(s, a) for s, a in states_actions if s == state]
    proba = [dict_pi[filtered_keys[i]] for i in range(len(filtered_keys))]
    cumulative_sum = proba[0]
    if cumulative_sum > e:
        return filtered_keys[0][1]
    for i in range(1, len(filtered_keys)):
        cumulative_sum += proba[i]
        if cumulative_sum > e:
            return filtered_keys[i][1]
    return filtered_keys[-1][1]

# Construire un chemin en choisissant une action pour chaque état
def construct_path(pi, states_actions, transitions, S, T):
    """
    Construit un chemin depuis l'état initial en suivant la politique pi et les transitions.
    
    Args:
    - pi (dict): Politique de la grille.
    - states_actions (list): Liste des paires état-action.
    - transitions (dict): Dictionnaire des transitions possibles pour chaque état.
    - S (tuple): État initial.
    - T (tuple): État terminal.
    
    Returns:
    - pi_final (dict): Chemin construit.
    """
    state_no_absorbing = {s for s, a in states_actions if a is not None}
    pi_final = {}
    list_state = [S]
    
    while S in state_no_absorbing:
        a = choice(S, pi, states_actions)
        pi_final[S] = (a, transitions[S][a])
        S = transitions[S][a]
        if S in list_state:
            return pi_final
        list_state.append(S)
    return pi_final

# Obtenir le chemin de la politique optimale
def optimal_policy(pi, rewards, S, T, states_actions):
    """
    Retourne le chemin optimal et la récompense moyenne obtenue en suivant la politique pi.
    
    Args:
    - pi (dict): Politique de la grille.
    - rewards (dict): Récompenses pour chaque état.
    - S (tuple): État initial.
    - T (tuple): État terminal.
    - states_actions (list): Liste des paires état-action.
    
    Returns:
    - paths (list): Chemin suivi.
    - reward (float): Récompense moyenne obtenue.
    """
    state_no_absorbing = {s for s, a in states_actions if a is not None}
    paths = [S]
    reward = [0]
    cpt = 0
    
    while S in state_no_absorbing:
        if cpt > 100:
            return paths, -1e18  # Chemin infini
        paths.append(pi[S][1])
        reward.append(rewards[pi[S][1]])
        S = pi[S][1]
        if S in paths:
            return paths, -1e18  # Chemin infini
        cpt += 1
    
    reward.pop(0)
    return paths, np.mean(reward)

# Vous pouvez exécuter les fonctions ci-dessus pour tester l’optimisation et le comportement des transitions et récompenses.


# Créer une grille pour les fonctions de valeurs
def grid_function_value(dict_function_value, title):
    
    data_frame = pd.DataFrame([(k[0], k[1], v) for k, v in dict_function_value.items()], columns=['x', 'y', 'value function'])
    data_frame_pivot = data_frame.pivot(index='y', columns='x', values='value function')
    data_frame_pivot = data_frame_pivot.sort_index(ascending=False)
    # Créer une heatmap avec Seaborn
    plt.figure(figsize=(16, 16))
    sns.heatmap(data_frame_pivot, annot=True, fmt=".1f")
    plt.title(title + ": Value function" )
    #plt.savefig("graphs/" + title + ".png")
    plt.show()

# Fonction pour dessiner une flèche à une position donnée avec une direction donnée

# Calculer la récompense maximale pour un état précis key
def maximum(transitions, state, actions, rewards, gamma, v, states_actions):
    best_rewards = - np.inf
    for action in actions:
        if((state, action) in states_actions):
            new_state = transitions[state][action]
            somme = rewards[new_state] + gamma * v[new_state]
            #print("somme = ", somme)
            if(somme > best_rewards):
                best_rewards = somme
    v[state] = best_rewards
    #print("best_reward = ", best_rewards)
    return v[state]

# Calculer l'action qui maximise la fonction de valeur
def argmax(transitions, state, actions, rewards, gamma, v, pi, states_actions):
    best_action = ""
    best_rewards = - np.inf
    for action in actions:
        if((state, action) in states_actions):
            new_state = transitions[state][action]
            #print(rewards)
            somme = rewards[new_state] + gamma * v[new_state]
            if(somme > best_rewards):
                best_rewards = somme
                best_action = action
    pi[state] = (best_action, transitions[state][best_action])
    return pi[state]

# Initialisation de Q
def initilize_q(states_actions, T, state_no_absorbing):
    q = {}
    for state_action in states_actions:
        if(state_action[0] in state_no_absorbing):
            q[state_action] = np.random.random_sample()
        
        elif(state_action[0] == T):
            q[(state_action[0], None)] = 0
        else:
            q[(state_action[0], None)] = 0
    return q  

# Initialisation de pi
def initilize_pi(states_actions):
    pi = {}
    states = list(set([pair[0] for pair in states_actions if pair[1]!=None]))
    for state in states:
        paires = [pair[1] for pair in states_actions if pair[0] == state]
        index = random.randint(0, len(paires)-1)
        pi[state] = (paires[index], transitions[state][paires[index]])
    return pi  

# implementation of eps-greedy
def maximize_q(q_function, state, actions, alpha, transitions, dict_rewards, dict_pi, eps):

    best_a = ""
    best_q = - math.inf
    #print(best_q)
    
    """
    Prendre en compte les contraintes des états (par ex, on ne peut pas faire down ou left si on est dans l'état (1,1))
    """
    set_action = list(transitions[state].keys())
    #print("state = ", state, "  set_action = ", set_action)
    """
    Chercher l'action "a" maximisant la q_fonction
    """
    list_q = []
    for a in set_action:
        q = q_function[(state, a)]
        #q = calculate_q(q_function, state, a, alpha, phi, dict_prob, dict_rewards, dict_pi)
        if(q > best_q):
            best_q = q
            best_a = a
        list_q.append((q, a))
    #print(list_q)
    #print("best_action", best_a)
    if(best_a not in actions):
        print("problem: best_a = ", best_a)
    """
    Déterminer la politique de l'état state pour chaque action compatibles (avec l'état) 
    """
    for a in set_action:
        #print("a = ", a)
        if(a != best_a):
            dict_pi[(state, a)] = eps/len(set_action)
        else:
            dict_pi[(state, a)] = (1-eps) + eps/len(set_action)
    return dict_pi

# Calculer l'action qui maximise la fonction de valeur
def argmax_q(transitions, state, actions, rewards, gamma, q, pi, states_actions):
    
    set_action = list(transitions[state].keys())
    q_to_maximise = [q[(state, action)] for action in set_action]
    best_a = set_action[q_to_maximise.index(max(q_to_maximise))]
    
    pi[state] = (best_a, transitions[state][best_a])
    
    return pi[state]

def action_maximize_q(q, state):

    set_action = list(transitions[state].keys())
    q_to_maximise = [q[(state, action)] for action in set_action]
    best_a = set_action[q_to_maximise.index(max(q_to_maximise))]
    
    return best_a




