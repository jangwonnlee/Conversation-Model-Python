#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse, json, copy, os
import cPickle as pickle
from easydict import EasyDict as edict
from collections import defaultdict
import random

args = edict(
    {      
        'agt': 9,
        'usr': 1,
        'max_turn': 40,
        'movie_kb_path': './movie_kb.1k.p',
        'dqn_hidden_size': 80,
        'experience_replay_pool_size': 1000,
        'episodes': 500,
        'simulation_epoch_size': 100,
        'write_model_dir': './',
        'run_mode': 3,
        'auto_suggest': 0,
        'act_level': 1,
        'slot_err_prob': 0.00,
        'intent_err_prob': 0.00,
        'batch_size': 16,
        'goal_file_path': './user_goals_first_turn_template.part.movie.v1.p',
        'warm_start': 1,
        'warm_start_epochs': 120,
        'dict_path': './dicts.v3.p',
        'act_set': './dia_acts.txt',
        'slot_set': './slot_set.txt',
        'epsilon': 0,
        'gamma': 0.9,
        'predict_mode': False,
        'trained_model_path': None,
        'cmd_input_mode': 0,
        'slot_err_mode': 0,
        'learning_phase': 'all',
        'nlg_model_path': './lstm_tanh_relu_[1468202263.38]_2_0.610.p',
        'diaact_nl_pairs': './dia_act_nl_pairs.v6.json',
        'nlu_model_path': './lstm_[1468447442.91]_39_80_0.921.p',
        'success_rate_threshold': 0.3,
        'save_check_point': 10
    }
)


# In[2]:


params = vars(args)


# In[3]:


print('Dialog Parameters: ')
print(json.dumps(params, indent=2))


# In[4]:


max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

dict_path = params['dict_path']
goal_file_path = params['goal_file_path']

# load the user goals from .p file
all_goal_set = pickle.load(open(goal_file_path, 'rb'))


# In[5]:


# split goal set
split_fold = params.get('split_fold', 5)
goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
    else: goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set


# In[6]:


movie_kb_path = params['movie_kb_path']
movie_kb = pickle.load(open(movie_kb_path, 'rb'))


# In[7]:


def text_to_dict(path):
    """ Read in a text file as a dictionary where keys are text and values are indices (line numbers) """
    
    slot_set = {}
    with open(path, 'r') as f:
        index = 0
        for line in f.readlines():
            slot_set[line.strip('\n').strip('\r')] = index
            index += 1
    return slot_set


# In[8]:


act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])


# In[9]:


movie_dictionary = pickle.load(open(dict_path, 'rb'))

run_mode = params['run_mode']
auto_suggest = params['auto_suggest']


# In[10]:


agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']

agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']


# In[11]:


class Agent:
    """ Prototype for all agent classes, defining the interface they must uphold """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        """ Constructor for the Agent class
        Arguments:
        movie_dict      --  This is here now but doesn't belong - the agent doesn't know about movies
        act_set         --  The set of acts. #### Shouldn't this be more abstract? Don't we want our agent to be more broadly usable?
        slot_set        --  The set of available slots
        """
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        self.current_action = {}                    #   TODO Changed this variable's name to current_action
        self.current_action['diaact'] = None        #   TODO Does it make sense to call it a state if it has an act? Which act? The Most recent?
        self.current_action['inform_slots'] = {}
        self.current_action['request_slots'] = {}
        self.current_action['turn'] = 0

    def state_to_action(self, state, available_actions):
        """ Take the current state and return an action according to the current exploration/exploitation policy
        We define the agents flexibly so that they can either operate on act_slot representations or act_slot_value representations.
        We also define the responses flexibly, returning a dictionary with keys [act_slot_response, act_slot_value_response]. This way the command-line agent can continue to operate with values
        Arguments:
        state      --   A tuple of (history, kb_results) where history is a sequence of previous actions and kb_results contains information on the number of results matching the current constraints.
        user_action         --   A legacy representation used to run the command line agent. We should remove this ASAP but not just yet
        available_actions   --   A list of the allowable actions in the current state
        Returns:
        act_slot_action         --   An action consisting of one act and >= 0 slots as well as which slots are informed vs requested.
        act_slot_value_action   --   An action consisting of acts slots and values in the legacy format. This can be used in the future for training agents that take value into account and interact directly with the database
        """
        act_slot_response = None
        act_slot_value_response = None
        return {"act_slot_response": act_slot_response, "act_slot_value_response": act_slot_value_response}


    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """  Register feedback from the environment, to be stored as future training data
        Arguments:
        s_t                 --  The state in which the last action was taken
        a_t                 --  The previous agent action
        reward              --  The reward received immediately following the action
        s_tplus1            --  The state transition following the latest action
        episode_over        --  A boolean value representing whether the this is the final action.
        Returns:
        None
        """
        pass
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
     
       
    def add_nl_to_action(self, agent_action):
        """ Add NL to Agent Dia_Act """
        
        if agent_action['act_slot_response']:
            agent_action['act_slot_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_response'], 'agt') #self.nlg_model.translate_diaact(agent_action['act_slot_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence
        elif agent_action['act_slot_value_response']:
            agent_action['act_slot_value_response']['nl'] = ""
            user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(agent_action['act_slot_value_response'], 'agt') #self.nlg_model.translate_diaact(agent_action['act_slot_value_response']) # NLG
            agent_action['act_slot_response']['nl'] = user_nlg_sentence


# In[12]:


sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']

feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_question actions
    ############################################################################
    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_answer actions
    ############################################################################
    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   deny actions
    ############################################################################
    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
]

############################################################################
#   Adding the inform actions
############################################################################
for slot in sys_inform_slots:
    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

############################################################################
#   Adding the request actions
############################################################################
for slot in sys_request_slots:
    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})


# In[13]:


import math
import numpy as np

def initWeight(n,d):
    scale_factor = math.sqrt(float(6)/(n + d))
    #scale_factor = 0.1
    return (np.random.rand(n,d)*2-1)*scale_factor

def initWeights(n,d):
    """ Initialization Strategy """
    #scale_factor = 0.1
    scale_factor = math.sqrt(float(6)/(n + d))
    return (np.random.rand(n,d)*2-1)*scale_factor

nlg_beam_size = 10
NO_OUTCOME_YET = 0
CONSTRAINT_CHECK_FAILURE = 0
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
I_DO_NOT_CARE = "I do not care"
TICKET_AVAILABLE = 'Ticket Available'

FAILED_DIALOG = -1
SUCCESS_DIALOG = 1

CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
}

def mergeDicts(d0, d1):
    for k in d1:
        if k in d0:
            d0[k] += d1[k]
        else:
            d0[k] = d1[k]


# In[14]:


class DQN:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # input-hidden
        self.model['Wxh'] = initWeight(input_size, hidden_size)
        self.model['bxh'] = np.zeros((1, hidden_size))
      
        # hidden-output
        self.model['Wd'] = initWeight(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wxh', 'bxh', 'Wd', 'bd']
        self.regularize = ['Wxh', 'Wd']

        self.step_cache = {}
        

    def getStruct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """Activation Function: Sigmoid, or tanh, or ReLu"""
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        active_func = params.get('activation_func', 'relu')
 
        # input layer to hidden layer
        Wxh = self.model['Wxh']
        bxh = self.model['bxh']
        Xsh = Xs.dot(Wxh) + bxh
        
        hidden_size = self.model['Wd'].shape[0] # size of hidden layer
        H = np.zeros((1, hidden_size)) # hidden layer representation
        
        if active_func == 'sigmoid':
            H = 1/(1+np.exp(-Xsh))
        elif active_func == 'tanh':
            H = np.tanh(Xsh)
        elif active_func == 'relu': # ReLU 
            H = np.maximum(Xsh, 0)
        else: # no activation function
            H = Xsh
                
        # decoder at the end; hidden layer to output layer
        Wd = self.model['Wd']
        bd = self.model['bd']
        Y = H.dot(Wd) + bd
        
        # cache the values in forward pass, we expect to do a backward pass
        cache = {}
        if not predict_mode:
            cache['Wxh'] = Wxh
            cache['Wd'] = Wd
            cache['Xs'] = Xs
            cache['Xsh'] = Xsh
            cache['H'] = H
            
            cache['bxh'] = bxh
            cache['bd'] = bd
            cache['activation_func'] = active_func
            
            cache['Y'] = Y
            
        return Y, cache
    
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        H = cache['H']
        Xs = cache['Xs']
        Xsh = cache['Xsh']
        Wxh = cache['Wxh']
 
        active_func = cache['activation_func']
        n,d = H.shape
        
        dH = dY.dot(Wd.transpose())
        # backprop the decoder
        dWd = H.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims=True)
        
        dXsh = np.zeros(Xsh.shape)
        dXs = np.zeros(Xs.shape)
        
        if active_func == 'sigmoid':
            dH = (H-H**2)*dH
        elif active_func == 'tanh':
            dH = (1-H**2)*dH
        elif active_func == 'relu':
            dH = (H>0)*dH # backprop ReLU
        else:
            dH = dH
              
        # backprop to the input-hidden connection
        dWxh = Xs.transpose().dot(dH)
        dbxh = np.sum(dH, axis=0, keepdims = True)
        
        # backprop to the input
        dXsh = dH
        dXs = dXsh.dot(Wxh.transpose())
        
        return {'Wd': dWd, 'bd': dbd, 'Wxh':dWxh, 'bxh':dbxh}
    
    
    """batch Forward & Backward Pass"""
    def batchForward(self, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Xs = np.array([x['cur_states']], dtype=float)
            
            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchDoubleForward(self, batch, params, clone_dqn, predict_mode = False):
        caches = []
        Ys = []
        tYs = []
        
        for i,x in enumerate(batch):
            Xs = x[0]
            Y, out_cache = self.fwdPass(Xs, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
            
            tXs = x[3]
            tY, t_cache = clone_dqn.fwdPass(tXs, params, predict_mode = False)
                
            tYs.append(tY)
            
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache, tYs
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ cost function, returns cost and gradients for model """
    def costFunc(self, batch, params, clone_dqn):
        regc = params.get('reg_cost', 1e-3)
        gamma = params.get('gamma', 0.9)
        
        # batch forward
        Ys, caches, tYs = self.batchDoubleForward(batch, params, clone_dqn, predict_mode = False)
        
        loss_cost = 0.0
        dYs = []
        for i,x in enumerate(batch):
            Y = Ys[i]
            nY = tYs[i]
            
            action = np.array(x[1], dtype=int)
            reward = np.array(x[2], dtype=float)
            
            n_action = np.nanargmax(nY[0])
            max_next_y = nY[0][n_action]
            
            eposide_terminate = x[4]
            
            target_y = reward
            if eposide_terminate != True: target_y += gamma*max_next_y
            
            pred_y = Y[0][action]
            
            nY = np.zeros(nY.shape)
            nY[0][action] = target_y
            Y = np.zeros(Y.shape)
            Y[0][action] = pred_y
            
            # Cost Function
            loss_cost += (target_y - pred_y)**2
                
            dY = -(nY - Y)
            #dY = np.minimum(dY, 1)
            #dY = np.maximum(dY, -1)
            dYs.append(dY)
        
        # backprop the RNN
        grads = self.batchBackward(dYs, caches)
        
        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:    
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, batch, params, clone_dqn):
        learning_rate = params.get('learning_rate', 0.001)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0.1)
        grad_clip = params.get('grad_clip', -1e-3)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')
        activation_func = params.get('activation_func', 'relu')
        
        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(batch, params, clone_dqn)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if activation_func.lower() == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0:
                        dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else:
                        dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        out = {}
        out['cost'] = cost
        return out
    
    """ prediction """
    def predict(self, Xs, params, **kwargs):
        Ys, caches = self.fwdPass(Xs, params, predict_model=True)
        pred_action = np.argmax(Ys)
        
        return pred_action


# In[15]:


class AgentDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())
        
        self.feasible_actions = feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool = [] #experience replay pool <s_t, a_t, r_t, s_t+1>
        
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn
        
        self.dqn = DQN(self.state_dimension, self.hidden_size, self.num_actions)
        self.clone_dqn = copy.deepcopy(self.dqn)
        
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2
            
            
    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
    
    
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        user_act_rep[0,self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation
      
    def run_policy(self, representation):
        """ epsilon-greedy policy """
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.dqn.predict(representation, {}, predict_model=True)
    
    def rule_policy(self):
        """ Rule Policy """
        
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }
                
        return self.action_index(act_slot_response)
    
    def action_index(self, act_slot_response):
        """ Return the index of action """
        
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None
    
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register feedback from the environment, to be stored as future training data """
        
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            self.experience_replay_pool.append(training_example)
    
    def train(self, batch_size=1, num_batches=100):
        """ Train DQN with experience replay """
        
        for iter_batch in range(num_batches):
            self.cur_bellman_err = 0
            for iter in range(len(self.experience_replay_pool)/(batch_size)):
                batch = [random.choice(self.experience_replay_pool) for i in xrange(batch_size)]
                batch_struct = self.dqn.singleBatch(batch, {'gamma': self.gamma}, self.clone_dqn)
                self.cur_bellman_err += batch_struct['cost']['total_cost']
            
            print ("cur bellman err %.4f, experience replay pool %s" % (float(self.cur_bellman_err)/len(self.experience_replay_pool), len(self.experience_replay_pool)))
            
            
    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """
        
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path, )
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path, )
            print e         
    
    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""
        
        self.experience_replay_pool = pickle.load(open(path, 'rb'))
    
             
    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """
        
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        
        print "trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model


# In[16]:


if agt == 0:
    agent = AgentCmd(movie_kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 4:
    agent = EchoAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 5:
    agent = RequestBasicsAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 9:
    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
else:
    pass


# In[17]:


usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']


# In[18]:


class UserSimulator:
    """ Parent class for all user sims to inherit from """

    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog)"""

        print "initialize episode called, generating goal"
        self.goal =  random.choice(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1),' but we just started'
        return user_action


    def next(self, system_action):
        pass
    
    
    
    def set_nlg_model(self, nlg_model):
        self.nlg_model = nlg_model  
    
    def set_nlu_model(self, nlu_model):
        self.nlu_model = nlu_model
    
    
    
    def add_nl_to_action(self, user_action):
        """ Add NL to User Dia_Act """
        
        user_nlg_sentence = self.nlg_model.convert_diaact_to_nl(user_action, 'usr')
        user_action['nl'] = user_nlg_sentence
        
        if self.simulator_act_level == 1:
            user_nlu_res = self.nlu_model.generate_dia_act(user_action['nl']) # NLU
            if user_nlu_res != None:
                #user_nlu_res['diaact'] = user_action['diaact'] # or not?
                user_action.update(user_nlu_res)


# In[19]:


class RuleSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """
    
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """
        
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        
        self.max_turn = params['max_turn']
        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        
        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        
        self.learning_phase = params['learning_phase']
    
    def initialize_episode(self):
        """ Initialize a new episode (dialog) 
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """
        
        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        
        self.episode_over = False
        self.dialog_status = NO_OUTCOME_YET
        
        #self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = CONSTRAINT_CHECK_FAILURE
  
        """ Debug: build a fake goal mannually """
        #self.debug_falk_goal()
        
        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1),' but we just started'
        return user_action  
        
    def _sample_action(self):
        """ randomly sample a start action based on user goal """
        
        self.state['diaact'] = random.choice(start_dia_acts.keys())
        
        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(self.goal['inform_slots'].keys())
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal['inform_slots'].keys(): # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']
                
            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)
        
        self.state['rest_slots'].extend(self.goal['request_slots'].keys())
        
        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'
        
        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks','closing']): self.episode_over = True #episode_over = True
        else: self.episode_over = False #episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']
        
        self.add_nl_to_action(sample_action)
        return sample_action
    
    def _sample_goal(self, goal_set):
        """ sample a user goal  """
        
        sample_goal = random.choice(self.start_set[self.learning_phase])
        return sample_goal
    
    
    def corrupt(self, user_action):
        """ Randomly corrupt an action with error probs (slot_err_probability and slot_err_mode) on Slot and Intent (intent_err_probability). """
        
        for slot in user_action['inform_slots'].keys():
            slot_err_prob_sample = random.random()
            if slot_err_prob_sample < self.slot_err_probability: # add noise for slot level
                if self.slot_err_mode == 0: # replace the slot_value only
                    if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                elif self.slot_err_mode == 1: # combined
                    slot_err_random = random.random()
                    if slot_err_random <= 0.33:
                        if slot in self.movie_dict.keys(): user_action['inform_slots'][slot] = random.choice(self.movie_dict[slot])
                    elif slot_err_random > 0.33 and slot_err_random <= 0.66:
                        del user_action['inform_slots'][slot]
                        random_slot = random.choice(self.movie_dict.keys())
                        user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                    else:
                        del user_action['inform_slots'][slot]
                elif self.slot_err_mode == 2: #replace slot and its values
                    del user_action['inform_slots'][slot]
                    random_slot = random.choice(self.movie_dict.keys())
                    user_action[random_slot] = random.choice(self.movie_dict[random_slot])
                elif self.slot_err_mode == 3: # delete the slot
                    del user_action['inform_slots'][slot]
                    
        intent_err_sample = random.random()
        if intent_err_sample < self.intent_err_probability: # add noise for intent level
            user_action['diaact'] = random.choice(self.act_set.keys())
    
    def debug_falk_goal(self):
        """ Debug function: build a fake goal mannually (Can be moved in future) """
        
        self.goal['inform_slots'].clear()
        #self.goal['inform_slots']['city'] = 'seattle'
        self.goal['inform_slots']['numberofpeople'] = '2'
        #self.goal['inform_slots']['theater'] = 'amc pacific place 11 theater'
        #self.goal['inform_slots']['starttime'] = '10:00 pm'
        #self.goal['inform_slots']['date'] = 'tomorrow'
        self.goal['inform_slots']['moviename'] = 'zoology'
        self.goal['inform_slots']['distanceconstraints'] = 'close to 95833'
        self.goal['request_slots'].clear()
        self.goal['request_slots']['ticket'] = 'UNK'
        self.goal['request_slots']['theater'] = 'UNK'
        self.goal['request_slots']['starttime'] = 'UNK'
        self.goal['request_slots']['date'] = 'UNK'
        
    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        
        self.state['turn'] += 2
        self.episode_over = False
        self.dialog_status = NO_OUTCOME_YET
        
        sys_act = system_action['diaact']
        
        if (self.max_turn > 0 and self.state['turn'] > self.max_turn):
            self.dialog_status = FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots'])
            self.state['inform_slots'].clear()

            if sys_act == "inform":
                self.response_inform(system_action)
            elif sys_act == "multiple_choice":
                self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action) 
            elif sys_act == "thanks":
                self.response_thanks(system_action)
            elif sys_act == "confirm_answer":
                self.response_confirm_answer(system_action)
            elif sys_act == "closing":
                self.episode_over = True
                self.state['diaact'] = "thanks"

        self.corrupt(self.state)
        
        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']
        response_action['nl'] = ""
        
        # add NL to dia_act
        self.add_nl_to_action(response_action)                       
        return response_action, self.episode_over, self.dialog_status
    
    
    def response_confirm_answer(self, system_action):
        """ Response for Confirm_Answer (System Action) """
    
        if len(self.state['rest_slots']) > 0:
            request_slot = random.choice(self.state['rest_slots'])

            if request_slot in self.goal['request_slots'].keys():
                self.state['diaact'] = "request"
                self.state['request_slots'][request_slot] = "UNK"
            elif request_slot in self.goal['inform_slots'].keys():
                self.state['diaact'] = "inform"
                self.state['inform_slots'][request_slot] = self.goal['inform_slots'][request_slot]
                if request_slot in self.state['rest_slots']:
                    self.state['rest_slots'].remove(request_slot)
        else:
            self.state['diaact'] = "thanks"
            
    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """
        
        self.episode_over = True
        self.dialog_status = SUCCESS_DIALOG

        request_slot_set = copy.deepcopy(self.state['request_slots'].keys())
        if 'ticket' in request_slot_set:
            request_slot_set.remove('ticket')
        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
        if 'ticket' in rest_slot_set:
            rest_slot_set.remove('ticket')

        if len(request_slot_set) > 0 or len(rest_slot_set) > 0:
            self.dialog_status = FAILED_DIALOG

        for info_slot in self.state['history_slots'].keys():
            if self.state['history_slots'][info_slot] == NO_VALUE_MATCH:
                self.dialog_status = FAILED_DIALOG
            if info_slot in self.goal['inform_slots'].keys():
                if self.state['history_slots'][info_slot] != self.goal['inform_slots'][info_slot]:
                    self.dialog_status = FAILED_DIALOG

        if 'ticket' in system_action['inform_slots'].keys():
            if system_action['inform_slots']['ticket'] == dialog_config.NO_VALUE_MATCH:
                self.dialog_status = FAILED_DIALOG
                
        if self.constraint_check == CONSTRAINT_CHECK_FAILURE:
            self.dialog_status = FAILED_DIALOG
    
    def response_request(self, system_action):
        """ Response for Request (System Action) """
        
        if len(system_action['request_slots'].keys()) > 0:
            slot = system_action['request_slots'].keys()[0] # only one slot
            if slot in self.goal['inform_slots'].keys(): # request slot in user's constraints  #and slot not in self.state['request_slots'].keys():
                self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                self.state['diaact'] = "inform"
                if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
                self.state['request_slots'].clear()
            elif slot in self.goal['request_slots'].keys() and slot not in self.state['rest_slots'] and slot in self.state['history_slots'].keys(): # the requested slot has been answered
                self.state['inform_slots'][slot] = self.state['history_slots'][slot]
                self.state['request_slots'].clear()
                self.state['diaact'] = "inform"
            elif slot in self.goal['request_slots'].keys() and slot in self.state['rest_slots']: # request slot in user's goal's request slots, and not answered yet
                self.state['diaact'] = "request" # "confirm_question"
                self.state['request_slots'][slot] = "UNK"

                ########################################################################
                # Inform the rest of informable slots
                ########################################################################
                for info_slot in self.state['rest_slots']:
                    if info_slot in self.goal['inform_slots'].keys():
                        self.state['inform_slots'][info_slot] = self.goal['inform_slots'][info_slot]

                for info_slot in self.state['inform_slots'].keys():
                    if info_slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(info_slot)
            else:
                if len(self.state['request_slots']) == 0 and len(self.state['rest_slots']) == 0:
                    self.state['diaact'] = "thanks"
                else:
                    self.state['diaact'] = "inform"
                self.state['inform_slots'][slot] = I_DO_NOT_CARE
        else: # this case should not appear
            if len(self.state['rest_slots']) > 0:
                random_slot = random.choice(self.state['rest_slots'])
                if random_slot in self.goal['inform_slots'].keys():
                    self.state['inform_slots'][random_slot] = self.goal['inform_slots'][random_slot]
                    self.state['rest_slots'].remove(random_slot)
                    self.state['diaact'] = "inform"
                elif random_slot in self.goal['request_slots'].keys():
                    self.state['request_slots'][random_slot] = self.goal['request_slots'][random_slot]
                    self.state['diaact'] = "request"

    def response_multiple_choice(self, system_action):
        """ Response for Multiple_Choice (System Action) """
        
        slot = system_action['inform_slots'].keys()[0]
        if slot in self.goal['inform_slots'].keys():
            self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
        elif slot in self.goal['request_slots'].keys():
            self.state['inform_slots'][slot] = random.choice(system_action['inform_slots'][slot])

        self.state['diaact'] = "inform"
        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
        if slot in self.state['request_slots'].keys(): del self.state['request_slots'][slot]
        
    def response_inform(self, system_action):
        """ Response for Inform (System Action) """
        
        if 'taskcomplete' in system_action['inform_slots'].keys(): # check all the constraints from agents with user goal
            self.state['diaact'] = "thanks"
            #if 'ticket' in self.state['rest_slots']: self.state['request_slots']['ticket'] = 'UNK'
            self.constraint_check = CONSTRAINT_CHECK_SUCCESS
                    
            if system_action['inform_slots']['taskcomplete'] == NO_VALUE_MATCH:
                self.state['history_slots']['ticket'] = NO_VALUE_MATCH
                if 'ticket' in self.state['rest_slots']: self.state['rest_slots'].remove('ticket')
                if 'ticket' in self.state['request_slots'].keys(): del self.state['request_slots']['ticket']
                    
            for slot in self.goal['inform_slots'].keys():
                #  Deny, if the answers from agent can not meet the constraints of user
                if slot not in system_action['inform_slots'].keys() or (self.goal['inform_slots'][slot].lower() != system_action['inform_slots'][slot].lower()):
                    self.state['diaact'] = "deny"
                    self.state['request_slots'].clear()
                    self.state['inform_slots'].clear()
                    self.constraint_check = CONSTRAINT_CHECK_FAILURE
                    break
        else:
            for slot in system_action['inform_slots'].keys():
                self.state['history_slots'][slot] = system_action['inform_slots'][slot]
                        
                if slot in self.goal['inform_slots'].keys():
                    if system_action['inform_slots'][slot] == self.goal['inform_slots'][slot]:
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                                
                        if len(self.state['request_slots']) > 0:
                            self.state['diaact'] = "request"
                        elif len(self.state['rest_slots']) > 0:
                            rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                            if 'ticket' in rest_slot_set:
                                rest_slot_set.remove('ticket')

                            if len(rest_slot_set) > 0:
                                inform_slot = random.choice(rest_slot_set) # self.state['rest_slots']
                                if inform_slot in self.goal['inform_slots'].keys():
                                    self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                    self.state['diaact'] = "inform"
                                    self.state['rest_slots'].remove(inform_slot)
                                elif inform_slot in self.goal['request_slots'].keys():
                                    self.state['request_slots'][inform_slot] = 'UNK'
                                    self.state['diaact'] = "request"
                            else:
                                self.state['request_slots']['ticket'] = 'UNK'
                                self.state['diaact'] = "request"
                        else: # how to reply here?
                            self.state['diaact'] = "thanks" # replies "closing"? or replies "confirm_answer"
                    else: # != value  Should we deny here or ?
                        ########################################################################
                        # TODO When agent informs(slot=value), where the value is different with the constraint in user goal, Should we deny or just inform the correct value?
                        ########################################################################
                        self.state['diaact'] = "inform"
                        self.state['inform_slots'][slot] = self.goal['inform_slots'][slot]
                        if slot in self.state['rest_slots']: self.state['rest_slots'].remove(slot)
                else:
                    if slot in self.state['rest_slots']:
                        self.state['rest_slots'].remove(slot)
                    if slot in self.state['request_slots'].keys():
                        del self.state['request_slots'][slot]

                    if len(self.state['request_slots']) > 0:
                        request_set = list(self.state['request_slots'].keys())
                        if 'ticket' in request_set:
                            request_set.remove('ticket')

                        if len(request_set) > 0:
                            request_slot = random.choice(request_set)
                        else:
                            request_slot = 'ticket'

                        self.state['request_slots'][request_slot] = "UNK"
                        self.state['diaact'] = "request"
                    elif len(self.state['rest_slots']) > 0:
                        rest_slot_set = copy.deepcopy(self.state['rest_slots'])
                        if 'ticket' in rest_slot_set:
                            rest_slot_set.remove('ticket')

                        if len(rest_slot_set) > 0:
                            inform_slot = random.choice(rest_slot_set) #self.state['rest_slots']
                            if inform_slot in self.goal['inform_slots'].keys():
                                self.state['inform_slots'][inform_slot] = self.goal['inform_slots'][inform_slot]
                                self.state['diaact'] = "inform"
                                self.state['rest_slots'].remove(inform_slot)
                                        
                                if 'ticket' in self.state['rest_slots']:
                                    self.state['request_slots']['ticket'] = 'UNK'
                                    self.state['diaact'] = "request"
                            elif inform_slot in self.goal['request_slots'].keys():
                                self.state['request_slots'][inform_slot] = self.goal['request_slots'][inform_slot]
                                self.state['diaact'] = "request"
                        else:
                            self.state['request_slots']['ticket'] = 'UNK'
                            self.state['diaact'] = "request"
                    else:
                        self.state['diaact'] = "thanks" # or replies "confirm_answer"


# In[20]:


if usr == 0:# real user
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1: 
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
else:
    pass


# In[21]:


class decoder:
    def __init__(self, input_size, hidden_size, output_size):
        pass
    
    def get_struct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """ Activation Function: Sigmoid, or tanh, or ReLu"""
    def fwdPass(self, Xs, params, **kwargs):
        pass
    
    def bwdPass(self, dY, cache):
        pass
    
    
    """ Batch Forward & Backward Pass"""
    def batchForward(self, ds, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Y, out_cache = self.fwdPass(x, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ Cost function, returns cost and gradients for model """
    def costFunc(self, ds, batch, params):
        regc = params['reg_cost'] # regularization cost
        
        # batch forward RNN
        Ys, caches = self.batchForward(ds, batch, params, predict_mode = False)
        
        loss_cost = 0.0
        smooth_cost = 1e-15
        dYs = []
        
        for i,x in enumerate(batch):
            labels = np.array(x['labels'], dtype=int)
            
            # fetch the predicted probabilities
            Y = Ys[i]
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            P = e/np.sum(e, axis=1, keepdims=True)
            
            # Cross-Entropy Cross Function
            loss_cost += -np.sum(np.log(smooth_cost + P[range(len(labels)), labels]))
            
            for iy,y in enumerate(labels):
                P[iy,y] -= 1 # softmax derivatives
            dYs.append(P)
            
        # backprop the RNN
        grads = self.batchBackward(dYs, caches)
        
        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:    
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, ds, batch, params):
        learning_rate = params.get('learning_rate', 0.0)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0)
        grad_clip = params.get('grad_clip', 1)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')

        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(ds, batch, params)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if params['activation_func'] == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0: dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else: dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        # create output dict and return
        out = {}
        out['cost'] = cost
        return out
    
    
    """ Evaluate on the dataset[split] """
    def eval(self, ds, split, params):
        acc = 0
        total = 0
        
        total_cost = 0.0
        smooth_cost = 1e-15
        perplexity = 0
        
        for i, ele in enumerate(ds.split[split]):
            #ele_reps = self.prepare_input_rep(ds, [ele], params)
            #Ys, cache = self.fwdPass(ele_reps[0], params, predict_model=True)
            #labels = np.array(ele_reps[0]['labels'], dtype=int)
            
            Ys, cache = self.fwdPass(ele, params, predict_model=True)
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            labels = np.array(ele['labels'], dtype=int)
            
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            log_perplex = 0
            log_perplex += -np.sum(np.log2(smooth_cost + probs[range(len(labels)), labels]))
            log_perplex /= len(labels)
            
            loss_cost = 0
            loss_cost += -np.sum(np.log(smooth_cost + probs[range(len(labels)), labels]))
            
            perplexity += log_perplex #2**log_perplex
            total_cost += loss_cost
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            for index, l in enumerate(labels):
                if pred_words_indices[index] == l:
                    acc += 1
            
            total += len(labels)
            
        perplexity /= len(ds.split[split])    
        total_cost /= len(ds.split[split])
        accuracy = 0 if total == 0 else float(acc)/total
        
        #print ("perplexity: %s, total_cost: %s, accuracy: %s" % (perplexity, total_cost, accuracy))
        result = {'perplexity': perplexity, 'cost': total_cost, 'accuracy': accuracy}
        return result
    
    
         
    """ prediction on dataset[split] """
    def predict(self, ds, split, params):
        inverse_word_dict = {ds.data['word_dict'][k]:k for k in ds.data['word_dict'].keys()}
        for i, ele in enumerate(ds.split[split]):
            pred_ys, pred_words = self.forward(inverse_word_dict, ele, params, predict_model=True)
            
            sentence = ' '.join(pred_words[:-1])
            real_sentence = ' '.join(ele['sentence'].split(' ')[1:-1])
            
            if params['dia_slot_val'] == 2 or params['dia_slot_val'] == 3: 
                sentence = self.post_process(sentence, ele['slotval'], ds.data['slot_dict'])
            
            print 'test case', i
            print 'real:', real_sentence
            print 'pred:', sentence
    
    """ post_process to fill the slot """
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        sentence = pred_template
        suffix = "_PLACEHOLDER"
        
        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix
            if slot == 'result' or slot == 'numberofpeople': continue
            for slot_val in slot_vals:
                tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1)
                sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            for slot_val in slot_vals:
                tmp_sentence = sentence.replace(slot_placeholder, slot_val, 1)
                sentence = tmp_sentence
                
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
        
        return sentence


# In[22]:


class lstm_decoder_tanh(decoder):
    def __init__(self, diaact_input_size, input_size, hidden_size, output_size):
        self.model = {}
        # connections from diaact to hidden layer
        self.model['Wah'] = initWeights(diaact_input_size, 4*hidden_size)
        self.model['bah'] = np.zeros((1, 4*hidden_size))
        
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['Wah', 'bah', 'WLSTM', 'Wd', 'bd']
        self.regularize = ['Wah', 'WLSTM', 'Wd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        feed_recurrence = params.get('feed_recurrence', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        n, xd = Ws.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
    
        for t in xrange(n):
            prev = np.zeros(d) if t==0 else Hout[t-1]
            Hin[t,0] = 1 # bias
            Hin[t, 1:1+xd] = Ws[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            IFOG[t] = Hin[t].dot(WLSTM)
            
            # add diaact vector here
            if feed_recurrence == 0:
                if t == 0: IFOG[t] += Dsh[0]
            else:
                IFOG[t] += Dsh[0]

            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d])) # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh for input value
            
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
            
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['WLSTM'] = WLSTM
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Ds'] = Ds
            cache['Hin'] = Hin
            cache['Dsh'] = Dsh
            cache['Wah'] = Wah
            cache['feed_recurrence'] = feed_recurrence
            
        return Y, cache
    
    """ Forward pass on prediction """
    def forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        pred_y = []
        pred_words = []
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
            
        if decoder_sampling == 0: # sampling or argmax
            pred_y_index = np.nanargmax(Y)
        else:
            pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
        pred_y.append(pred_y_index)
        pred_words.append(dict[pred_y_index])
        
        time_stamp = 0
        while True:
            if dict[pred_y_index] == 'e_o_s' or time_stamp >= max_len: break
            
            X = np.zeros(xd)
            X[pred_y_index] = 1
            Hin[0,0] = 1 # bias
            Hin[0,1:1+xd] = X
            Hin[0, 1+xd:] = Hout[0]
            
            IFOG[0] = Hin[0].dot(WLSTM)
            if feed_recurrence == 1:
                IFOG[0] += Dsh[0]
        
            IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
            IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
            C = IFOGf[0, :d]*IFOGf[0, 3*d:]
            Cellin[0] = C + IFOGf[0, d:2*d]*Cellin[0]
            Cellout[0] = np.tanh(Cellin[0])
            Hout[0] = IFOGf[0, 2*d:3*d]*Cellout[0]
            
            Y = Hout.dot(Wd) + bd
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            if decoder_sampling == 0:
                pred_y_index = np.nanargmax(Y)
            else:
                pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
            pred_y.append(pred_y_index)
            pred_words.append(dict[pred_y_index])
            
            time_stamp += 1
            
        return pred_y, pred_words
    
    """ Forward pass on prediction with Beam Search """
    def beam_forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 30)
        feed_recurrence = params.get('feed_recurrence', 0)
        beam_size = params.get('beam_size', 10)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        Ds = Xs['diaact']
        Ws = Xs['words']
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = Ds.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = Ws.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = Ws[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        # keep a beam here
        beams = [] 
        
        Y = Hout.dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
        
        # add beam search here
        if decoder_sampling == 0: # no sampling
            beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        else:
            beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
        #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        for ele in beam_candidate_t:
            beams.append((np.log(probs[0][ele]), [ele], [dict[ele]], Hout[0], Cellin[0]))
        
        #beams.sort(key=lambda x:x[0], reverse=True)
        #beams.sort(reverse = True)
        
        time_stamp = 0
        while True:
            beam_candidates = []
            for b in beams:
                log_prob = b[0]
                pred_y_index = b[1][-1]
                cell_in = b[4]
                hout_prev = b[3]
                
                if b[2][-1] == "e_o_s": # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
        
                X = np.zeros(xd)
                X[pred_y_index] = 1
                Hin[0,0] = 1 # bias
                Hin[0,1:1+xd] = X
                Hin[0, 1+xd:] = hout_prev
                
                IFOG[0] = Hin[0].dot(WLSTM)
                if feed_recurrence == 1: IFOG[0] += Dsh[0]
        
                IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
                IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
                C = IFOGf[0, :d]*IFOGf[0, 3*d:]
                cell_in = C + IFOGf[0, d:2*d]*cell_in
                cell_out = np.tanh(cell_in)
                hout_prev = IFOGf[0, 2*d:3*d]*cell_out
                
                Y = hout_prev.dot(Wd) + bd
                maxes = np.amax(Y, axis=1, keepdims=True)
                e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
                probs = e/np.sum(e, axis=1, keepdims=True)
                
                if decoder_sampling == 0: # no sampling
                    beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                else:
                    beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
                #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                for ele in beam_candidate_t:
                    beam_candidates.append((log_prob+np.log(probs[0][ele]), np.append(b[1], ele), np.append(b[2], dict[ele]), hout_prev, cell_in))
            
            beam_candidates.sort(key=lambda x:x[0], reverse=True)
            #beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size]
            time_stamp += 1

            if time_stamp >= max_len: break
        
        return beams[0][1], beams[0][2]
    
    """ Backward Pass """
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        Ws = cache['Ws']
        Ds = cache['Ds']
        Dsh = cache['Dsh']
        Wah = cache['Wah']
        feed_recurrence = cache['feed_recurrence']
        
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        dWs = np.zeros(Ws.shape)
        
        dDsh = np.zeros(Dsh.shape)
        
        for t in reversed(xrange(n)):
            dIFOGf[t,2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t,2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t>0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t,d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t,3*d:] * dCellin[t]
            dIFOGf[t,3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0: dHout[t-1] += dHin[t,1+Ws.shape[1]:]
            
            if feed_recurrence == 0:
                if t == 0: dDsh[t] = dIFOG[t]
            else: 
                dDsh[0] += dIFOG[t]
        
        # backprop to the diaact-hidden connections
        dWah = Ds.transpose().dot(dDsh)
        dbah = np.sum(dDsh, axis=0, keepdims = True)
             
        return {'Wah':dWah, 'bah':dbah, 'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}
    
    
    """ Batch data representation """
    def prepare_input_rep(self, ds, batch, params):
        batch_reps = []
        for i,x in enumerate(batch):
            batch_rep = {}
            
            vec = np.zeros((1, self.model['Wah'].shape[0]))
            vec[0][x['diaact_rep']] = 1
            for v in x['slotrep']:
                vec[0][v] = 1
            
            word_arr = x['sentence'].split(' ')
            word_vecs = np.zeros((len(word_arr), self.model['Wxh'].shape[0]))
            labels = [0] * (len(word_arr)-1)
            for w_index, w in enumerate(word_arr[:-1]):
                if w in ds.data['word_dict'].keys():
                    w_dict_index = ds.data['word_dict'][w]
                    word_vecs[w_index][w_dict_index] = 1
                
                if word_arr[w_index+1] in ds.data['word_dict'].keys():
                    labels[w_index] = ds.data['word_dict'][word_arr[w_index+1]] 
            
            batch_rep['diaact'] = vec
            batch_rep['words'] = word_vecs
            batch_rep['labels'] = labels
            batch_reps.append(batch_rep)
        return batch_reps


# In[23]:


class nlg:
    def __init__(self):
        pass
    
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        """ post_process to fill the slot in the template sentence """
        
        sentence = pred_template
        suffix = "_PLACEHOLDER"
        
        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = slot + suffix
            if slot == 'result' or slot == 'numberofpeople': continue
            if slot_vals == NO_VALUE_MATCH: continue
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = 'numberofpeople' + suffix
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
    
        for slot in slot_dict.keys():
            slot_placeholder = slot + suffix
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
            
        return sentence

    
    def convert_diaact_to_nl(self, dia_act, turn_msg):
        """ Convert Dia_Act into NL: Rule + Model """
        
        sentence = ""
        boolean_in = False
        
        # remove I do not care slot in task(complete)
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] != NO_VALUE_MATCH:
            inform_slot_set = dia_act['inform_slots'].keys()
            for slot in inform_slot_set:
                if dia_act['inform_slots'][slot] == I_DO_NOT_CARE: del dia_act['inform_slots'][slot]
        
        if dia_act['diaact'] in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][dia_act['diaact']]:
                if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
                    sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
                    boolean_in = True
                    break
        
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == NO_VALUE_MATCH:
            sentence = "Oh sorry, there is no ticket available."
        
        if boolean_in == False: sentence = self.translate_diaact(dia_act)
        return sentence
        
        
    def translate_diaact(self, dia_act):
        """ prepare the diaact into vector representation, and generate the sentence by Model """
        
        word_dict = self.word_dict
        template_word_dict = self.template_word_dict
        act_dict = self.act_dict
        slot_dict = self.slot_dict
        inverse_word_dict = self.inverse_word_dict
    
        act_rep = np.zeros((1, len(act_dict)))
        act_rep[0, act_dict[dia_act['diaact']]] = 1.0
    
        slot_rep_bit = 2
        slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit)) 
    
        suffix = "_PLACEHOLDER"
        if self.params['dia_slot_val'] == 2 or self.params['dia_slot_val'] == 3:
            word_rep = np.zeros((1, len(template_word_dict)))
            words = np.zeros((1, len(template_word_dict)))
            words[0, template_word_dict['s_o_s']] = 1.0
        else:
            word_rep = np.zeros((1, len(word_dict)))
            words = np.zeros((1, len(word_dict)))
            words[0, word_dict['s_o_s']] = 1.0
    
        for slot in dia_act['inform_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit] = 1.0
        
            for slot_val in dia_act['inform_slots'][slot]:
                if self.params['dia_slot_val'] == 2:
                    slot_placeholder = slot + suffix
                    if slot_placeholder in template_word_dict.keys():
                        word_rep[0, template_word_dict[slot_placeholder]] = 1.0
                elif self.params['dia_slot_val'] == 1:
                    if slot_val in word_dict.keys():
                        word_rep[0, word_dict[slot_val]] = 1.0
                    
        for slot in dia_act['request_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0
    
        if self.params['dia_slot_val'] == 0 or self.params['dia_slot_val'] == 3:
            final_representation = np.hstack([act_rep, slot_rep])
        else: # dia_slot_val = 1, 2
            final_representation = np.hstack([act_rep, slot_rep, word_rep])
    
        dia_act_rep = {}
        dia_act_rep['diaact'] = final_representation
        dia_act_rep['words'] = words
    
        #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
        pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
        pred_sentence = ' '.join(pred_words[:-1])
        sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
            
        return sentence
    
    
    def load_nlg_model(self, model_path):
        """ load the trained NLG model """  
        
        model_params = pickle.load(open(model_path, 'rb'))
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
            diaact_input_size = model_params['model']['Wah'].shape[0]
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)
        
        rnnmodel.model = copy.deepcopy(model_params['model'])
        model_params['params']['beam_size'] = nlg_beam_size
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.template_word_dict = copy.deepcopy(model_params['template_word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.inverse_word_dict = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
        self.params = copy.deepcopy(model_params['params'])
        
    
    def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
        """ Replace the slots with its values """
        
        sentence = template_sentence
        counter = 0
        for slot in dia_act['inform_slots'].keys():
            slot_val = dia_act['inform_slots'][slot]
            if slot_val == NO_VALUE_MATCH:
                sentence = slot + " is not available!"
                break
            elif slot_val == I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace('$'+slot+'$', '', 1)
                continue
            
            sentence = sentence.replace('$'+slot+'$', slot_val, 1)
        
        if counter > 0 and counter == len(dia_act['inform_slots']):
            sentence = I_DO_NOT_CARE
        
        return sentence
    
    
    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """
        
        self.diaact_nl_pairs = json.load(open(path, 'rb'))
        
        for key in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][key]:
                ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
                ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue


# In[24]:


class SeqToSeq:
    def __init__(self, input_size, hidden_size, output_size):
        pass
    
    def get_struct(self):
        return {'model': self.model, 'update': self.update, 'regularize': self.regularize}
    
    
    """ Forward Function"""
    def fwdPass(self, Xs, params, **kwargs):
        pass
    
    def bwdPass(self, dY, cache):
        pass
    
    
    """ Batch Forward & Backward Pass"""
    def batchForward(self, ds, batch, params, predict_mode = False):
        caches = []
        Ys = []
        for i,x in enumerate(batch):
            Y, out_cache = self.fwdPass(x, params, predict_mode = predict_mode)
            caches.append(out_cache)
            Ys.append(Y)
           
        # back up information for efficient backprop
        cache = {}
        if not predict_mode:
            cache['caches'] = caches

        return Ys, cache
    
    def batchBackward(self, dY, cache):
        caches = cache['caches']
        grads = {}
        for i in xrange(len(caches)):
            single_cache = caches[i]
            local_grads = self.bwdPass(dY[i], single_cache)
            mergeDicts(grads, local_grads) # add up the gradients wrt model parameters
            
        return grads


    """ Cost function, returns cost and gradients for model """
    def costFunc(self, ds, batch, params):
        regc = params['reg_cost'] # regularization cost
        
        # batch forward RNN
        Ys, caches = self.batchForward(ds, batch, params, predict_mode = False)
        
        loss_cost = 0.0
        smooth_cost = 1e-15
        dYs = []
        
        for i,x in enumerate(batch):
            labels = np.array(x['tags_rep'], dtype=int)
            
            # fetch the predicted probabilities
            Y = Ys[i]
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            P = e/np.sum(e, axis=1, keepdims=True)
            
            # Cross-Entropy Cross Function
            loss_cost += -np.sum(np.log(smooth_cost + P[range(len(labels)), labels]))
            
            for iy,y in enumerate(labels):
                P[iy,y] -= 1 # softmax derivatives
            dYs.append(P)
            
        # backprop the RNN
        grads = self.batchBackward(dYs, caches)
        
        # add L2 regularization cost and gradients
        reg_cost = 0.0
        if regc > 0:    
            for p in self.regularize:
                mat = self.model[p]
                reg_cost += 0.5*regc*np.sum(mat*mat)
                grads[p] += regc*mat

        # normalize the cost and gradient by the batch size
        batch_size = len(batch)
        reg_cost /= batch_size
        loss_cost /= batch_size
        for k in grads: grads[k] /= batch_size

        out = {}
        out['cost'] = {'reg_cost' : reg_cost, 'loss_cost' : loss_cost, 'total_cost' : loss_cost + reg_cost}
        out['grads'] = grads
        return out


    """ A single batch """
    def singleBatch(self, ds, batch, params):
        learning_rate = params.get('learning_rate', 0.0)
        decay_rate = params.get('decay_rate', 0.999)
        momentum = params.get('momentum', 0)
        grad_clip = params.get('grad_clip', 1)
        smooth_eps = params.get('smooth_eps', 1e-8)
        sdg_type = params.get('sdgtype', 'rmsprop')

        for u in self.update:
            if not u in self.step_cache: 
                self.step_cache[u] = np.zeros(self.model[u].shape)
        
        cg = self.costFunc(ds, batch, params)
        
        cost = cg['cost']
        grads = cg['grads']
        
        # clip gradients if needed
        if params['activation_func'] == 'relu':
            if grad_clip > 0:
                for p in self.update:
                    if p in grads:
                        grads[p] = np.minimum(grads[p], grad_clip)
                        grads[p] = np.maximum(grads[p], -grad_clip)
        
        # perform parameter update
        for p in self.update:
            if p in grads:
                if sdg_type == 'vanilla':
                    if momentum > 0: dx = momentum*self.step_cache[p] - learning_rate*grads[p]
                    else: dx = -learning_rate*grads[p]
                    self.step_cache[p] = dx
                elif sdg_type == 'rmsprop':
                    self.step_cache[p] = self.step_cache[p]*decay_rate + (1.0-decay_rate)*grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                elif sdg_type == 'adgrad':
                    self.step_cache[p] += grads[p]**2
                    dx = -(learning_rate*grads[p])/np.sqrt(self.step_cache[p] + smooth_eps)
                    
                self.model[p] += dx

        # create output dict and return
        out = {}
        out['cost'] = cost
        return out
    
    
    """ Evaluate on the dataset[split] """
    def eval(self, ds, split, params):
        acc = 0
        total = 0
        
        total_cost = 0.0
        smooth_cost = 1e-15
        
        if split == 'test':
            res_filename = 'res_%s_[%s].txt' % (params['model'], time.time())
            res_filepath = os.path.join(params['test_res_dir'], res_filename)
            res = open(res_filepath, 'w')
            inverse_tag_dict = {ds.data['tag_set'][k]:k for k in ds.data['tag_set'].keys()}
            
        for i, ele in enumerate(ds.split[split]):
            Ys, cache = self.fwdPass(ele, params, predict_model=True)
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            labels = np.array(ele['tags_rep'], dtype=int)
            
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            loss_cost = 0
            loss_cost += -np.sum(np.log(smooth_cost + probs[range(len(labels)), labels]))
            total_cost += loss_cost
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            
            tokens = ele['raw_seq']
            real_tags = ele['tag_seq']
            for index, l in enumerate(labels):
                if pred_words_indices[index] == l: acc += 1
                
                if split == 'test':
                    res.write('%s %s %s %s\n' % (tokens[index], 'NA', real_tags[index], inverse_tag_dict[pred_words_indices[index]]))
            if split == 'test': res.write('\n')
            total += len(labels)
            
        total_cost /= len(ds.split[split])
        accuracy = 0 if total == 0 else float(acc)/total
        
        #print ("total_cost: %s, accuracy: %s" % (total_cost, accuracy))
        result = {'cost': total_cost, 'accuracy': accuracy}
        return result


# In[25]:


class lstm(SeqToSeq):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = {}
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(input_size + hidden_size + 1, 4*hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(hidden_size, output_size)*0.1
        self.model['bd'] = np.zeros((1, output_size))

        self.update = ['WLSTM', 'Wd', 'bd']
        self.regularize = ['WLSTM', 'Wd']

        self.step_cache = {}
        
    """ Activation Function: Sigmoid, or tanh, or ReLu """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        
        Ws = Xs['word_vectors']
        
        WLSTM = self.model['WLSTM']
        n, xd = Ws.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
    
        for t in xrange(n):
            prev = np.zeros(d) if t==0 else Hout[t-1]
            Hin[t,0] = 1 # bias
            Hin[t, 1:1+xd] = Ws[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            IFOG[t] = Hin[t].dot(WLSTM)
            
            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d])) # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh for input value
            
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
            
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Ws'] = Ws
            cache['Hin'] = Hin
            
        return Y, cache
    
    """ Backward Pass """
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        Ws = cache['Ws']
        
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        
        for t in reversed(xrange(n)):
            dIFOGf[t,2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t,2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t>0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t,d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t,3*d:] * dCellin[t]
            dIFOGf[t,3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0: dHout[t-1] += dHin[t, 1+Ws.shape[1]:]
        
        #dXs = dXsh.dot(Wxh.transpose())  
        return {'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd}


# In[26]:


class nlu:
    def __init__(self):
        pass
    
    def generate_dia_act(self, annot):
        """ generate the Dia-Act with NLU model """
        
        if len(annot) > 0:
            tmp_annot = annot.strip('.').strip('?').strip(',').strip('!') 
            
            rep = self.parse_str_to_vector(tmp_annot)
            Ys, cache = self.model.fwdPass(rep, self.params, predict_model=True) # default: True
            
            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)
            
            # special handling with intent label
            for tag_id in self.inverse_tag_dict.keys():
                if self.inverse_tag_dict[tag_id].startswith('B-') or self.inverse_tag_dict[tag_id].startswith('I-') or self.inverse_tag_dict[tag_id] == 'O':
                    probs[-1][tag_id] = 0
            
            pred_words_indices = np.nanargmax(probs, axis=1)
            pred_tags = [self.inverse_tag_dict[index] for index in pred_words_indices]
            
            diaact = self.parse_nlu_to_diaact(pred_tags, tmp_annot)
            return diaact
        else:
            return None

    
    def load_nlu_model(self, model_path):
        """ load the trained NLU model """  
        
        model_params = pickle.load(open(model_path, 'rb'))
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm': # lstm_
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm(input_size, hidden_size, output_size)
        elif model_params['params']['model'] == 'bi_lstm': # bi_lstm
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = biLSTM(input_size, hidden_size, output_size)
           
        rnnmodel.model = copy.deepcopy(model_params['model'])
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.tag_set = copy.deepcopy(model_params['tag_set'])
        self.params = copy.deepcopy(model_params['params'])
        self.inverse_tag_dict = {self.tag_set[k]:k for k in self.tag_set.keys()}
        
           
    def parse_str_to_vector(self, string):
        """ Parse string into vector representations """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')
        
        vecs = np.zeros((len(words), len(self.word_dict)))
        for w_index, w in enumerate(words):
            if w.endswith(',') or w.endswith('?'): w = w[0:-1]
            if w in self.word_dict.keys():
                vecs[w_index][self.word_dict[w]] = 1
            else: vecs[w_index][self.word_dict['unk']] = 1
        
        rep = {}
        rep['word_vectors'] = vecs
        rep['raw_seq'] = string
        return rep

    def parse_nlu_to_diaact(self, nlu_vector, string):
        """ Parse BIO and Intent into Dia-Act """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')
    
        diaact = {}
        diaact['diaact'] = "inform"
        diaact['request_slots'] = {}
        diaact['inform_slots'] = {}
        
        intent = nlu_vector[-1]
        index = 1
        pre_tag = nlu_vector[0]
        pre_tag_index = 0
    
        slot_val_dict = {}
    
        while index<(len(nlu_vector)-1): # except last Intent tag
            cur_tag = nlu_vector[index]
            if cur_tag == 'O' and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
                if cur_tag.split('-')[1] != pre_tag.split('-')[1]:           
                    slot = pre_tag.split('-')[1]
                    slot_val_str = ' '.join(words[pre_tag_index:index])
                    slot_val_dict[slot] = slot_val_str
            elif cur_tag == 'O' and pre_tag.startswith('I-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
               
            if cur_tag.startswith('B-'): pre_tag_index = index
        
            pre_tag = cur_tag
            index += 1
    
        if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
            slot = cur_tag.split('-')[1]
            slot_val_str = ' '.join(words[pre_tag_index:-1])
            slot_val_dict[slot] = slot_val_str
    
        if intent != 'null':
            arr = intent.split('+')
            diaact['diaact'] = arr[0]
            diaact['request_slots'] = {}
            for ele in arr[1:]: 
                #request_slots.append(ele)
                diaact['request_slots'][ele] = 'UNK'
        
        diaact['inform_slots'] = slot_val_dict
         
        # add rule here
        for slot in diaact['inform_slots'].keys():
            slot_val = diaact['inform_slots'][slot]
            if slot_val.startswith('bos'): 
                slot_val = slot_val.replace('bos', '', 1)
                diaact['inform_slots'][slot] = slot_val.strip(' ')
        
        self.refine_diaact_by_rules(diaact)
        return diaact

    def refine_diaact_by_rules(self, diaact):
        """ refine the dia_act by rules """
        
        # rule for taskcomplete
        if 'request_slots' in diaact.keys():
            if 'taskcomplete' in diaact['request_slots'].keys():
                del diaact['request_slots']['taskcomplete']
                diaact['inform_slots']['taskcomplete'] = 'PLACEHOLDER'
        
            # rule for request
            if len(diaact['request_slots'])>0: diaact['diaact'] = 'request'
    
    
    
    
    def diaact_penny_string(self, dia_act):
        """ Convert the Dia-Act into penny string """
        
        penny_str = ""
        penny_str = dia_act['diaact'] + "("
        for slot in dia_act['request_slots'].keys():
            penny_str += slot + ";"
    
        for slot in dia_act['inform_slots'].keys():
            slot_val_str = slot + "="
            if len(dia_act['inform_slots'][slot]) == 1:
                slot_val_str += dia_act['inform_slots'][slot][0]
            else:
                slot_val_str += "{"
                for slot_val in dia_act['inform_slots'][slot]:
                    slot_val_str += slot_val + "#"
                slot_val_str = slot_val_str[:-1]
                slot_val_str += "}"
            penny_str += slot_val_str + ";"
    
        if penny_str[-1] == ";": penny_str = penny_str[:-1]
        penny_str += ")"
        return penny_str


# In[27]:


nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)


# In[28]:


nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)


# In[29]:


class KBHelper:
    """ An assistant to fill in values for the agent (which knows about slots of values) """
    
    def __init__(self, movie_dictionary):
        """ Constructor for a KBHelper """
        
        self.movie_dictionary = movie_dictionary
        self.cached_kb = defaultdict(list)
        self.cached_kb_slot = defaultdict(list)


    def fill_inform_slots(self, inform_slots_to_be_filled, current_slots):
        """ Takes unfilled inform slots and current_slots, returns dictionary of filled informed slots (with values)
        Arguments:
        inform_slots_to_be_filled   --  Something that looks like {starttime:None, theater:None} where starttime and theater are slots that the agent needs filled
        current_slots               --  Contains a record of all filled slots in the conversation so far - for now, just use current_slots['inform_slots'] which is a dictionary of the already filled-in slots
        Returns:
        filled_in_slots             --  A dictionary of form {slot1:value1, slot2:value2} for each sloti in inform_slots_to_be_filled
        """
        
        kb_results = self.available_results_from_kb(current_slots)
        if auto_suggest == 1:
            print 'Number of movies in KB satisfying current constraints: ', len(kb_results)

        filled_in_slots = {}
        if 'taskcomplete' in inform_slots_to_be_filled.keys():
            filled_in_slots.update(current_slots['inform_slots'])
        
        for slot in inform_slots_to_be_filled.keys():
            if slot == 'numberofpeople':
                if slot in current_slots['inform_slots'].keys():
                    filled_in_slots[slot] = current_slots['inform_slots'][slot]
                elif slot in inform_slots_to_be_filled.keys():
                    filled_in_slots[slot] = inform_slots_to_be_filled[slot]
                continue

            if slot == 'ticket' or slot == 'taskcomplete':
                filled_in_slots[slot] = TICKET_AVAILABLE if len(kb_results)>0 else NO_VALUE_MATCH
                continue
            
            if slot == 'closing': continue

            ####################################################################
            #   Grab the value for the slot with the highest count and fill it
            ####################################################################
            values_dict = self.available_slot_values(slot, kb_results)

            values_counts = [(v, values_dict[v]) for v in values_dict.keys()]
            if len(values_counts) > 0:
                filled_in_slots[slot] = sorted(values_counts, key = lambda x: -x[1])[0][0]
            else:
                filled_in_slots[slot] = NO_VALUE_MATCH #"NO VALUE MATCHES SNAFU!!!"
           
        return filled_in_slots


    def available_slot_values(self, slot, kb_results):
        """ Return the set of values available for the slot based on the current constraints """
        
        slot_values = {}
        for movie_id in kb_results.keys():
            if slot in kb_results[movie_id].keys():
                slot_val = kb_results[movie_id][slot]
                if slot_val in slot_values.keys():
                    slot_values[slot_val] += 1
                else: slot_values[slot_val] = 1
        return slot_values

    def available_results_from_kb(self, current_slots):
        """ Return the available movies in the movie_kb based on the current constraints """
        
        ret_result = []
        current_slots = current_slots['inform_slots']
        constrain_keys = current_slots.keys()

        constrain_keys = filter(lambda k : k != 'ticket' and                                            k != 'numberofpeople' and                                            k!= 'taskcomplete' and                                            k != 'closing' , constrain_keys)
        constrain_keys = [k for k in constrain_keys if current_slots[k] != I_DO_NOT_CARE]

        query_idx_keys = frozenset(current_slots.items())
        cached_kb_ret = self.cached_kb[query_idx_keys]

        cached_kb_length = len(cached_kb_ret) if cached_kb_ret != None else -1
        if cached_kb_length > 0:
            return dict(cached_kb_ret)
        elif cached_kb_length == -1:
            return dict([])

        # kb_results = copy.deepcopy(self.movie_dictionary)
        for id in self.movie_dictionary.keys():
            kb_keys = self.movie_dictionary[id].keys()
            if len(set(constrain_keys).union(set(kb_keys)) ^ (set(constrain_keys) ^ set(kb_keys))) == len(
                    constrain_keys):
                match = True
                for idx, k in enumerate(constrain_keys):
                    if str(current_slots[k]).lower() == str(self.movie_dictionary[id][k]).lower():
                        continue
                    else:
                        match = False
                if match:
                    self.cached_kb[query_idx_keys].append((id, self.movie_dictionary[id]))
                    ret_result.append((id, self.movie_dictionary[id]))

            # for slot in current_slots['inform_slots'].keys():
            #     if slot == 'ticket' or slot == 'numberofpeople' or slot == 'taskcomplete' or slot == 'closing': continue
            #     if current_slots['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: continue
            #
            #     if slot not in self.movie_dictionary[movie_id].keys():
            #         if movie_id in kb_results.keys():
            #             del kb_results[movie_id]
            #     else:
            #         if current_slots['inform_slots'][slot].lower() != self.movie_dictionary[movie_id][slot].lower():
            #             if movie_id in kb_results.keys():
            #                 del kb_results[movie_id]
            
        if len(ret_result) == 0:
            self.cached_kb[query_idx_keys] = None

        ret_result = dict(ret_result)
        return ret_result
    
    def available_results_from_kb_for_slots(self, inform_slots):
        """ Return the count statistics for each constraint in inform_slots """
        
        kb_results = {key:0 for key in inform_slots.keys()}
        kb_results['matching_all_constraints'] = 0
        
        query_idx_keys = frozenset(inform_slots.items())
        cached_kb_slot_ret = self.cached_kb_slot[query_idx_keys]

        if len(cached_kb_slot_ret) > 0:
            return cached_kb_slot_ret[0]

        for movie_id in self.movie_dictionary.keys():
            all_slots_match = 1
            for slot in inform_slots.keys():
                if slot == 'ticket' or inform_slots[slot] == I_DO_NOT_CARE:
                    continue

                if slot in self.movie_dictionary[movie_id].keys():
                    if inform_slots[slot].lower() == self.movie_dictionary[movie_id][slot].lower():
                        kb_results[slot] += 1
                    else:
                        all_slots_match = 0
                else:
                    all_slots_match = 0
            kb_results['matching_all_constraints'] += all_slots_match

        self.cached_kb_slot[query_idx_keys].append(kb_results)
        return kb_results

    
    def database_results_for_agent(self, current_slots):
        """ A dictionary of the number of results matching each current constraint. The agent needs this to decide what to do next. """

        database_results ={} # { date:100, distanceconstraints:60, theater:30,  matching_all_constraints: 5}
        database_results = self.available_results_from_kb_for_slots(current_slots['inform_slots'])
        return database_results
    
    def suggest_slot_values(self, request_slots, current_slots):
        """ Return the suggest slot values """
        
        avail_kb_results = self.available_results_from_kb(current_slots)
        return_suggest_slot_vals = {}
        for slot in request_slots.keys():
            avail_values_dict = self.available_slot_values(slot, avail_kb_results)
            values_counts = [(v, avail_values_dict[v]) for v in avail_values_dict.keys()]
            
            if len(values_counts) > 0:
                return_suggest_slot_vals[slot] = []
                sorted_dict = sorted(values_counts, key = lambda x: -x[1])
                for k in sorted_dict: return_suggest_slot_vals[slot].append(k[0])
            else:
                return_suggest_slot_vals[slot] = []
        
        return return_suggest_slot_vals


# In[30]:


class StateTracker:
    """ The state tracker maintains a record of which request slots are filled and which inform slots are filled """

    def __init__(self, act_set, slot_set, movie_dictionary):
        """ constructor for statetracker takes movie knowledge base and initializes a new episode
        Arguments:
        act_set                 --  The set of all acts availavle
        slot_set                --  The total set of available slots
        movie_dictionary        --  A representation of all the available movies. Generally this object is accessed via the KBHelper class
        Class Variables:
        history_vectors         --  A record of the current dialog so far in vector format (act-slot, but no values)
        history_dictionaries    --  A record of the current dialog in dictionary format
        current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots'] and which are requested current_slots['request_slots'] (but not filed)
        action_dimension        --  # TODO indicates the dimensionality of the vector representaiton of the action
        kb_result_dimension     --  A single integer denoting the dimension of the kb_results features.
        turn_count              --  A running count of which turn we are at in the present dialog
        """
        self.movie_dictionary = movie_dictionary
        self.initialize_episode()
        self.history_vectors = None
        self.history_dictionaries = None
        self.current_slots = None
        self.action_dimension = 10      # TODO REPLACE WITH REAL VALUE
        self.kb_result_dimension = 10   # TODO  REPLACE WITH REAL VALUE
        self.turn_count = 0
        self.kb_helper = KBHelper(movie_dictionary)
        

    def initialize_episode(self):
        """ Initialize a new episode (dialog), flush the current state and tracked slots """
        
        self.action_dimension = 10
        self.history_vectors = np.zeros((1, self.action_dimension))
        self.history_dictionaries = []
        self.turn_count = 0
        self.current_slots = {}
        
        self.current_slots['inform_slots'] = {}
        self.current_slots['request_slots'] = {}
        self.current_slots['proposed_slots'] = {}
        self.current_slots['agent_request_slots'] = {}


    def dialog_history_vectors(self):
        """ Return the dialog history (both user and agent actions) in vector representation """
        return self.history_vectors


    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.history_dictionaries


    def kb_results_for_state(self):
        """ Return the information about the database results based on the currently informed slots """
        ########################################################################
        # TODO Calculate results based on current informed slots
        ########################################################################
        kb_results = self.kb_helper.database_results_for_agent(self.current_slots) # replace this with something less ridiculous
        # TODO turn results into vector (from dictionary)
        results = np.zeros((0, self.kb_result_dimension))
        return results
        

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """
        #state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, 'kb_results': self.kb_results_for_state()}
        state = {'user_action': self.history_dictionaries[-1], 'current_slots': self.current_slots, #'kb_results': self.kb_results_for_state(), 
                 'kb_results_dict':self.kb_helper.database_results_for_agent(self.current_slots), 'turn': self.turn_count, 'history': self.history_dictionaries, 
                 'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None}
        return copy.deepcopy(state)
    
    def get_suggest_slots_values(self, request_slots):
        """ Get the suggested values for request slots """
        
        suggest_slot_vals = {}
        if len(request_slots) > 0: 
            suggest_slot_vals = self.kb_helper.suggest_slot_values(request_slots, self.current_slots)
        
        return suggest_slot_vals
    
    def get_current_kb_results(self):
        """ get the kb_results for current state """
        kb_results = self.kb_helper.available_results_from_kb(self.current_slots)
        return kb_results
    
    
    def update(self, agent_action=None, user_action=None):
        """ Update the state based on the latest action """

        ########################################################################
        #  Make sure that the function was called properly
        ########################################################################
        assert(not (user_action and agent_action))
        assert(user_action or agent_action)

        ########################################################################
        #   Update state to reflect a new action by the agent.
        ########################################################################
        if agent_action:
            
            ####################################################################
            #   Handles the act_slot response (with values needing to be filled)
            ####################################################################
            if agent_action['act_slot_response']:
                response = copy.deepcopy(agent_action['act_slot_response'])
                
                inform_slots = self.kb_helper.fill_inform_slots(response['inform_slots'], self.current_slots) # TODO this doesn't actually work yet, remove this warning when kb_helper is functional
                agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots']}
                
                agent_action['act_slot_response'].update({'diaact': response['diaact'], 'inform_slots': inform_slots, 'request_slots':response['request_slots'], 'turn':self.turn_count})
                
            elif agent_action['act_slot_value_response']:
                agent_action_values = copy.deepcopy(agent_action['act_slot_value_response'])
                # print("Updating state based on act_slot_value action from agent")
                agent_action_values['turn'] = self.turn_count
                agent_action_values['speaker'] = "agent"
                
            ####################################################################
            #   This code should execute regardless of which kind of agent produced action
            ####################################################################
            for slot in agent_action_values['inform_slots'].keys():
                self.current_slots['proposed_slots'][slot] = agent_action_values['inform_slots'][slot]
                self.current_slots['inform_slots'][slot] = agent_action_values['inform_slots'][slot] # add into inform_slots
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in agent_action_values['request_slots'].keys():
                if slot not in self.current_slots['agent_request_slots']:
                    self.current_slots['agent_request_slots'][slot] = "UNK"

            self.history_dictionaries.append(agent_action_values)
            current_agent_vector = np.ones((1, self.action_dimension))
            self.history_vectors = np.vstack([self.history_vectors, current_agent_vector])
                            
        ########################################################################
        #   Update the state to reflect a new action by the user
        ########################################################################
        elif user_action:
            
            ####################################################################
            #   Update the current slots
            ####################################################################
            for slot in user_action['inform_slots'].keys():
                self.current_slots['inform_slots'][slot] = user_action['inform_slots'][slot]
                if slot in self.current_slots['request_slots'].keys():
                    del self.current_slots['request_slots'][slot]

            for slot in user_action['request_slots'].keys():
                if slot not in self.current_slots['request_slots']:
                    self.current_slots['request_slots'][slot] = "UNK"
            
            self.history_vectors = np.vstack([self.history_vectors, np.zeros((1,self.action_dimension))])
            new_move = {'turn': self.turn_count, 'speaker': "user", 'request_slots': user_action['request_slots'], 'inform_slots': user_action['inform_slots'], 'diaact': user_action['diaact']}
            self.history_dictionaries.append(copy.deepcopy(new_move))

        ########################################################################
        #   This should never happen if the asserts passed
        ########################################################################
        else:
            pass

        ########################################################################
        #   This code should execute after update code regardless of what kind of action (agent/user)
        ########################################################################
        self.turn_count += 1


# In[31]:


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """
    
    def __init__(self, agent, user, act_set, slot_set, movie_dictionary):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set
        self.state_tracker = StateTracker(act_set, slot_set, movie_dictionary)
        self.user_action = None
        self.reward = 0
        self.episode_over = False

    def initialize_episode(self):
        """ Refresh state for new dialog """
        
        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action = self.user.initialize_episode()
        self.state_tracker.update(user_action = self.user_action)
        
        if run_mode < 3:
            print ("New episode, user goal:")
            print json.dumps(self.user.goal, indent=2)
        self.print_function(user_action = self.user_action)
            
        self.agent.initialize_episode()

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """
        
        ########################################################################
        #   CALL AGENT TO TAKE HER TURN
        ########################################################################
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action = self.agent.state_to_action(self.state)
        
        ########################################################################
        #   Register AGENT action with the state_tracker
        ########################################################################
        self.state_tracker.update(agent_action=self.agent_action)
        
        self.agent.add_nl_to_action(self.agent_action) # add NL to Agent Dia_Act
        self.print_function(agent_action = self.agent_action['act_slot_response'])
        
        ########################################################################
        #   CALL USER TO TAKE HER TURN
        ########################################################################
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, dialog_status = self.user.next(self.sys_action)
        self.reward = self.reward_function(dialog_status)
        
        ########################################################################
        #   Update state tracker with latest user action
        ########################################################################
        if self.episode_over != True:
            self.state_tracker.update(user_action = self.user_action)
            self.print_function(user_action = self.user_action)

        ########################################################################
        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        ########################################################################
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward, self.state_tracker.get_state_for_agent(), self.episode_over)
        
        return (self.episode_over, self.reward)

    
    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == FAILED_DIALOG:
            reward = -self.user.max_turn #10
        elif dialog_status == SUCCESS_DIALOG:
            reward = 2*self.user.max_turn #20
        else:
            reward = -1
        return reward
    
    def reward_function_without_penalty(self, dialog_status):
        """ Reward Function 2: a reward function without penalty on per turn and failure dialog """
        if dialog_status == dialog_config.FAILED_DIALOG:
            reward = 0
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = 2*self.user.max_turn
        else:
            reward = 0
        return reward
    
    
    def print_function(self, agent_action=None, user_action=None):
        """ Print Function """
            
        if agent_action:
            if run_mode == 0:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            elif run_mode == 1:
                if self.agent.__class__.__name__ != 'AgentCmd':
                    print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
            elif run_mode == 2: # debug mode
                print("Turn %d sys: %s, inform_slots: %s, request slots: %s" % (agent_action['turn'], agent_action['diaact'], agent_action['inform_slots'], agent_action['request_slots']))
                print ("Turn %d sys: %s" % (agent_action['turn'], agent_action['nl']))
            
            if auto_suggest == 1:
                print('(Suggested Values: %s)' % (self.state_tracker.get_suggest_slots_values(agent_action['request_slots'])))
        elif user_action:
            if run_mode == 0:
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            elif run_mode == 1: 
                print ("Turn %s usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
            elif run_mode == 2: # debug mode, show both
                print ("Turn %d usr: %s, inform_slots: %s, request_slots: %s" % (user_action['turn'], user_action['diaact'], user_action['inform_slots'], user_action['request_slots']))
                print ("Turn %d usr: %s" % (user_action['turn'], user_action['nl']))
            
            if self.agent.__class__.__name__ == 'AgentCmd': # command line agent
                user_request_slots = user_action['request_slots']
                if 'ticket'in user_request_slots.keys(): del user_request_slots['ticket']
                if len(user_request_slots) > 0:
                    possible_values = self.state_tracker.get_suggest_slots_values(user_action['request_slots'])
                    for slot in possible_values.keys():
                        if len(possible_values[slot]) > 0:
                            print('(Suggested Values: %s: %s)' % (slot, possible_values[slot]))
                        elif len(possible_values[slot]) == 0:
                            print('(Suggested Values: there is no available %s)' % (slot))
                else:
                    kb_results = self.state_tracker.get_current_kb_results()
                    print ('(Number of movies in KB satisfying current constraints: %s)' % len(kb_results))


# In[32]:


dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)


# In[33]:


status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size'] # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']


# In[34]:


""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}


# In[35]:


""" Save model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if agt == 9: checkpoint['model'] = copy.deepcopy(agent.dqn.model)
    checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print 'saved model in %s' % (filepath, )
    except Exception, e:
        print 'Error: Writing model fails: %s' % (filepath, )
        print e

""" save performance numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "wb"))
        print 'saved model in %s' % (filepath, )
    except Exception, e:
        print 'Error: Writing model fails: %s' % (filepath, )
        print e

""" Run N simulation Dialogues """
def simulation_epoch(simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    res = {}
    for episode in xrange(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0: 
                    successes += 1
                    print ("simulation episode %s: Success" % (episode))
                else: print ("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
    
    res['success_rate'] = float(successes)/simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward)/simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns)/simulation_epoch_size
    print ("simulation success rate %s, ave reward %s, ave turns %s" % (res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res

""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    res = {}
    warm_start_run_epochs = 0
    for episode in xrange(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0: 
                    successes += 1
                    print ("warm_start simulation episode %s: Success" % (episode))
                else: print ("warm_start simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
        
        warm_start_run_epochs += 1
        
        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break
        
    agent.warm_start = 2
    res['success_rate'] = float(successes)/warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward)/warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns)/warm_start_run_epochs
    print ("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode+1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print ("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))



def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    if agt == 9 and params['trained_model_path'] == None and warm_start == 1:
        print ('warm_start starting ...')
        warm_start_simulation()
        print ('warm_start finished, start RL training ...')
    
    for episode in xrange(count):
        print ("Episode: %s" % (episode))
        dialog_manager.initialize_episode()
        episode_over = False
        
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
                
            if episode_over:
                if reward > 0:
                    print ("Successful Dialog!")
                    successes += 1
                else: print ("Failed Dialog!")
                
                cumulative_turns += dialog_manager.state_tracker.turn_count
        
        # simulation
        if agt == 9 and params['trained_model_path'] == None:
            agent.predict_mode = True
            simulation_res = simulation_epoch(simulation_epoch_size)
            
            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']
            
            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold: # threshold = 0.30
                    agent.experience_replay_pool = [] 
                    simulation_epoch(simulation_epoch_size)
                
            if simulation_res['success_rate'] > best_res['success_rate']:
                best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode
                
            agent.clone_dqn = copy.deepcopy(agent.dqn)
            agent.train(batch_size, 1)
            agent.predict_mode = False
            
            print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (performance_records['success_rate'][episode], performance_records['ave_reward'][episode], performance_records['ave_turns'][episode], best_res['success_rate']))
            if episode % save_check_point == 0 and params['trained_model_path'] == None: # save the model every 10 episodes
                save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records)
        
        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode+1, count, successes, episode+1, float(cumulative_reward)/(episode+1), float(cumulative_turns)/(episode+1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes, count, float(cumulative_reward)/count, float(cumulative_turns)/count))
    status['successes'] += successes
    status['count'] += count
    
    if agt == 9 and params['trained_model_path'] == None:
        save_model(params['write_model_dir'], agt, float(successes)/count, best_model['model'], best_res['epoch'], count)
        save_performance_records(params['write_model_dir'], agt, performance_records)


# In[36]:


run_episodes(num_episodes, status)

