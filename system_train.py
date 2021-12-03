import csv
import numpy as np

# Global Variables
state_names = ['GENRE_FILLED', 'RATING_FILLED', 'LANGUAGE_FILLED', 'GENRE_CONF', 'RATING_CONF',
'LANGUAGE_CONF', 'BEST_ACTION']

action_indices = {0:'REQUEST_GENRE', 1:'REQUEST_RATING',
2:'REQUEST_LANGUAGE', 3:'EXPLICIT_CONFIRM_GENRE', 4:'EXPLICIT_CONFIRM_RATING',
5:'EXPLICIT_CONFIRM_LANGUAGE'}

action_indices_invert = {'REQUEST_GENRE':0, 'REQUEST_RATING':1,
'REQUEST_LANGUAGE':2, 'EXPLICIT_CONFIRM_GENRE':3, 'EXPLICIT_CONFIRM_RATING':4,
'EXPLICIT_CONFIRM_LANGUAGE':5}

map_user_prob = {'REQUEST_GENRE': ['PROVIDE_GENRE', 'IRRELEVANT'],
                 'REQUEST_RATING' : ['PROVIDE_RATING','IRRELEVANT'], 'REQUEST_LANGUAGE': ['PROVIDE_LANGUAGE','IRRELEVANT'],
                 'EXPLICIT_CONFIRM_GENRE': ['CONFIRM_POS_GENRE','CONFIRM_NEG_GENRE','IRRELEVANT'],
                 'EXPLICIT_CONFIRM_RATING': ['CONFIRM_POS_RATING','CONFIRM_NEG_RATING','IRRELEVANT'], 'EXPLICIT_CONFIRM_LANGUAGE': ['CONFIRM_POS_LANGUAGE', 'CONFIRM_NEG_LANGUAGE', 'IRRELEVANT']}

mapping_user_to_state = {'PROVIDE_GENRE': 0, 'PROVIDE_RATING': 1, 'PROVIDE_LANGUAGE': 2, 'CONFIRM_POS_GENRE': 3,'CONFIRM_POS_RATING': 4, 'CONFIRM_POS_LANGUAGE': 5}

exploration_rate = 0.30
gamma = 0.99
alpha = 1

def get_states():
    key_length = len(state_names)-1
    key_list_length = 2**key_length

    combinations = []
    for i in range(key_list_length):
        binary = list(bin(i)[2:])
        binary = [int(i) for i in binary]
        if len(binary)<6:
            temp = (6-len(binary))
            temp_ls = [0]*temp
            binary = temp_ls+binary
            binary = tuple(binary)
            combinations.append(binary)
        else:
            binary = tuple(binary)
            combinations.append(binary)
    return combinations

def get_Q(states):
    # terminal state no actions, Q vals?
    
    Q_values = dict()
    for key in states:
        Q_values[key] = [0]*6

    return Q_values

def get_episodes(random_serial, num_episodes):
    episodes = dict()
    for num in range(num_episodes):
        episode = dict()
        for key, action in action_indices.items():
            if random_serial == 1:
                if key == 0 or key ==1 or key == 2:
                    action_list = []
                    rand_val = np.random.choice([0, 1], p=[0.8, 0.2])
                    action_list.append(map_user_prob[action][rand_val])
                    counter =0 
                    while rand_val != 0:
                        rand_val = np.random.choice([0, 1], p=[0.8, 0.2])
                        action_list.append(map_user_prob[action][rand_val])
                        counter+=1

                        # handle length
                        if counter>40:
                            len_rand = np.random.choice([0, 1], p=[0.85, 0.15])
                            if len_rand == 0:
                                action_list[-1] = map_user_prob[action][0]
                            else:
                                continue
                else:
                    action_list = []
                    rand_val = np.random.choice([0, 1,2], p=[0.4, 0.4, 0.2])
                    action_list.append(map_user_prob[action][rand_val])
                    counter = 0 
                    while rand_val != 0:
                        rand_val = np.random.choice([0, 1, 2], p= [0.4, 0.4, 0.2])
                        action_list.append(map_user_prob[action][rand_val])
                        counter+=1

                        # handle length
                        if counter>40:
                            len_rand = np.random.choice([0, 1], p=[0.85, 0.15])
                            if len_rand == 0:
                                action_list[-1] = map_user_prob[action][0]
                            else:
                                continue
                            
                episode[action] = action_list
        episodes[num] = episode
    return episodes

## Global states/values
states = get_states()
Q_values = get_Q(states)
episodes = get_episodes(1,4000) # interspersed random ||# good episodes in beginning, irrelevant in next
#rewards_table = create_reward_table()

# create simulated user
def similated_user(episode_key, cur_state, action):
    global epidodes
    user_action_list = episodes[episode_key][action]

    if len(user_action_list) <=1:
        user_action = user_action_list[0]
    else:
        user_action = episodes[episode_key][action].pop()

    if user_action == 'CONFIRM_POS_GENRE' and cur_state[0] == 0:
        user_action = np.random.choice(['CONFIRM_NEG_GENRE','IRRELEVANT'], p=[0.8, 0.2])

    elif user_action == 'CONFIRM_POS_RATING' and cur_state[1] == 0:
        user_action = np.random.choice(['CONFIRM_NEG_RATING','IRRELEVANT'], p=[0.8, 0.2])

    elif user_action == 'CONFIRM_POS_LANGUAGE' and cur_state[2] == 0:
        user_action = np.random.choice(['CONFIRM_NEG_LANGUAGE', 'IRRELEVANT'], p=[0.8, 0.2])

    else:
        user_action = user_action
        
    return user_action

# create rewards # every action -5, final reward \\\\\ we get rewards for actions, so terminal state no reward
    
def update_state(cur_state, user_action):
    if user_action == 'IRRELEVANT':
        new_state = cur_state
    elif user_action == 'CONFIRM_NEG_RATING':
        cur_state[1] = 0
        cur_state[4] = 0
        new_state = cur_state
    elif user_action == 'CONFIRM_NEG_LANGUAGE':
        cur_state[2] = 0
        cur_state[5] = 0
        new_state = cur_state
    elif user_action == 'CONFIRM_NEG_GENRE':
        cur_state[0] = 0
        cur_state[3] = 0
        new_state = cur_state
    else:
        cur_state[mapping_user_to_state[user_action]] = 1
        new_state = cur_state

    return new_state

def action_epsilon(cur_state):
    exploration_flag = np.random.choice([0, 1], p=[(1.0-exploration_rate), exploration_rate])
    if exploration_flag == 0:
        if cur_state != [0,0,0,0,0,0]:
            action = action_indices[np.argmax(Q_values[tuple(cur_state)])]
        else:
            #action = action_indices[np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])]
            action = action_indices[np.random.choice([0, 1, 2, 3, 4, 5], p=[0.167, 0.167, 0.167, 0.167, 0.166, 0.166])]
    else:
        ### WATCH THIS - ALL ACTIONS ARE BEING EXPLORED
        action = action_indices[np.random.choice([0, 1, 2, 3, 4, 5], p=[0.167, 0.167, 0.167, 0.167, 0.166, 0.166])]
    return action

def RL_train(episodes):
    global exploration_rate
    global alpha

    rewards_episode = []
    counter =0 
    for episode_key, episode in episodes.items():
        # initialize state
        cur_state = [0,0,0,0,0,0]
        dialogue_counter = 1
        reward_episode = 0
        while True:
            # choose action, based on best, tuple argmax, beginning if use random on 3 |||| exploration random value
            action = action_epsilon(cur_state)
            #print(action)

            ### sim user, handles confirm request as none, if already filled is no
            user_action = similated_user(episode_key, cur_state, action)
            #print(user_action)
            
            ### update state after simulated user response - if confirm is no, set filled as zero again
            # state update handle irrelevant
            new_state = update_state(cur_state, user_action)
            #print(new_state)

            #if cur_state == [0,1,1,0,1,1] or cur_state == [1,1,0, 1,1,0]:
            #    print(new_state, action, user_action)
            
            # reward table
            # normal reward
            if sum(new_state) ==6:
                reward = 495
            elif cur_state[action_indices_invert[action]]==1:
                reward = -40
                
            else:
                reward = -5
                
            reward_episode+= reward
            ## set counter for break statements, reward negative .... //// 
            if dialogue_counter >=30:
                reward = -150
                reward_episode+= reward
                #update Q vals =
                internal_term = reward + gamma*max(Q_values[tuple(new_state)]) - Q_values[tuple(cur_state)][action_indices_invert[action]]
                Q_values[tuple(cur_state)][action_indices_invert[action]] = Q_values[tuple(cur_state)][action_indices_invert[action]] + alpha*internal_term
                break
            
            # Q values update
            internal_term = reward+ gamma*max(Q_values[tuple(new_state)]) - Q_values[tuple(cur_state)][action_indices_invert[action]]
            Q_values[tuple(cur_state)][action_indices_invert[action]] = Q_values[tuple(cur_state)][action_indices_invert[action]] + alpha*internal_term

            cur_state = new_state
            
            ## update alpha and
            alpha = 1/(1+counter)

            # exploration rate
            if exploration_rate >=0.3:
                if counter % 30 == 0:
                    exploration_rate -=0.03

            #or all states == 1 break
            if sum(cur_state) == 6:
                break

            
            dialogue_counter +=1
        counter+=1
        rewards_episode.append(reward_episode)

    return rewards_episode
                

def get_policy():
    policy_dic = dict()
    for key,value in Q_values.items():
        ## Totally WATCH THIS
        if key == (1,1,1,1,1,1):
            policy_dic[key] = 'NULL'
        elif key[0]==0 and key[3] ==1 :
            policy_dic[key] = 'NULL'
        elif key[1]==0 and key[4] ==1 :
            policy_dic[key] = 'NULL'
        elif key[2]==0 and key[5] ==1 :
            policy_dic[key] = 'NULL'
        else:
            policy_dic[key] = action_indices[np.argmax(value)]
            # testing
            #if list(key)[action_indices_invert[policy_dic[key]]]==1:
                #print(key, action_indices[np.argmax(value)], '\n')#, policy_dic)
    return policy_dic       

def write_csv(filename):

    ######
    policy_dictionary = get_policy() # index to action name
    keys_policy = sorted(policy_dictionary.keys())

    with open(filename, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames= state_names)
        writer.writeheader()

        for i in keys_policy:
            writer.writerow({'GENRE_FILLED': i[0], 'RATING_FILLED': i[1], 'LANGUAGE_FILLED': i[2], 'GENRE_CONF': i[3], 'RATING_CONF': i[4],'LANGUAGE_CONF': i[5], 'BEST_ACTION' : policy_dictionary[i]})


def write_rewards(rewards_episodes, rewards_filename):
    with open(rewards_filename, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames= ['CURRENT_EPISODE_NUMBER', 'TOTAL_REWARD_AT_END_OF_THIS_EPISODE'])
        writer.writeheader()

        for cnt,value in enumerate(rewards_episodes):
            writer.writerow({'CURRENT_EPISODE_NUMBER': cnt, 'TOTAL_REWARD_AT_END_OF_THIS_EPISODE': value})

    
    
def main():
    model_filename = 'policy-submitted.csv'
    rewards_filename = 'rewards-submitted.csv'
    #episodes = get_episodes() # dictionary for each episode - action: [response1, response2, right response],  dictionary - action: [response1, response2,.....] 
    rewards_episodes = RL_train(episodes)
    write_csv(model_filename)  # write get policy from Q values inside this function, and sort the keys (0,0,0..)
    write_rewards(rewards_episodes,rewards_filename )

    #print('\n', action_indices)
    #print(Q_values[tuple([1,1,0,1,1,0])], action_indices[np.argmax(Q_values[tuple([1,1,0,1,1,0])])], 'LANGUAGE')
    #print(Q_values[tuple([0,1,1,0,1,1])], action_indices[np.argmax(Q_values[tuple([0,1,1,0,1,1])])], 'type')
    #print(Q_values[tuple([1,0,1,1,0,1])], action_indices[np.argmax(Q_values[tuple([1,0,1,1,0,1])])], 'RATING')
    #print(Q_values[tuple([1,1,1,1,0,1])], action_indices[np.argmax(Q_values[tuple([1,1,1,1,0,1])])], 'ex_RATING')

if __name__== "__main__":
  main()
