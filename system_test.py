import csv
import re
from gensim import models
from gensim.models import KeyedVectors
import gensim.downloader as api
from Levenshtein import distance as lev
import nltk

#wv = api.load('word2vec-google-news-300')
wv = models.KeyedVectors.load("w2v_movies.model", mmap='r')

conv_complete = 0
punctuations = set('''!()[]{};:'"\,<>./?@#$%^&*_~''')
stopwords_list = set(nltk.corpus.stopwords.words("english"))
NLU_states = {'REQUEST_GENRE': set(['any', 'comedy', 'action', 'horror', 'biography', 'romance']), 'REQUEST_RATING' : set(['any', '1', '2', '3', '4', '5']), 'REQUEST_LANGUAGE': {'any': [1], 'english': [1], 'telugu':[1], 'hindi': [1], 'persian': [1], 'french': [1], 'chinese':[1]}, 'EXPLICIT_CONFIRM_LANGUAGE': set(['yes', 'no']),'EXPLICIT_CONFIRM_GENRE':set(['yes', 'no']), 'EXPLICIT_CONFIRM_RATING': set(['yes', 'no'])  }
genre_list = ['any', 'comedy', 'action', 'horror', 'biography', 'romance']
language_list = ['any', 'english', 'telugu', 'hindi', 'persian', 'french','chinese']
states = {'GENRE': 'empty', 'RATING':'empty', 'LANGUAGE': 'empty'}
RL_states = {'GENRE_FILLED': 'no', 'RATING_FILLED': 'no', 'LANGUAGE_FILLED': 'no', 'GENRE_CONF': 'no', 'RATING_CONF': 'no', 'LANGUAGE_CONF': 'no'}
map_action_states= {'REQUEST_GENRE': 'GENRE', 'REQUEST_RATING': 'RATING','REQUEST_LANGUAGE' : 'LANGUAGE'}
map_action_RL_states = {'REQUEST_GENRE': 'GENRE_FILLED', 'REQUEST_RATING': 'RATING_FILLED','REQUEST_LANGUAGE': 'LANGUAGE_FILLED', 'EXPLICIT_CONFIRM_LANGUAGE': 'LANGUAGE_CONF', 'EXPLICIT_CONFIRM_RATING' : 'RATING_CONF', 'EXPLICIT_CONFIRM_GENRE' : 'GENRE_CONF'}

# read model
## check this function after building the model
def read_model(modelname):
    model_dictionary = dict()
    with open(modelname, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for cnt, row in enumerate(reader):
            if cnt != 0:
                key = [int(i) for i in row[:-1]]
                key = tuple(key)
                model_dictionary[key] = row[-1]
    return model_dictionary


# read database
def read_database(filename):
    data_dictionary = dict()
    file_handle = open(filename, 'r')
    data = file_handle.readlines()
    data_matrix = []
    for line_index in range(1,len(data)):
        line = data[line_index]
        split_line = line[:-1].split('\t')
        for i in range(len(split_line)):
            if i!=0 and i!=1:
                split_line[i] = split_line[i].lower()

        combinations_tuple = [tuple([split_line[2], split_line[3], split_line[4]]),tuple([split_line[2], split_line[3], 'any']), tuple([split_line[2], 'any', split_line[4]]),tuple(['any', split_line[3], split_line[4]]), tuple(['any', 'any', 'any']), tuple(['any', 'any', split_line[4]]), tuple(['any', split_line[3], 'any']), tuple([split_line[2], 'any', 'any'])]
        for tuple_comb in combinations_tuple:
            if tuple_comb not in data_dictionary:
                data_dictionary[tuple_comb] = [(split_line[0], split_line[1])]
            else:
                data_dictionary[tuple_comb].append((split_line[0], split_line[1]))

    return data_dictionary


def update_states(key_word, action):
    global states
    global RL_states

    if action in map_action_states:
        states[map_action_states[action]] = key_word
        RL_states[map_action_RL_states[action]] = 'yes'
    else:
        #print('action', 'remove print')
        if key_word == 'yes':
            RL_states[map_action_RL_states[action]] = key_word
        else:
            # look out for this bug

            if action ==  'EXPLICIT_CONFIRM_LANGUAGE':
                RL_states['LANGUAGE_CONF'] = 'no'
                RL_states['LANGUAGE_FILLED'] = 'no'
                states['LANGUAGE'] = 'empty'

            elif action == 'EXPLICIT_CONFIRM_RATING':
                RL_states['RATING_CONF'] = 'no'
                RL_states['RATING_FILLED'] = 'no'
                states['RATING'] = 'empty'

            elif action == 'EXPLICIT_CONFIRM_GENRE':
                RL_states['GENRE_CONF'] = 'no'
                RL_states['GENRE_FILLED'] = 'no'
                states['GENRE'] = 'empty'
    #print(states, RL_states)
    
def check_similarity(word,states):
    max_similar = -99
    similar_word = ""
    for state in states:
        current_sim = wv.similarity(word,state)
        if current_sim > max_similar:
            max_similar = current_sim
            similar_word = state
    if max_similar>0.5:
        return similar_word
    else:
        return word

def check_spelling(word,states):
    min_val=99
    threshold=2
    min_state=0
    for state in states:
        if min_val>lev(word, state):
            min_val=lev(word, state)
            min_state=state
    if min_val<=threshold:
        return min_state
    else:
        return word


def NLU(text, action):
    if action!='REQUEST_RATING':

        text = re.sub(r'\d+', '', text)
        text2 = ""

        for char in text:
            if char not in punctuations:
                text2+=char
    else:
        text2 = ""

        for char in text:
            if char not in punctuations:
                text2+=char
    text = text2.split()
    for ind in range(len(text)):
        text[ind] = text[ind].lower()

    keys_dialogue = NLU_states[action]
    new_flag = 0

    for cnt,word in enumerate(text):
        if action=='REQUEST_GENRE' and word not in genre_list:
            if word not in stopwords_list:
                word = check_similarity(word, genre_list)
        if action=='REQUEST_LANGUAGE' and word not in language_list:
            word = check_spelling(word, language_list)
        if word in keys_dialogue:
            if action == 'REQUEST_LANGUAGE' and len(keys_dialogue[word])>=1 and keys_dialogue[word][0]!=1:
                flag = 0
                pass_update = word
                if len(text)<len(keys_dialogue[word])+1:
                    print(len(text), len(keys_dialogue[word])+1)
                    continue
                for index in range(cnt+1,cnt+1+len(keys_dialogue[word])):
                    if keys_dialogue[word][index-cnt-1] != text[index]:
                        flag =1
                    else:
                        pass_update+=(' '+text[index])

                if flag == 0:
                    update_states(pass_update, action)
                    new_flag = 1
            else:
                update_states(word, action) # yes/no confirm
                new_flag = 1
                

     ## no states change, give previous action hardcode
    return new_flag


def RL_agent(model,acc, output):
    # category lets me check type of text to check for, type of cuisine etc.

    state_seq = ['GENRE_FILLED', 'RATING_FILLED', 'LANGUAGE_FILLED', 'GENRE_CONF', 'RATING_CONF','LANGUAGE_CONF']
    key = []
    for i in state_seq:
        if RL_states[i] == 'yes':
            key.append(1)
        else:
            key.append(0)

    key = tuple(key)
    if sum(key) == 6:
        action = 'DATABASE'
    elif output == 0:
        action = model[key]
    else:
        action = model[key]
    return action

def nlp_generate(action, database):
    action_text_dictionary = {'REQUEST_GENRE': 'What genre of movie would you like to watch?', 'REQUEST_RATING': 'What is the movie rating you are looking for (1-5)?' , 'REQUEST_LANGUAGE': 'What language movie would you like to watch?', 'EXPLICIT_CONFIRM_LANGUAGE': 'Okay, you want to watch a movie in '+states['LANGUAGE']+', right?', 'EXPLICIT_CONFIRM_RATING': 'Okay, you would like to watch a movie with a rating of '+states['RATING']+', right?', 'EXPLICIT_CONFIRM_GENRE': 'Okay, you wanted to watch a/an '+states['GENRE']+' movie, right?'}
    global conv_complete

    if action == 'DATABASE':
        key_generate = (states['GENRE'], states['RATING'], states['LANGUAGE'])
        if key_generate in database:
            database_text = database[key_generate]
            out_text = 'I found '+str(len(database_text))+' movie(s) matching your query. \n'
            for value in database_text:
                out_text+= value[0]+' is a/an '+ states['GENRE']+' movie, in '+states['LANGUAGE']+' with a rating of '+ states['RATING']+'.'+' The release date is '+value[1]+'. \n'
        else:
            out_text = 'I found 0 movies matching your query! :( \n'
        conv_complete = 1
    else:
        out_text = action_text_dictionary[action]
    return out_text

def main():

    model_name = 'policy-submitted.csv'
    model = read_model(model_name)
    database_name = 'movieDatabase.txt'
    database = read_database(database_name)
    action = 'None'
    output = -1

    while conv_complete == 0:
        action = RL_agent(model,action, output)
        # set token as 1 when all states are full
        nlp_gen_text = nlp_generate(action, database)
        print('MOVIE_BOT: ',nlp_gen_text)
        if conv_complete != 1:
            text_input = input('USER: ')
            output = NLU(text_input, action)
        else:
            break

if __name__== "__main__":
  main()
