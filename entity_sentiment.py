'''
ENTITY SENTIMENT

This file contains various functions that, given a text and word within the text, estimates sentiment towards the particular word.
Typically the text should be a sentence and the word should be an entity of interest, for example a person, place, or thing.
Determination of sentiment is implemented in 3 different ways:

1. Split method: sentence is split on commas and comparison words. Sentiment is determined from split that contains the word of interest.
2. Neighborhood method: finds the word in the sentence and looks at nearby words to determine sentiment.
3. Tree method: creates a dependency tree of the sentence using the Stanford dependency parser. Sentiment is determined by looking at
words that appear nearby in the tree to the word of interest.

The function compile_ensemble_sentiment combines all three of these methods and is generally the most robust way to determine sentiment
towards an entity.
'''



from IPython.display import display
import string
import numpy as np
from copy import copy

# set java path
import os
java_path = '/path/to/your/Java/home'
os.environ['JAVAHOME'] = java_path

# import dependency parser
from nltk.parse.stanford import StanfordDependencyParser
sdp = StanfordDependencyParser(path_to_jar = '/path/to/stanford-parser.jar',
                     path_to_models_jar = '/path/to/stanford-parser-3.5.2-models.jar')

# import sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()



def display_dep_tree(sentence):
    dep_result = list(sdp.raw_parse(sentence))

    # print the dependency tree
    dep_tree = [parse.tree() for parse in dep_result][0]
    print(dep_tree)
    display(dep_tree)



class DependencyTreeDict:
    '''
    A DependencyTreeDict object is initialized with a sentence. Object is a dictionary which encodes a sentences dependency tree.
    Object methods allow for easy retrieval of parents, children, siblings, etc. for a word in the dependency tree. Objects of this
    class are created using the Stanford Dependency Parser.
    '''

    def __init__(self, sentence):
        '''
        Initializes an instance of the DependencyTreeDict class.
        '''

        tree_dict = {}

        # Create the dependency tree
        dep_result = list(sdp.raw_parse(sentence))
        dep_tree = [parse.tree() for parse in dep_result][0]

        tree_word_list = str(dep_tree).replace('(', '').replace(')', '').split(' ')
        tree_positions =  dep_tree.treepositions()

        # Add word, level, and position to dictionary
        word_count = 0
        for word, pos in zip(tree_word_list, tree_positions):
            temp_dict = {}
            depth = len(pos)
            temp_dict['word'] = word
            temp_dict['depth'] = depth
            temp_dict['tree_position'] = pos
            temp_dict['children'] = []

            # Determine if word has parents
            if depth == 0:
                temp_dict['parent'] = 0
            # Find the parent
            else:
                for i in range(word_count):
                    key = word_count - i - 1

                    # Once parent is found assign parent and children
                    if tree_dict[key]['depth'] == depth - 1:
                        temp_dict['parent'] = key
                        tree_dict[key]['children'].append(word_count)
                        break

            tree_dict[word_count] = temp_dict
            word_count += 1


        self.tree_dict = tree_dict

    def get_parent(self, word_pos):
        '''
        Get parent of a particular node in tree graph
        '''
        return self.tree_dict[word_pos]['parent']



    def get_children(self, word_pos, word_list = [], get_all = False):
        '''
        Gets all descendants of a certain node in tree graph
        '''
        # Get immediate children
        children = self.tree_dict[word_pos]['children']

        # If no children return an empty list
        if children == []:
            return word_list

        # If children exist add them to word list
        word_list += children

        # Call get children for each child. This will capture grand-children etc.
        if get_all == True:
            for child in children:
                self.get_children(child, word_list = word_list, get_all = True)

        # After all recursion return word_list
        return word_list



    def get_adjacents(self, word_pos):
        '''
        Get adjacent words of a particular node on tree graph
        '''
        words = self.get_children(word_pos, word_list = [], get_all = False)
        if word_pos != 0:
            words.append(self.get_parent(word_pos))
        return words



    def get_siblings(self, word_pos):
        '''
        Get siblings of a particular node on tree graph
        '''
        if word_pos == 0:
            return []
        else:
            parent = self.get_parent(word_pos)
            children = self.get_children(parent, word_list = [], get_all = False)
            siblings = list(set(children) - set([word_pos]))
            return siblings



    def get_words_by_dist(self, word_pos, dist):
        '''
        Gets all words within a particular distance on tree graph
        '''
        # Initialize words
        if type(word_pos) == list:
            words = word_pos
        else:
            words = [word_pos]

        # iterate through distance and continue to add adjacents to the list
        for i in range(dist):
            # distance 0 takes a word and all of its siblings
            new_words = copy(words)
            for word in words:
                new_words += self.get_adjacents(word)
            words = list(set(new_words))

        return words


def tree_sentiment(word_pos, tree_dict, decay = 0.5, propagation = 10):
    '''
    Takes as input a word position (typically a named entity) and a DependencyTreeDict object.
    Output is the compiled sentiment for the word at the specified position.
    Decay controls the decay of sentiment across tree distance.
    Propagation controls how far to search through tree
    '''

    # Set key parameters
    first_sent = 0
    compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

    # Iterate through propagation
    # Add children before parents
    for i in range(propagation):
        phrase = ""

        # On even iterations get words by distance
        if i % 2 == 0:
            dist = int(i/2 + 1)
            words = tree_dict.get_words_by_dist(word_pos, dist)

        # On odd iterations get the same words as previous, but add children
        if i % 2 == 1:
            dist = int(i/2 + 0.5)
            words = tree_dict.get_words_by_dist(word_pos, dist)
            new_words = copy(words)
            for word in words:
                new_words += tree_dict.get_children(word, word_list = [], get_all = False)
            words = list(set(new_words))

        for word in words:
            phrase += ' ' + tree_dict.tree_dict[word]['word']

        # Remove the entity from phrase
        phrase = phrase.replace(tree_dict.tree_dict[word_pos]['word'], '')

        sentiment = analyzer.polarity_scores(phrase)

        # Add sentiment scores into sentiment dictionary as weighted average that decreases
        # the more levels up the algorithm has compiled. Only count levels after we've established
        # some significant sentiment.

        if first_sent == 1:
            weight = (decay)**(i - first_sent_level)
            for key in compiled_sentiment.keys():
                compiled_sentiment[key] = (compiled_sentiment[key] + sentiment[key] * weight) / (1 + weight)

        if first_sent == 0 and sentiment['neu'] < 0.9:
            if sum(sentiment.values()) != 0:
                first_sent = 1
                first_sent_level = i
                compiled_sentiment = sentiment

    return compiled_sentiment



def compile_tree_sentiment(sentence, entity, tree_dict = False, decay = 0.5, propagation = 10):
    '''
    Takes as input a sentence and an entity. Creates a word tree dictionary from the sentence, then finds
    the location(s) of the entity in the word tree dictionary. For each location the function gets the
    tree based sentiment, then compiles sentiment for each location.
    '''
    compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    sentence = sentence.lower().replace(',', '')
    if tree_dict == False:
        tree_dict = DependencyTreeDict(sentence)
    else:
        tree_dict = tree_dict

    # Find where the entity occurs in the tree dictionary
    entity_locs = []
    for key in tree_dict.tree_dict.keys():
        if tree_dict.tree_dict[key]['word'] == entity.lower():
            entity_locs.append(key)

    # Number of times the entity appears
    n_locs = len(entity_locs)

    # Only execute rest of function if entity is actually in sentence
    if n_locs != 0:

        # For each instance of the entity get tree sentiment
        # Add to compiled sentiment
        for loc in entity_locs:
            sentiment = tree_sentiment(loc, tree_dict)
            for key in compiled_sentiment.keys():
                compiled_sentiment[key] += sentiment[key]

        # Divide by n_locs to get average
        for key in compiled_sentiment.keys():
            compiled_sentiment[key] = compiled_sentiment[key] / n_locs

    return compiled_sentiment, tree_dict



def find_all(a_str, sub):
    '''
    Find all matches of sub within a_str. Returns starting index of matches.
    '''
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches



def compile_split_sentiment(sentence, entity):
    '''
    Split the sentence by comparison words and commas. Determine which sections the entity is in.
    Return average sentiment for those sections.
    '''
    # List of comparison words
    comp_words = ['but', 'however', 'albeit', 'although', 'in contrast', 'in spite of', 'though', 'on one hand', 'on the other hand',
                  'then again', 'even so', 'unlike', 'while', 'conversely', 'nevertheless', 'nonetheless', 'notwithstanding', 'yet']

    # Lowercase sentence and split on commas
    sentence = sentence.lower()
    sentence = sentence.split(',')

    # Iterate through sections and split them based on comparison words
    splits = []
    for section in sentence:

        all_comps = []
        for word in comp_words:
            # Use find all function to find location of comparison words
            all_comps += list(find_all(section, word))

        # Sort list of comparison words indexes
        all_comps.sort()

        # Split the section and append to splits
        last_split = 0
        for comp in all_comps:
            splits.append(section[last_split:comp])
            last_split = comp
        splits.append(section[last_split:])

    # Find the sections where the entity has been named
    # Add sentiment for that section to list
    sentiments = []
    for section in splits:
        if entity.lower() in section:
            # remove entity from section
            cleaned_section = section.replace(entity.lower(), '')
            sentiments.append(analyzer.polarity_scores(cleaned_section))

    # Add sentiment for each section up
    compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    for sentiment in sentiments:
        for key in compiled_sentiment.keys():
            compiled_sentiment[key] += sentiment[key]

    # Divide all sections by lenth of sentiments list to get average
    denom = len(sentiments)
    if denom != 0:
        for key in compiled_sentiment.keys():
            compiled_sentiment[key] = compiled_sentiment[key] / denom

    return compiled_sentiment



def compile_neighborhood_sentiment(sentence, entity, decay = 0.5, propagation = 10):
    '''
    Find instances of entity in sentence. Add sentiment of neighboring words. Incrementally expand neighborhood
    up to limit set by propagation variable. Add up sentiment with decay as neighborhood expands.
    '''
    first_sent = 0
    compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

    sentence = sentence.lower().replace(',', '').split(' ')

    sen_len = len(sentence)
    ent_locs = []

    # Find locations of entity within sentence
    for i, word in enumerate(sentence):
        if word == entity.lower():
            ent_locs.append(i)

    # how many entity locations there are
    n_locs = len(ent_locs)

    # Only execute rest of function if word is actually in sentence
    if n_locs != 0:

        # Iterate through propagation parameter
        for i in range(propagation):
            neighborhoods = []

            # Compile list of entity neighborhoods
            for loc in ent_locs:
                exp_locs = list(range(loc-i-1,loc+i+2))
                neigh = []
                # Only add locations to neighborhoods that are actually in the sentence
                for j in exp_locs:
                    if j >= 0 and j < sen_len:
                        neigh.append(sentence[j])
                # Join all words in neighborhood then add to neighborhoods list
                neigh = ' '.join(neigh)
                # Remove entity
                neigh = neigh.replace(entity.lower(), '')

                neighborhoods.append(neigh)

            # Get average sentiment for all neighborhoods
            sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

            for neigh in neighborhoods:
                neigh_sent = analyzer.polarity_scores(neigh)
                # Add up sentiments
                for key in sentiment.keys():
                    sentiment[key] += neigh_sent[key]

            # Divide by n_locs to get average
            for key in sentiment.keys():
                sentiment[key] = sentiment[key] / n_locs

            # Compile into main sentiment
            if first_sent == 1:
                weight = (decay)**(i - first_sent_level)
                for key in compiled_sentiment.keys():
                    compiled_sentiment[key] = (compiled_sentiment[key] + sentiment[key] * weight) / (1 + weight)

            if first_sent == 0 and sentiment['neu'] < 0.9:
                if sum(sentiment.values()) != 0:
                    first_sent = 1
                    first_sent_level = i
                    compiled_sentiment = sentiment

    return compiled_sentiment



def compile_ensemble_sentiment(sentence, entity, tree_dict = False):
    '''
    Determines sentiment using three different methods: neighborhood, tree, and split.
    Sentiment for each method is compiled to return a final sentiment.
    '''
    compiled_sentiment = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}


    tree_sentiment, tree_dict = compile_tree_sentiment(sentence, entity, tree_dict = tree_dict)
    neighborhood_sentiment = compile_neighborhood_sentiment(sentence, entity)
    split_sentiment = compile_split_sentiment(sentence, entity)

    all_sentiments = [tree_sentiment, neighborhood_sentiment, split_sentiment]

    # Add up all sentiment
    for sent in all_sentiments:
        for key in compiled_sentiment.keys():
            compiled_sentiment[key] += sent[key]

    # Divide by 3 to get average
    for key in compiled_sentiment.keys():
        compiled_sentiment[key] = compiled_sentiment[key] / 3

    return compiled_sentiment, tree_dict
