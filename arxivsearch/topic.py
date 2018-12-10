import abc
import six

#setting up some exceptions
class InvalidTopic(TypeError):
    def __init__(self, arg):
        print("""Incorrect type for initialising topic. 
        Input {} of type {} """.format(arg, type(arg)))
        print(""" Expecting either "bnn","interpretable", "fairness" or "variational" """) 
        


@six.add_metaclass(abc.ABCMeta)
class topic(object):
    """Abstract Class for search topics
    
    Class defines methods for formatting search terms.
    Any topic that you want to search for should inherit from this class,
    and define the exact general and specific terms you want to search for.
    """
    def __init__(self, dirname, category=None):
        """Initialises variables and path
        
        Args:
          dirname (str):
            name of the directory where the results will be stored
            Will also be the name of the topic you a sreaching for
          category (default=None):
            The arXiv categories you want to search under,
            Eg. cs.CV, stat.ML etc. If nothing is specified, will use the 
            parameters specified in this init method.
        """
        self.dirname = dirname
        if category == None:
            self.category = ['cs.CV', 'cs.LG', 'cs.CL', 'cs.NE', 'stat.ML']
        else:
            self.category = category
        self.general_terms = []
        self.specific_terms = []

    
    def format_search(self):
        """
        format_search()
        Description:
        Will create the complete list of different combination
        of search terms for you.

        Keyword Arguments:
        
        Returns:
            (list(string)) of all different search combinations
        """
        search = []
        for sub in self.category:
            for gen in self.general_terms:
                search_string = 'cat:{}+AND+%28ti:{}+OR+abs:{}%29+AND+%28'.format(
                    sub, gen, gen)

                #remove the last +OR+ from the string
                for spec in self.specific_terms:
                    search_string += 'all:' + spec +  '+OR+'
                #remove the last +OR+ from the string
                search_string = search_string[:-4] + '%29'
                #add the suffix to say how to order the list
                search_string += '&sortBy=submittedDate&sortOrder=descending'
                #now add this search string to the list
                search.append(search_string)
        return search

        
class bnn_topic(topic):
    def __init__(self):
        topic.__init__(self, 'bnn')
        self.general_terms = [
            'bayesian',
            'probabilistic'
        ]
        self.specific_terms = [
            'convol',
            'neural',
            'uncertain',
            'variational',
            'inference',
            'marginali'
        ]


class fairness_topic(topic):
    def __init__(self):
        topic.__init__(self, 'fairness')
        self.general_terms = [
            'fairness',
            'bias',
            'ethic',
            'equality',
            'equal'
        ]
        self.specific_terms = [
            'algorithms',
            'accountability',
            'transparency',
            'discrimination',
            'legal',
            'social',
            'unfair',
            'audit',
            '%22machine+learning%22',
            '%22artificial+intelligence%22',
            'statistics'
        ]


class interpretable_topic(topic):
    def __init__(self):
        topic.__init__(self, 'interpretable')
        self.general_terms = [
            'interpretab',
            'explain',
            'understandab',
            'artificial'
            'machine'
            'learning'
        ]
        self.specific_terms = [
            'reason',
            'cause',
            'trust'
        ]

        

class variational_topic(topic):
    #also include Monte Carlo methods
    def __init__(self):
        topic.__init__(self, 'variational')
        self.general_terms = [
            '%22variational+inference%22',
            'approximat',
            'bayesian',
            'probabilistic',
            'uncertain'            
        ]
        self.specific_terms = [
            'ELBO',
            'divergence'
            'marginali',
            'uncertain',
            '%22Monte+Carlo%22',
            'MCMC',
            'stochastic',
            'reparameterization'
        ]
        
        
def get_topic(arg):
    if not isinstance(arg, str):
        raise InvalidTopic(arg)

    if('bnn' in arg):
        return bnn_topic()
    elif('fairness' in arg):
        return fairness_topic()
    elif('interpretable' in arg):
        return interpretable_topic()
    elif('variational' in arg):
        return variational_topic()
    else:
        raise InvalidTopic(arg)

    
