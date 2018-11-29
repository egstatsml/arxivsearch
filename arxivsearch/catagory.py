#setting up some exceptions
class InvalidCatagory(TypeError):
    def __init__(self, arg):
        print("""Incorrect type for initialising catagory. 
        Input {} of type {} """.format(arg, type(arg)))
        print(""" Expecting either "bnn","interpretable", "fairness" or "variational" """) 
        


#setting this up as a base class
#essentially just virtual class variables
class catagory(object):
    def __init__(self, dirname, subject=None):
        self.dirname = dirname
        if subject == None:
            self.subject = ['cs.CV', 'cs.LG', 'cs.CL', 'cs.NE', 'stat.ML']
        else:
            self.subject = subject
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
        for sub in self.subject:
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

        
class bnn_catagory(catagory):
    def __init__(self):
        catagory.__init__(self, 'bnn')
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


class fairness_catagory(catagory):
    def __init__(self):
        catagory.__init__(self, 'fairness')
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


class interpretable_catagory(catagory):
    def __init__(self):
        catagory.__init__(self, 'interpretable')
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

        

class variational_catagory(catagory):
    #also include Monte Carlo methods
    def __init__(self):
        catagory.__init__(self, 'variational')
        self.general_terms = [
            'variational',
            'inference',
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
        
        
def get_catagory(arg):
    if not isinstance(arg, str):
        raise InvalidCatagory(arg)

    if('bnn' in arg):
        return bnn_catagory()
    elif('fairness' in arg):
        return fairness_catagory()
    elif('interpretable' in arg):
        return interpretable_catagory()
    elif('variational' in arg):
        return variational_catagory()
    else:
        raise InvalidCatagory(arg)

    
