import os, pickle, warnings, types, re
from time import perf_counter
from tqdm import tqdm
import numpy as np
import scipy.special as sp
from multiprocessing import Pool, cpu_count

def map_id_to_card(id:int): return (id // 13, id % 13) # returns (suit, rank)

def _dataframe_str(data, columns = ['S','H','D','C'], index = ['N','E','S','W']):
    return '  '+''.join([str(c).rjust(3) for c in columns]) + '\n' + \
        '\n'.join([str(index[i]).ljust(2) + ''.join([str(data[i,j]).rjust(3) for j in range(data.shape[1])]) for i in range(data.shape[0])])

def solve_one_board(pbn:str):
    import dds, ctypes
    solver = dds.ddTableDealPBN()
    resTable = ctypes.pointer(dds.ddTableResults())
    dds.SetMaxThreads(0)
    ctypes.create_string_buffer(80)
    solver.cards = bytes(pbn, 'utf-8')
    dds.CalcDDtablePBN(solver, resTable)
    res = resTable.contents.resTable
    S=[res[0][k] for k in [0,1,2,3]]
    H=[res[0][4],res[1][0],res[1][1],res[1][2]]
    D=[res[1][3],res[1][4],res[2][0],res[2][1]]
    C=[res[2][2],res[2][3],res[2][4],res[3][0]]
    N=[res[3][k] for k in [1,2,3,4]]
    return np.stack([N, S, H, D, C]).T

def calc_par(dds_result: np.ndarray):
    '''input: (4,5) array of double dummy results
    returns: (4,4) array of par scores, axis 0 = dealer, axis 1 = (Non, NS, EW, Both)'''
    assert dds_result.shape == (4,5), 'DDS result must be a (4,5) array.'
    

class Hand():
    def __init__(
            self, 
            given = None,
            **_
        ):

        if isinstance(given, Hand): # copies input hand
            self.array_rep = given.array_rep.copy()
            self.fixed = given.fixed.copy()
        elif given is not None:
            assert given.shape == (4,4,13), 'Given hand must be a (4,4,13) numpy array.'
            self.array_rep = given.astype('?')
            self.fixed = given.any(axis = 0) # axes 0,1 = suit, card
        else: 
            self.array_rep = np.zeros((4,4,13)).astype('?') # axes 0,1,2 = player, suit, card
            self.fixed = np.zeros((4,13)).astype('?') # axes 0,1 = suit, card
        self.shape = self._get_shape() # axes 0,1 = player, suit
        self.hcp = self._get_hcp() # axes 0 = player
        self.pbn = ''
        self.dds = None

    def __str__(self): return self._get_str()

    def _dealt(self):
        return self.array_rep.sum() == 52

    def _get_shape(self):
        self.shape = self.array_rep.sum(axis=2)
        return self.array_rep.sum(axis=2)
    
    def _get_hcp(self):
        hcp_filter = np.zeros((4,13)).astype(int)
        hcp_filter[:,:4] = np.repeat([[4,3,2,1]], 4, axis = 0)
        self.hcp = (self.array_rep * hcp_filter).sum(axis=(1,2))
        return self.hcp

    def _get_pbn(self):
        card_filter = np.array(['A','K','Q','J','T','9','8','7','6','5','4','3','2'])
        pbn = []
        for i in range(4):
            tmp = []
            for j in range(4): tmp.append(''.join(card_filter[self.array_rep[i,j]]))
            pbn.append('.'.join(tmp))
        self.pbn = '"N:'+' '.join(pbn)+'"'
        return self.pbn
    
    def _get_str(self):
        suit_filter = np.array(['S','H','D','C'])
        card_filter = np.array(['A','K','Q','J','T','9','8','7','6','5','4','3','2'])
        out = []
        # north hand
        for i in range(4):
            out.append(' '*9+ suit_filter[i]+' '+(''.join(card_filter[self.array_rep[0,i,:]])).ljust(23))
        
        # EW hands
        for i in range(4):
            out.append(
                suit_filter[i] + ' ' + (''.join(card_filter[self.array_rep[3,i,:]])).ljust(15) +
                suit_filter[i] + ' ' + (''.join(card_filter[self.array_rep[1,i,:]])).ljust(15)
            )
        
        # south hand
        for i in range(4):
            out.append(' '*9+ suit_filter[i]+' '+(''.join(card_filter[self.array_rep[2,i,:]])).ljust(23))
        return '\n'.join(out)
    
    def _dealcard(self, hand:int, suit:int, rank:int):
        if self.fixed[suit, rank]: raise ValueError('Cannot deal a fixed card.')
        if self.array_rep[hand, suit, rank]: raise ValueError('Card already dealt to this hand.')
        self.array_rep[hand, suit, rank] = True
        self.shape = self._get_shape()

    def _to_deal(self): return ~self.array_rep.any(axis=0)

    def reset(self): self.array_rep[:, ~self.fixed] = False

    def copy(self): return Hand(given = self.array_rep.copy())

    def solve_dds(self): self.dds = solve_one_board(self._get_pbn())

class SuitPermuter():
    def __init__(self, overall = False, **kwargs):
        if isinstance(overall, bool):
            self.s = overall; self.h = overall; self.d = overall; self.c = overall
        elif len(overall) == 4:
            self.s = overall[0]; self.h = overall[1]; self.d = overall[2]; self.c = overall[3]
        else: raise ValueError('Overall parameter should be an array of length 4 or boolean')
        for k,v in kwargs.items():
            if k not in ['s','h','d','c']: continue
            if not isinstance(v, bool): raise ValueError(f'SuitPermuter argument {k} must be a boolean.')
            setattr(self, k, v)
        
        permutable = []
        if self.s: permutable.append(0)
        if self.h: permutable.append(1)
        if self.d: permutable.append(2)
        if self.c: permutable.append(3)
        self.permutable = permutable
    
    def __str__(self):
        if any(self.permutable):
            return 'Following suits can be exchanged:' + \
                ', '.join(np.array(['S','H','D','C'])[self.permutable].tolist())
        else: return 'All suits are as given and not exchangeable.'
    
    def __eq__(self, other):
        if not isinstance(other, SuitPermuter): return False
        return (self.permutable == other.permutable)

    def permute(self, hand: Hand):
        perm = np.arange(4)
        np.random.shuffle(perm[self.permutable])
        hand.array_rep = hand.array_rep[:, perm, :]
        hand.shape = hand.shape[:, perm]
        return hand
    
    def all_shapes(self, hand:Hand):
        _ = hand._get_shape()
        if not any(self.permutable): return [hand.shape]
        from itertools import permutations
        perm = np.arange(4)
        permutable_suits = [i for i in perm if i in self.permutable]
        all_perms = set(permutations(permutable_suits))
        all_shapes = []
        for perm in all_perms:
            shape = hand.shape.copy()
            shape[:, permutable_suits] = shape[:, list(perm)]
            all_shapes.append(shape)
        return all_shapes

class HCPConstraint():
    def __init__(
        self,
        hcp_max:np.ndarray = np.ones(4) * 37,
        hcp_min:np.ndarray = np.zeros(4),
        **_ # additional params are ignored
    ):
        
        hcp_max = np.array(hcp_max); hcp_min = np.array(hcp_min)
        # sense check
        for cond in [hcp_max, hcp_min]:
            assert len(cond) == 4, 'Each condition array must have exactly 4 elements.'
        assert (37 >= hcp_max).all() and (hcp_max >= hcp_min).all() and (hcp_min >= 0).all(), \
            '0 <= min HCP <= max HCP <= 37'
        assert sum(hcp_max) >= 40 and sum(hcp_min) <= 40, 'Total HCP must be exactly 40'
        self.hcp_max = hcp_max.astype(int) # if hcp_max != np.ones(4) * 37 else None
        self.hcp_min = hcp_min.astype(int) # if hcp_min != np.zeros(4) else None

        # flag for quicker dealing
        self.no_hcp_cond = (self.hcp_max >= 37).all() and (self.hcp_min == 0).all()
    
    def __str__(self):
        return 'HCP Constraint:\n' + \
            '    North East  South West\n' + \
            'Max '+ ' '.join([str(x).ljust(5) for x in self.hcp_max]) + '\n' + \
            'Min '+ ' '.join([str(x).ljust(5) for x in self.hcp_min])

    def copy(self):
        return HCPConstraint(self.hcp_max.copy(), self.hcp_min.copy())

    def check(self, hand: Hand):
        if self.no_hcp_cond: return True
        _ = hand._get_hcp() # forcibly updates the HCP attribute of the hand
        return all(hand.hcp <= self.hcp_max) and all(hand.hcp >= self.hcp_min)

class SingleHandShapeConstraint():
    def __init__(
        self,
        hand : int = 0, # 0 = North, 1 = East, 2 = South, 3 = West
        suit_max:np.ndarray = np.repeat(13, 4),
        suit_min:np.ndarray = np.zeros(4),
        permute_suits = False, # True if you want to set longest suit etc.
        **_ # additional params are ignored
    ):
        if isinstance(suit_max, (int, float)): suit_max = np.repeat(suit_max, 4).astype(int)
        if isinstance(suit_min, (int, float)): suit_min = np.repeat(suit_min, 4).astype(int)
        suit_max = np.array(suit_max); suit_min = np.array(suit_min)
        # sense check
        assert 0 <= hand <= 3, 'Hand must be an integer between 0 and 3.'
        assert (13 >= suit_max).all() and (suit_max >= suit_min).all() and (suit_min >= 0).all(), \
            '0 <= min cards <= max cards <= 13'
        assert sum(suit_max) >= 13 and sum(suit_min) <= 13, 'Hands must be exactly 13 cards'
        
        self.hand = hand
        self.suit_min = suit_min.astype(int); self.suit_max = suit_max.astype(int)
        self.permute_suits = permute_suits if isinstance(permute_suits, SuitPermuter) else SuitPermuter(permute_suits)

        # flag for quicker dealing
        self.no_shape_cond = (suit_max == 13).all() and (suit_min == 0).all()

    def __str__(self):
        out = [f'Shape constraint for hand {["North","East","South","West"][self.hand]}:',
               f'Max number of cards: S={self.suit_max[0]}, H={self.suit_max[1]}, D={self.suit_max[2]}, C={self.suit_max[3]}',
               f'Min number of cards: S={self.suit_min[0]}, H={self.suit_min[1]}, D={self.suit_min[2]}, C={self.suit_min[3]}',
               self.permute_suits.__str__()]
        return '\n'.join(out)

    def copy(self):
        return SingleHandShapeConstraint(self.hand, self.suit_max.copy(), self.suit_min.copy(), self.permute_suits)
    
    def check(self, hand: Hand):
        if self.no_shape_cond: return True
        _ = hand._get_shape() # forcibly updates the shape attribute of the hand
        for shape in self.permute_suits.all_shapes(hand):
            shape = shape[self.hand,:]
            if (shape <= self.suit_max).all() and (shape >= self.suit_min).all():
                return True
        return False

class ShapeConstraint():
    def __new__(
        cls,
        shape_max = np.ones((4,4))*13, # axis 0 = hand, axis 1 = suit
        shape_min = np.zeros((4,4)),
        permute_suits: SuitPermuter | bool = False, # True if you want to set longest suit etc.
        **_ # additional params are ignored
    ):
        shape_max = np.array(shape_max); shape_min = np.array(shape_min)
        # sense check
        for cond in [shape_max, shape_min]: assert cond.shape==(4,4), 'Shape specifications should be (4,4) arrays'
        assert np.all(shape_max.sum(axis=0) >= 13), 'The sum of max cards in all suits must be at least 13.'
        assert np.all(shape_max.sum(axis=1) >= 13), 'The sum of max cards in all hands must be at least 13.'
        assert np.all(shape_min.sum(axis=0) <= 13), 'The sum of min cards in all suits must be no more than 13.'
        assert np.all(shape_min.sum(axis=1) <= 13), 'The sum of min cards in all hands must be no more than 13.'
        assert (13 >= shape_max).all() and (shape_max >= shape_min).all() and (shape_min >= 0).all(), \
            '0 <= min cards <= max cards <= 13'

        shape_max[:,shape_min.sum(axis = 0) == 13] = 13 # if a suit is fixed
        shape_max[:,shape_min.sum(axis = 1) == 13] = 13 # if a hand is fixed

        # check if conditions are only on a single hand
        no_cond = np.zeros(4).astype('?')
        for i in range(4):
            if (shape_max[i,:] == 13).all() and (shape_min[i,:] == 0).all(): no_cond[i] = True
        if sum(no_cond) == 3:
            idx = np.where(no_cond == False)[0][0]
            return SingleHandShapeConstraint(
                idx, shape_max[idx,:], shape_min[idx,:], permute_suits = permute_suits
            )
        return super().__new__(cls)
    
    def __init__(
            self,
            shape_max = np.ones((4,4))*13, # axis 0 = hand, axis 1 = suit
            shape_min = np.zeros((4,4)),
            permute_suits: SuitPermuter | bool = False, # True if you want to set longest suit etc.
            **_ # additional params are ignored
        ):
        self.shape_max = shape_max.astype(int)
        self.shape_min = shape_min.astype(int)
        self.permute_suits = SuitPermuter(permute_suits) if isinstance(permute_suits, bool) else permute_suits

        # flag for quicker dealing
        self.no_shape_cond = (self.shape_max == 13).all() and (self.shape_min == 0).all()

    def __str__(self):
        out = ['Shape constraint:', 'Max number of cards:',
               _dataframe_str(self.shape_max, columns=['S','H','D','C'], index=['N','E','S','W']).__str__(),
               'Min number of cards:',
               _dataframe_str(self.shape_min, columns=['S','H','D','C'], index=['N','E','S','W']).__str__(),
               'Following suits can be exchanged:',
               ', '.join(np.array(['S','H','D','C'])[self.permute_suits.permutable].tolist())]
        return '\n'.join(out)

    def copy(self):
        return ShapeConstraint(self.shape_max.copy(), self.shape_min.copy(), self.permute_suits)

    def check(self, hand: Hand):
        if self.no_shape_cond: return True
        _ = hand._get_shape() # forcibly updates the shape attribute of the hand
        for shape in self.permute_suits.all_shapes(hand):
            if (shape <= self.shape_max).all() and (shape >= self.shape_min).all():
                return True
        return False

class Constraint():
    def __init__(
        self,
        shape_constraint : ShapeConstraint | SingleHandShapeConstraint | types.NoneType = None,
        hcp_constraint : HCPConstraint | types.NoneType = None,
        **kwargs
    ):
        if shape_constraint is not None: self.shape_constraint = shape_constraint
        elif len(kwargs) > 0: self.shape_constraint = ShapeConstraint(**kwargs)
        else: self.shape_constraint = None
        if hcp_constraint is not None: self.hcp_constraint = hcp_constraint
        elif len(kwargs) > 0: self.hcp_constraint = HCPConstraint(**kwargs)
        else: self.hcp_constraint = None
        self.no_shape_cond = self.shape_constraint is None or self.shape_constraint.no_shape_cond
        self.no_hcp_cond = self.hcp_constraint is None or self.hcp_constraint.no_hcp_cond
        self.permute_suits = self.shape_constraint.permute_suits if self.shape_constraint is not None else SuitPermuter()

    def __str__(self): return self.shape_constraint.__str__() + '\n\n' + self.hcp_constraint.__str__()

    def copy(self):
        return Constraint(
            self.shape_constraint.copy() if self.shape_constraint is not None else None,
            self.hcp_constraint.copy() if self.hcp_constraint is not None else None
        )

    def check(self, hand: Hand):
        if not self.no_shape_cond and not self.shape_constraint.check(hand): return False
        if not self.no_hcp_cond and not self.hcp_constraint.check(hand): return False
        return True

class DDSConstraint():
    def __init__(
        self,
        dds_max = np.ones((4,5)) * 14,
        dds_min = np.zeros((4,5)),
        par_max = np.inf, par_min = -np.inf,
    ):
        self.dds_max = dds_max.astype(int); self.dds_min = dds_min.astype(int)
        self.par_max = par_max; self.par_min = par_min

def combine_single_hand_shape_constraints(*constraints: SingleHandShapeConstraint | types.NoneType):
    shape_max = np.ones((4,4))*13; shape_min = np.zeros((4,4))
    permute_suits = None
    for constraint in constraints:
        if constraint is None: continue
        shape_max[constraint.hand,:] = constraint.suit_max
        shape_min[constraint.hand,:] = constraint.suit_min
        if constraint.permute_suits != permute_suits and permute_suits != None:
            raise ValueError('Cannot combine SingleHandShapeConstraints with different permute_suits settings.')
        else: permute_suits = constraint.permute_suits
    return ShapeConstraint(shape_max, shape_min, permute_suits = permute_suits)

def parse_string(constraint_string: str):
    '''
    Constraint string format:
        north (conditions) east (conditions) south (conditions) west (conditions)
        no spaces are allowed within each condition
        following conditions are accepted:
            5+S / 3-C / 3-5H : shape constraint for each suit
            5332 / 5(3)3(2) / 5(332) : shape constraint for all suits, must add to 13; suits in brackets can be permuted
                this constraint cannot be used for suits with 10+ cards
            longest>=6 / second==4 / third<3 / shortest==0 : shape constraint for longest/second/third/shortest suits
            --- above constraints are mutually exclusive ---
            10+HCP / 10-13P / 3-P : HCP constraint
            hascard SA 10D CT (...) : given cards (accepts both 10 and T)
    everything can be upper or lower case
    '''
    constraint_string = constraint_string.lower().split()

    # first separate conditions to each hand
    hand_conditions = {'north':[], 'east':[], 'south':[], 'west':[]}
    current_hand = ''
    for token in constraint_string:
        if token in hand_conditions.keys(): current_hand = token
        elif current_hand == '': raise ValueError('Constraint string must start with a hand name (north/east/south/west).')
        else: hand_conditions[current_hand].append(token)
    
    def classify_consition(token:str):
        if re.match(r'^\d{1,2}[\+\-]\d{0,2}[shdc]$', token): return 'shape_suit'
        elif re.match(r'^[0-9()]*$', token): return 'shape_all'
        elif re.match(r'^(longest|second|third|shortest)([<>=])(={0,1})(\d{1,2})$', token): return 'shape_ranked'
        elif re.match(r'^\d{1,2}[\+\-]\d{0,2}(hcp|p)$', token): return 'hcp'
        else: raise ValueError(f'Unrecognized constraint token: {token}; given cards must follow the "hascard" keyword.')

    def parse_shape_suit(*tokens:str):
        suit_map = {'s':0, 'h':1, 'd':2, 'c':3}
        shape_max = np.ones(4)*13; shape_min = np.zeros(4)
        for token in tokens:
            match = re.match(r'^(\d{1,2})([\+\-])(\d{0,2})([shdc])$', token)
            number = int(match.group(1))
            operator = match.group(2)
            limit = match.group(3)
            suit = suit_map[match.group(4)]
            if operator == '+':
                shape_min[suit] = max(shape_min[suit], number)
            elif operator == '-':
                if limit == '': shape_max[suit] = min(shape_max[suit], number)
                else: shape_min[suit] = max(shape_min[suit], number); shape_max[suit] = min(shape_max[suit], int(limit))
        return shape_max, shape_min, SuitPermuter(False)
    
    def parse_shape_all(token:str):
        permute_suits = []
        suit_lengths = []
        in_bracket = False
        for char in token:
            if char == '(': in_bracket = True; continue
            if char == ')': in_bracket = False; continue
            if char.isdigit():
                suit_lengths.append(int(char))
                permute_suits.append(in_bracket)
        shape_max = np.array(suit_lengths).astype(int); shape_min = np.array(suit_lengths).astype(int)
        assert shape_max.sum() == 13, 'Total cards in shape constraint must sum to 13.'
        return shape_max, shape_min, SuitPermuter(permute_suits)

    def parse_shape_ranked(*tokens:str):
        shape_max = np.ones(4)*13; shape_min = np.zeros(4)
        rank_map = {'longest':0, 'second':1, 'third':2, 'shortest':3}
        for token in tokens:
            match = re.match(r'^(longest|second|third|shortest)([<>=])(={0,1})(\d{1,2})$', token)
            rank = rank_map[match.group(1)]
            operator = match.group(2)
            equal_sign = match.group(3)
            number = int(match.group(4))
            if operator == '>':
                shape_min[rank] = np.maximum(shape_min[rank], number + (1 if equal_sign == '=' else 0))
            elif operator == '<':
                shape_max[rank] = np.minimum(shape_max[rank], number - (1 if equal_sign == '=' else 0))
            elif operator == '=':
                shape_min[rank] = np.maximum(shape_min[rank], number)
                shape_max[rank] = np.minimum(shape_max[rank], number)
        return shape_max, shape_min, SuitPermuter(True)
    
    def parse_hcp(token:str, hand: int, current_constraint: HCPConstraint | types.NoneType):
        if current_constraint is None: current_constraint = HCPConstraint()
        match = re.match(r'^(\d{1,2})([\+\-])(\d{0,2})(hcp|p)$', token)
        number = int(match.group(1))
        operator = match.group(2)
        limit = match.group(3)
        hcp_min = current_constraint.hcp_min.copy(); hcp_max = current_constraint.hcp_max.copy()
        if operator == '+':
            hcp_min[hand] = max(hcp_min[hand], number)
        elif operator == '-':
            if limit == '': hcp_max[hand] = min(hcp_max[hand], number)
            else: 
                hcp_min[hand] = max(hcp_min[hand], number)
                hcp_max[hand] = min(hcp_max[hand], int(limit))
        return HCPConstraint(hcp_max, hcp_min)
    
    def parse_given_cards(hand: int, given_hand: Hand | types.NoneType , *tokens:str):
        suit_map = {'s':0, 'h':1, 'd':2, 'c':3}
        rank_map = {'a':0, 'k':1, 'q':2, 'j':3, 't':4, '9':5, '8':6, '7':7, '6':8, '5':9, '4':10, '3':11, '2':12, '10':4}
        given_hand = Hand() if given_hand is None else given_hand
        for token in tokens:
            if match := re.match(r'^(s|h|d|c)(a|k|q|j|t|10|9|8|7|6|5|4|3|2)$', token):
                suit = suit_map[match.group(1)]; rank = rank_map[match.group(2)]
            elif match := re.match(r'^(a|k|q|j|t|10|9|8|7|6|5|4|3|2)(s|h|d|c)$', token):
                suit = suit_map[match.group(2)]; rank = rank_map[match.group(1)]
            else: raise ValueError(f'Unrecognized card token: {token}.')
            given_hand._dealcard(hand, suit, rank)
        return given_hand

    # initialise constraints
    shape_constraints = []
    hcp_constraint = HCPConstraint()
    given_hand = Hand()
    for idx, constraits in enumerate(hand_conditions.values()):
        shape_tokens = []
        shape_type = None
        hcp_tokens = []
        given_tokens = []
        i = 0
        while i < len(constraits):
            token = constraits[i]
            if token == 'hascard':
                given_tokens = constraits[i+1:]
                break
            token_type = classify_consition(token)
            if token_type.startswith('shape'):
                if shape_type is None: shape_type = token_type
                elif shape_type != token_type:
                    raise ValueError('Different types of shape constraints cannot be combined.')
                shape_tokens.append(token)
            elif token_type == 'hcp':
                hcp_tokens.append(token)
            i += 1
        
        # parse shape constraints
        if shape_type == 'shape_suit':
            s_max, s_min, permute_suits = parse_shape_suit(*shape_tokens)
            shape_constraints.append(SingleHandShapeConstraint(idx, s_max, s_min, permute_suits))
        elif shape_type == 'shape_all':
            s_max, s_min, permute_suits = parse_shape_all(''.join(shape_tokens))
            shape_constraints.append(SingleHandShapeConstraint(idx, s_max, s_min, permute_suits))
        elif shape_type == 'shape_ranked':
            s_max, s_min, permute_suits = parse_shape_ranked(*shape_tokens)
            shape_constraints.append(SingleHandShapeConstraint(idx, s_max, s_min, permute_suits))
        
        # parse hcp constraints
        for token in hcp_tokens:
            hcp_constraint = parse_hcp(token, idx, hcp_constraint)
        
        # parse given cards
        if len(given_tokens) > 0: given_hand = parse_given_cards(idx, given_hand, *given_tokens)
    shape_constraint = combine_single_hand_shape_constraints(*shape_constraints) if len(shape_constraints) > 0 else None
    return Constraint(shape_constraint, hcp_constraint), given_hand

class Dealer():
    def __init__(
            self, 
            constraint: Constraint | ShapeConstraint | SingleHandShapeConstraint | HCPConstraint | types.NoneType = None,
            given: np.ndarray | Hand | types.NoneType = None,
            rng: np.random.Generator | int | bool | types.NoneType = False,
            hcp_exact: bool = False,
            **kwargs
        ):
        if isinstance(constraint, Constraint): self.constraint = constraint
        elif isinstance(constraint, (ShapeConstraint, SingleHandShapeConstraint)): 
            self.constraint = Constraint(shape_constraint = constraint, **kwargs)
        elif isinstance(constraint, HCPConstraint): 
            self.constraint = Constraint(hcp_constraint = constraint, **kwargs)
        else: self.constraint = Constraint(**kwargs)

        if given is not None: self.partial_deal = Hand(given = given)
        else: self.partial_deal = Hand()

        self.hcp_exact = hcp_exact

        if isinstance(rng, np.random.Generator): self.rng = rng
        elif isinstance(rng, int): self.rng = np.random.default_rng(rng)
        elif rng is None: self.rng = np.random.default_rng() # uses the same seed throughout
        elif rng is False: self.rng = np.random # randomly generates a seed every time

    # deals all remaining cards from a partial deal
    def _random(self, hand:Hand | types.NoneType = None):
        if hand is None: hand = self.partial_deal.copy()
        available_cards = np.stack(np.where(hand._to_deal())).T # (suit, rank)
        self.rng.shuffle(available_cards)
        remaining_cards = np.ones(4)*13 - hand.array_rep.sum(axis=(1,2))
        for idx, card in enumerate(available_cards):
            if idx < remaining_cards[0]: hand.array_rep[0, card[0], card[1]] = True
            elif idx < remaining_cards[0] + remaining_cards[1]: hand.array_rep[1, card[0], card[1]] = True
            elif idx < remaining_cards[0:3].sum(): hand.array_rep[2, card[0], card[1]] = True
            else: hand.array_rep[3, card[0], card[1]] = True
        return hand

    # deals all hands according to a single-hand constraint
    def _deal_shape_single_hand_constraint(
            self, 
            hand: Hand | types.NoneType = None,
            constraint: Constraint | SingleHandShapeConstraint | types.NoneType = None, 
        ):

        if hand is None: hand = self.partial_deal.copy()
        if constraint is None: constraint = self.constraint.shape_constraint
        if constraint is None or constraint.no_shape_cond: return self._random(hand)

        # first satisfy min shape constraints
        for suit in range(4):
            if constraint.suit_min[suit] == 0: continue
            available_cards = np.where(~hand.array_rep[:,suit,:].any(axis=0))[0]
            if len(available_cards) < constraint.suit_min[suit]:
                raise ValueError('Given hand conflicts with shape constraints.')
            chosen_cards = self.rng.choice(
                available_cards,
                size=constraint.suit_min[suit] - hand.array_rep[constraint.hand, suit,:].sum(), 
                replace=False)
            hand.array_rep[constraint.hand, suit, chosen_cards] = True

        # then satisfy max shape constraints by requiring some cards to be dealt to other hands
        # hence, only the first n_avail_to_restricted_hand cards in each suit can be dealt to the constrained hand
        avail_to_restricted_hand = []
        for suit in range(4):
            available_cards = np.where(~hand.array_rep[:,suit,:].any(axis=0))[0]
            if constraint.suit_max[suit] == 13: 
                avail_to_restricted_hand.append(np.stack((np.ones(len(available_cards))*suit,
                                                          available_cards)).T)
            self.rng.shuffle(available_cards)
            n_avail_to_restricted_hand = constraint.suit_max[suit] - hand.array_rep[constraint.hand, suit,:].sum()
            avail_to_restricted_hand.append(np.stack((np.ones(n_avail_to_restricted_hand)*suit, 
                                                      available_cards[:n_avail_to_restricted_hand])).T)
        
        # fill the constrained hand to 13 cards, using above available cards
        avail_to_restricted_hand = np.concatenate(avail_to_restricted_hand, axis=0).astype(int)
        self.rng.shuffle(avail_to_restricted_hand)
        n_to_restricted_hand = int(13 - hand.array_rep[constraint.hand,:,:].sum())
        for card in avail_to_restricted_hand[:n_to_restricted_hand]:
            hand.array_rep[constraint.hand, card[0], card[1]] = True

        # then deal all remaining cards
        hand = self._random(hand)
        return hand
    
    # deals all hands if all max shape constraints are voids (or filled by given cards)
    def _deal_shape_multi_void_constraint(
        self,
        hand: Hand | types.NoneType = None,
        constraint: Constraint | ShapeConstraint | types.NoneType = None,
    ):
        if hand is None: hand = self.partial_deal.copy()
        if constraint is None: constraint = self.constraint.shape_constraint
        if constraint is None or constraint.no_shape_cond: return self._random(hand)

        # need to ensure min shape constraints are satisfied
        assert (hand._get_shape() >= constraint.shape_min).all(), 'Need to satisfy min shape constraints first.'
        
        # need to ensure max shape constraints are voids
        assert ((hand._get_shape() == constraint.shape_max) | (constraint.shape_max == 13)).all(), \
            'Max shape constraints must be voids.'
        
        # deal cards in all suits with a void constraint first
        void_suits = np.where((hand._get_shape() == constraint.shape_max).any(axis = 0))
        available_cards = np.stack(np.where(hand._to_deal())).T # (suit, rank)
        available_cards = available_cards[np.isin(available_cards[:,0],void_suits),:]
        self.rng.shuffle(available_cards)
        for card in available_cards:
            weights_array = hand._to_deal()[np.newaxis,:,:].astype(int).repeat(4, axis=0)
            weights_array[hand._get_shape() == constraint.shape_max] = 0
            remaining_cards = hand._to_deal().sum(axis = 1) # empty slots in each hand
            weights_array = weights_array / weights_array.sum(axis=(1,2), keepdims=True)
            weights_array *= remaining_cards[:,np.newaxis,np.newaxis]
            hand.array_rep[self.rng.choice(4, p = weights_array[:,*card] / weights_array[:,*card].sum()), 
                           card[0], card[1]] = True

        # then deal all remaining suits
        return self._random(hand)
    
    def _deal_shape_multi_max_constraint(
        self,
        hand: Hand | types.NoneType = None,
        constraint: Constraint | ShapeConstraint | types.NoneType = None,
    ):
        if hand is None: hand = self.partial_deal.copy()
        if constraint is None: constraint = self.constraint.shape_constraint
        if constraint is None or constraint.no_shape_cond: return self._random(hand)

        # need to ensure min shape constraints are satisfied
        current_shape = hand._get_shape()
        assert (current_shape >= constraint.shape_min).all(), 'Need to satisfy min shape constraints first.'
        
        # need to ensure max shape constraints are voids
        if ((current_shape == constraint.shape_max) | (constraint.shape_max == current_shape)).all():
            return self._deal_shape_multi_void_constraint(hand, constraint)
        
        # use combinatorials to directly sample how many cards to deal to each constrained hand/suit
        # formula:
        #     remaining cards in suit x -> choose cards to deal to each hand constrained in suit x
        #     repeat for all constrained suits
        #     from unconstrained suits, choose cards to fill each hand to 13 cards
        #     hence, suppose constraints are applied to (hand 0, suit 0), (hand 1, suit 0), (hand 0, suit 2) = 
        #         s[0]! / i[0,0]!i[1,0]!(s[0]-i[0,0]-i[1,0])! *
        #         s[1]! / i[0,2]!(s[1]-i[0,2]) *
        #         (s[2]+s[3])! / (h[0] - i[0,0] - i[0,2])!(h[1] - i[1,0])
        #     where s[x] = cards left in suit x, h[x] = cards left in hand x, i[x,y] = cards to deal to hand x in suit y
        #     ignoring irrelevant terms

        cards_by_suit = 13 - current_shape.sum(axis=0) # cards left to deal in each suit
        cards_by_hand = 13 - current_shape.sum(axis=1) # cards left to deal in each hand
        constrain_4th_suit = np.ones(4)*13; constrain_4th_hand = np.ones(4)*13 # if 3 hands/suits are constrained, the 4th is fixed
        constrained_idx = np.stack(np.where((current_shape <= constraint.shape_max) & (constraint.shape_max != 13))).T # (hand, suit)
        for s in np.unique(constrained_idx[:,1]):
            suit_idx = np.where(constrained_idx[:,1] == s)[0] # indices of constraints in this suit
            if len(suit_idx) == 4:
                constrain_4th_hand[s] = constrained_idx[suit_idx[3],0]
                constrained_idx = np.delete(constrained_idx, suit_idx[3], axis=0)
        for h in np.unique(constrained_idx[:,0]):
            hand_idx = np.where(constrained_idx[:,0] == h)[0]
            if len(hand_idx) == 4:
                constrain_4th_suit[h] = constrained_idx[hand_idx[3],1]
                constrained_idx = np.delete(constrained_idx, hand_idx[3], axis=0)
        n_constraints = constrained_idx.shape[0]

        # determine number of possibilities
        n_possibilities = 1
        for idx in constrained_idx:
            max_cards = (constraint.shape_max[idx[0], idx[1]] - current_shape[idx[0], idx[1]]).astype(int)
            n_possibilities *= (max_cards + 1)
        if n_possibilities > 50000:
            # deal by iterating random deals until one fits the constraints
            max_attempts = 1000
            for _ in range(max_attempts):
                trial_hand = self._random(hand.copy())
                trial_shape = trial_hand._get_shape()
                if (trial_shape <= constraint.shape_max).all():
                    return trial_hand
            raise ValueError('Could not deal a hand satisfying the shape constraints after '+str(max_attempts)+
                             ' attempts. Consider using looser constraints.')

        # otherwise directly simulate the shape in the constrained hands/suits
        constrained_ranges = []
        weight_array = 1
        for axis, idx in enumerate(constrained_idx):
            max_cards = (constraint.shape_max[idx[0], idx[1]] - current_shape[idx[0], idx[1]]).astype(int)
            reshape_axis = np.ones(n_constraints).astype(int); reshape_axis[axis] = -1
            card_range = np.arange(0, max_cards+1).reshape(reshape_axis)
            constrained_ranges.append(card_range)
            weight_array = weight_array * sp.factorial(card_range)
        
        for s in np.unique(constrained_idx[:,1]):
            suit_idx = np.where(constrained_idx[:,1] == s)[0] # indices of constraints in this suit
            tmp = sum([constrained_ranges[i] for i in suit_idx])
            tmp[tmp + constrain_4th_hand[s] < cards_by_suit[s]] = 13
            weight_array = weight_array * sp.factorial( cards_by_suit[s] - tmp ) # ineligible hands are assigned weight 0
        for h in np.unique(constrained_idx[:,0]):
            hand_idx = np.where(constrained_idx[:,0] == h)[0]
            tmp = sum([constrained_ranges[i] for i in hand_idx])
            tmp[tmp + constrain_4th_suit[s] < cards_by_hand[h]] = 13
            weight_array = weight_array * sp.factorial( cards_by_hand[h] - tmp ) # NB not np.sum

        weight_array[weight_array == 0] = np.inf # combinations that violate the 13-card total will be discarded
        weight_array **= -1
        weight_array[np.isnan(weight_array)] = 0
        weight_array /= weight_array.sum()
        cards_to_deal = np.unravel_index(self.rng.choice(weight_array.size, p = weight_array.flatten()), weight_array.shape)

        # deal the chosen number of cards to each constrained hand/suit
        for i, idx in enumerate(constrained_idx):
            available_cards = np.where(hand._to_deal()[idx[1],:])[0]
            if len(available_cards) < cards_to_deal[i]:
                raise ValueError('Given hand conflicts with shape constraints.')
            chosen_cards = self.rng.choice(
                available_cards,
                size=cards_to_deal[i], 
                replace=False)
            hand.array_rep[idx[0], idx[1], chosen_cards] = True

        # then deal all remaining cards using void constraints
        new_constraint = constraint.copy()
        new_constraint.shape_max[3,constrain_4th_hand != 13] = 13
        new_constraint.shape_max[constrain_4th_suit != 13,3] = 13
        new_shape = hand._get_shape()
        for idx in constrained_idx:
            new_constraint.shape_max[idx[0], idx[1]] = new_shape[idx[0], idx[1]] 
            # need to enforce that no cards are dealt to constrained hands/suits
        return self._deal_shape_multi_void_constraint(hand, new_constraint)
        
    def _deal_shape_multi_hand_constraint(
        self,
        hand: Hand | types.NoneType = None,
        constraint: Constraint | ShapeConstraint | types.NoneType = None,
    ):
        if hand is None: hand = self.partial_deal.copy()
        if constraint is None: constraint = self.constraint.shape_constraint
        if constraint is None or constraint.no_shape_cond: return self._random(hand)

        # first satisfy min shape constraints
        for suit in range(4):
            for idx in range(4):
                if constraint.shape_min[idx,suit] == 0: continue
                available_cards = np.where(~hand.fixed[suit,:])[0]
                if len(available_cards) < constraint.shape_min[idx,suit]:
                    raise ValueError('Given hand conflicts with shape constraints.')
                chosen_cards = self.rng.choice(
                    available_cards,
                    size=constraint.shape_min[idx,suit] - hand.array_rep[idx, suit,:].sum(), 
                    replace=False)
                hand.array_rep[idx, suit, chosen_cards] = True
        
        # if all max constraints are in the same hand, same as above
        hands_with_max_cond = []
        for idx in range(4):
            if (constraint.shape_max[idx,:] != 13).any(): hands_with_max_cond.append(idx)
        
        if len(hands_with_max_cond) == 0:
            return self._random(hand)
        elif len(hands_with_max_cond) == 1:
            return self._deal_shape_single_hand_constraint(hand, SingleHandShapeConstraint(
                hands_with_max_cond[0], 
                constraint.shape_max[hands_with_max_cond[0],:], 
                constraint.shape_min[hands_with_max_cond[0],:],
                permute_suits = constraint.permute_suits
            ))
        else:
            return self._deal_shape_multi_max_constraint(hand, constraint)
        
    def _deal_hcp_fixed_shape(self, hand:Hand, constraint: HCPConstraint | types.NoneType = None, hcp_exact = False):
        if not hand._dealt(): raise ValueError('Hand must be fully dealt before calling _deal_hcp_fixed_shape.')
        if constraint is None: constraint = self.constraint.hcp_constraint
        if constraint is None or constraint.no_hcp_cond: return hand

        #region dealing with heuristic weights
        shape = hand._get_shape()
        hand.reset()
        fixed_shape = hand._get_shape()
        cards_to_deal = shape - fixed_shape

        if hcp_exact:
            from weights import heuristic_rank_weight
            hcp_weight = heuristic_rank_weight
        else:
            from weights import uniform_weight
            hcp_weight = uniform_weight
        weights = []
        for idx in range(4):
            if constraint.hcp_max[idx] == 37 and constraint.hcp_min[idx] == 0: weights.append(None); continue
            weights.append(hcp_weight(constraint.hcp_max[idx], constraint.hcp_min[idx]))
        
        for _ in range(10000): # try 10,000 times to get a valid deal
            for idx in range(4):
                if weights[idx] is None: continue
                for suit in range(4):
                    if cards_to_deal[idx, suit] == 0: continue
                    suit_available = np.where(hand.array_rep[:,suit,:].sum(axis=0) == 0)[0]
                    weight_available = weights[idx][suit_available]
                    hand.array_rep[idx, suit, self.rng.choice(
                        suit_available, 
                        size = cards_to_deal[idx, suit], 
                        replace = False, 
                        p = weight_available / weight_available.sum()
                    )] = True
                hand = self._random(hand)
            if constraint.check(hand): return hand
            hand.reset()
        raise ValueError('Failed to deal a hand satisfying the HCP constraints.')
        #endregion
    
    def deal(self, n:int = 50000):
        if n <= 0: raise ValueError('n must be a positive integer.')
        start = perf_counter()
        deals = []
        for _ in tqdm(range(n), desc = 'Dealing hands'):
            hand = self.partial_deal.copy()
            hand = self.constraint.permute_suits.permute(hand)
            if isinstance(self.constraint.shape_constraint, SingleHandShapeConstraint):
                hand = self._deal_shape_single_hand_constraint(hand, self.constraint.shape_constraint)
            else:
                hand = self._deal_shape_multi_hand_constraint(hand, self.constraint.shape_constraint)
            hand = self._deal_hcp_fixed_shape(hand, self.constraint.hcp_constraint, self.hcp_exact)
            deals.append(hand)
        end = perf_counter()
        print(f'Dealt {n} hands in {end-start:.2f} seconds ({n/(end-start):.2f} hands/second).')
        return deals  

class Simulator():
    def __init__(self, constraint = None, given = None, rng = None, dirname = None, **kwargs):
        if dirname != None: self.load(dirname, constraint = constraint, given = given, rng = rng, **kwargs); return
        self.dealer = Dealer(constraint, given, rng = rng, **kwargs)
        self.deals = []
        self.n_deals = 0
        self.dds = None

    def deal(self, n:int = 50000):
        self.deals += self.dealer.deal(n)
        self.n_deals += n
        return self

    def check(self, constraint):
        passed = [constraint.check(deal) for deal in self.deals]
        print(f'{sum(passed)}/{self.n_deals}({sum(passed)/self.n_deals:.2f}) deals passed the constraint check.')
        return sum(passed)/self.n_deals

    def subset(self, constraint):
        new_simulator = Simulator()
        new_simulator.dealer = self.dealer
        new_simulator.deals = [deal for deal in self.deals if constraint.check(deal)]
        new_simulator.n_deals = len(new_simulator.deals)
        return new_simulator

    def expectation(self):
        return np.mean([deal.array_rep for deal in self.deals], axis=0)
    
    def solve_dds(self):
        n_to_solve = min(self.n_deals, 5000) # deal limit depends on computer speed  
        # 5000 boards ~ 3 minutes
        with Pool(min(cpu_count(),8)) as p:
            results = list(tqdm(p.imap(
                solve_one_board,
                [x._get_pbn() for x in self.deals[:n_to_solve]],
                chunksize = 50,
            ), desc = 'Solving double dummy', total = n_to_solve))
        for idx, res in enumerate(results):
            self.deals[idx].dds = res
        results = np.array(results) # board id, hand, contract (NT, S, H, D, C)
        self.dds = results
        return results
    
    def save(self, dirname: str = None):
        if dirname is None: dirname = input('Please specify a directory name to save simulated deals: ')
        os.makedirs(dirname, exist_ok = True)
        
        if os.path.isfile(f'{dirname}/deals.pkl'):
            overwrite = input('deals.pkl already exists in the specified directory. Overwrite? (Y/n): ')
            if overwrite.lower() == 'n':
                print('Aborting.'); return
        with open(f'{dirname}/dealer.pkl', 'wb') as f: pickle.dump(self.dealer, f)

        # optimise memory usage in saving deals
        deals = np.stack([deal.array_rep for deal in self.deals], axis=0).astype(bool).reshape(-1)
        deals = np.packbits(deals) # bool array use 8 bits per entry, pack to uint8
        np.save(f'{dirname}/deals.npy', deals)

        # save dds results if available
        if self.dds is not None:
            np.save(f'{dirname}/dds.npy', self.dds)
        print(f'Saved simulated deals to directory: {dirname}')

    def load(self, dirname: str, **kwargs):
        os.makedirs(dirname, exist_ok = True)
        if os.path.isfile(f'{dirname}/dealer.pkl'):
            warnings.warn('Loading dealer.pkl from the specified directory. Ignoring other parameters for the dealer.')
            with open(f'{dirname}/dealer.pkl', 'rb') as f: self.dealer = pickle.load(f)
        else:
            warnings.warn('dealer.pkl not found in the specified directory. Specifying dealer with other parameters.')
            self.dealer = Dealer(**kwargs)
            
        if not os.path.isfile(f'{dirname}/deals.npy'):
            warnings.warn('deals.npy not found in the specified directory. No deals loaded.')
            self.deals = []
        else:
            deals = np.load(f'{dirname}/deals.npy')
            deals = np.unpackbits(deals).reshape((-1,4,4,13)).astype(bool)
            self.deals = [Hand(given = deal) for deal in deals]
        self.n_deals = len(self.deals)
        dds_path = f'{dirname}/dds.npy'
        if os.path.isfile(dds_path):
            self.dds = np.load(dds_path)
            for idx, deal in enumerate(self.deals):
                deal.dds = self.dds[idx]
        else: self.dds = None
        print(f'Loaded {self.n_deals} simulated deals from directory: {dirname}')
        return self

# test usage
if __name__ == '__main__':
    # a = parse_string('north 4441 13-18P south 3+S 4+H 14+HCP hascard HA')
    # test = Constraint(SingleHandShapeConstraint(0, 5, 2, False), HCPConstraint([13,37,37,37],[10,0,0,0])) # 1NT
    # test = Constraint(None, HCPConstraint(hcp_min = [16,0,0,0])) # 1C
    # test = Constraint()
    # test = Constraint(ShapeConstraint(
    #     shape_max = np.array([[5, 5, 5, 5],
    #                           [5, 5, 13, 13],
    #                           [5, 5, 13, 13],
    #                           [5, 13, 13, 13]]),
    #     shape_min = np.array([[0, 0, 0, 0],
    #                           [0, 0, 0, 0],
    #                           [0, 0, 0, 0],
    #                           [0, 0, 0, 0]]),
    #     permute_suits = False
    # ), HCPConstraint([37,37,37,37],[0,0,0,0]))
    # test = Constraint(ShapeConstraint(
    #     shape_max = np.ones((4,4)) * 5,
    #     shape_min = np.array([[0, 0, 0, 0],
    #                           [0, 0, 0, 0],
    #                           [0, 0, 0, 0],
    #                           [0, 0, 0, 0]]),
    #     permute_suits = False
    # ), HCPConstraint([37,37,37,37],[0,0,0,0]))
    # dealer = Simulator(constraint = test).deal(5000)
    # dealer = Simulator().deal(int(1e8))
    # dealer.save('example')
    import argparse
    parser = argparse.ArgumentParser(description='Deal bridge hands with constraints.')
    parser.add_argument('--constraint', type=str, default = [], nargs = '*', help='Constraint string specifying the dealing constraints.')
    parser.add_argument('-n', '--n_deals', type=int, default = 1000, help='Number of deals to simulate.')
    parser.add_argument('-o','--out', type =str, default = 'simulated_deals', help='Output directory to save simulated deals.')
    args = parser.parse_args()
    constraint_strings = ' '.join(args.constraint)
    constraint, given = parse_string(constraint_strings)
    dealer = Simulator(constraint = constraint, given = given, rng = None, dirname = args.out).deal(args.n_deals)
    dealer.save(args.out)
    # dealer.solve_dds()