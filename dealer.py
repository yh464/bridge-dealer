import os
import numpy as np
import pandas as pd
import dds
import ctypes

def shuffle():
    rng = np.random.Generator(np.random.PCG64()) 
    # reset RNG from secrets.randbits so that boards are cryptographically secure
    h = rng.permutation(52)
    hands = [np.sort(h[0:13]),np.sort(h[13:26]),np.sort(h[26:39]),np.sort(h[39:52])]
    card_readable = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    dstr = 'N:'
    for i in range(4):
        h = hands[i]
        h = h[::-1]
        suit = 0 # spades
        for j in range(13):
            # We first add the suit separation points
            if suit == 0 and h[j]< 39: suit += 1; dstr += '.'
            if suit == 1 and h[j]< 26: suit += 1; dstr += '.'
            if suit == 2 and h[j]< 13: suit += 1; dstr += '.'
            # We then add the cards
            dstr += card_readable[h[j] % 13]
        if suit < 3: dstr += (3-suit) * '.'
        if i < 3: dstr += ' '
    return dstr

class board():
    def __init__(self, idx, pbn = None, dlr = None, vul = None):
        if type(idx) == str: idx = idx.replace('"','')
        idx = int(idx)
        self.idx = idx
        
        if dlr == None: self.dlr = ['N','E','S','W'][(self.idx-1)%4]
        else: self.dlr = str(dlr).replace('"','')
        
        if vul == None: self.vul = ['None','NS','EW','Both'][int(self.idx-1+np.floor((self.idx-1)/4))%4]
        else: self.vul = str(vul).replace('"','')
        
        if pbn == None:
            self.pbn = None
            self.tabular = None
            self.info = None
            self.dd = None
        else:
            pbn = str(pbn).replace('"','')
            self.pbn = self._normalise_pbn(pbn)
            self.tabular = self._pbn_to_tabular()
            self.info = self._info_table()
            self.dd = self._dds_table()
        
    def _normalise_pbn(self, pbn):
        spl = pbn.split(':')
        start = spl[0]
        if start =='N': return pbn
        deal = spl[1].split()
        if start =='E':
            deal = [deal[-1]] + deal[:-1]
        elif start == 'S':
            deal = deal[2:] + deal[:2]
        elif start == 'W':
            deal = deal[1:] + [deal[0]]
        else: raise ValueError('PBN format deal not properly defined')
        
        return 'N:'+' '.join(deal)
    
    def _pbn_to_tabular(self):
        hands = self.pbn[2:].split()
        outdf = pd.DataFrame(index = ['N','E','S','W'], 
                             columns = ['S','H','D','C'],
                             data = '')
        for i, x in zip(range(4),hands):
            outdf.iloc[i,:] = x.split('.')
        return outdf
    
    def _info_table(self):
        outdf = pd.DataFrame(index = ['N','E','S','W'],
                             columns = ['HCP','Balanced','Singleton','Void','Long_7','Long_8'],
                             data = 0)
        hands = self.pbn[2:].split()
        for i, x in zip(range(4), hands):
            outdf.iloc[i,0] = 4*x.count('A') + 3*x.count('K') + 2*x.count('Q') + x.count('J')
            suits = x.split('.')
            shape = [len(s) for s in suits]
            shape.sort()
            outdf.iloc[i,2] = sum([s == 1 for s in shape])
            outdf.iloc[i,3] = sum([s == 0 for s in shape])
            outdf.iloc[i,1] = 1 if shape in [[3,3,3,4],[2,3,4,4],[2,3,3,5]] else 0
            outdf.iloc[i,4] = sum([len(s) >= 7 for s in suits])
            outdf.iloc[i,5] = sum([len(s) >= 8 for s in suits])
        return outdf
    
    def _dds_table(self):
        solver = dds.ddTableDealPBN()
        resTable = ctypes.pointer(dds.ddTableResults())
        dds.SetMaxThreads(0)
        ctypes.create_string_buffer(80)
        solver.cards = bytes(self.pbn, 'utf-8')
        dds.CalcDDtablePBN(solver, resTable)
        res = resTable.contents.resTable
        S=[res[0][k] for k in [0,1,2,3]]
        H=[res[0][4],res[1][0],res[1][1],res[1][2]]
        D=[res[1][3],res[1][4],res[2][0],res[2][1]]
        C=[res[2][2],res[2][3],res[2][4],res[3][0]]
        N=[res[3][k] for k in [1,2,3,4]]
        return pd.DataFrame(dict(C = C, D = D, H = H, S = S, NT = N),
                            index = ['N','E','S','W'])
    
    def _hand_to_latex(self,hand):
        # the hand parameter should be a row of the tabular format
        latex_cmd = r'\makecell[l]{$\spadesuit$ SPADES\\$\heartsuit$ HEARTS\\$\diamondsuit$ DIAMONDS\\$\clubsuit$ CLUBS}'
        latex_cmd = latex_cmd.replace('SPADES', hand['S']).replace('HEARTS', hand['H']).replace(
            'DIAMONDS',hand['D']).replace('CLUBS',hand['C'])
        return latex_cmd
    
    def _dds_to_latex(self, contract = False):
        latex_cmd = [r'\footnotesize{\begin{tabular}{cccccc}',
                     r'&$\clubsuit$&$\diamondsuit$&$\heartsuit$&$\spadesuit$&N\\']
        for x in ['N','S','E','W']:
            if contract:
                tmp = (self.dd.loc[x,:] - 6).astype(str).replace(
                    [str(i) for i in range(-6,1)], '-').tolist()
            else: tmp = self.dd.loc[x,:].astype(str).tolist()
            tmp = [x] + tmp
            tmp = '&'.join(tmp) + r'\\'
            latex_cmd.append(tmp)
        latex_cmd.append(r'\end{tabular}}')
        return '\n'.join(latex_cmd)
    
    def set_hand(self, pbn):
        pbn = str(pbn).replace('"','')
        self.pbn = self._normalise_pbn(pbn)
        self.tabular = self._pbn_to_tabular()
        self.info = self._info_table()
        self.dd = self._dds_table()
    
    def shuffle(self):
        self.set_hand(shuffle())
        
    def to_pbn(self):
        pbn_string = [
            f'[Board "{self.idx}"]',
             f'[Dealer "{self.dlr}"]',
             f'[Vulnerable "{self.vul}"]',
             f'[Deal "{self.pbn}"]'
            ]
        return '\n'.join(pbn_string)
    
    def to_latex(self, contract):
        latex_cmd = [
            # first row: board number, north hand, information
            r'\begin{tabular}{@{}m{9mm} m{22mm} m{15mm}@{}}',
            r'\makecell[c]{\huge IDX} &'.replace('IDX',str(self.idx)),
            self._hand_to_latex(self.tabular.loc['N',:]),
            r'& \makecell[tl]{Dlr: DEALER \\ Vul: VULNER}\\'.replace(
                'DEALER',self.dlr).replace('VULNER',self.vul),
            r'\end{tabular}',
            # second row: west and east hands
            r'\begin{tabular}{@{}m{22mm} m{22mm}@{}}',
            self._hand_to_latex(self.tabular.loc['W',:])+r'&',
            self._hand_to_latex(self.tabular.loc['E',:]),
            r'\end{tabular}',
            # last row: HCPs, south hand, double 
            r'\begin{tabular}{@{}m{9mm} m{20mm} m{17mm}@{}}',
            r'\makecell[c]{N\\W E\\S} &'.replace(
                'N',str(self.info.loc['N','HCP'])).replace(
                'W',str(self.info.loc['W','HCP'])).replace(
                'E',str(self.info.loc['E','HCP'])).replace(
                'S',str(self.info.loc['S','HCP'])),
            self._hand_to_latex(self.tabular.loc['S',:])+r'&',
            self._dds_to_latex(contract),
            r'\end{tabular}'
            ]
        return '\n'.join(latex_cmd)

class session():
    def __init__(self, prefix, n_boards = 24, force = False, **kwargs):
        if os.path.dirname(prefix) == '': prefix = './' + prefix
        self.prefix = prefix
        if os.path.isfile(f'{prefix}.pbn') and not force:
            self._parse_pbn()
        else:
            self.boards = [None] * n_boards
            self.n_boards = n_boards
            self.recap = None
            self.generate(**kwargs)
            
    def _parse_pbn(self):
        pbn = open(f'{self.prefix}.pbn').read().splitlines()
        boards = []
        idx = 1
        dlr = None
        vul = None
        for x in pbn:
            if x[:6] == '[Board':
                idx = x.replace('[Board ','').replace(']','').replace('"','')
                idx = int(idx)
            elif x[:6] == '[Deale':
                dlr = x.replace('[Dealer ','').replace(']','').replace('"','')
            elif x[:6] == '[Vulne':
                vul = x.replace('[Vulnerable ','').replace(']','').replace('"','')
            elif x[:6] == '[Deal ':
                boards.append(board(idx, x[6:-1], dlr, vul))
                idx += 1
                dlr = None
                vul = None
        self.boards = boards
        self.n_boards = len(boards)
        self.recap = self._calc_recap()
        self.latex = self.to_latex()
        
    def _calc_recap(self):
        outdf = sum([b.info for b in self.boards])
        outdf['HCP'] /= self.n_boards
        return outdf
    
    def _recap_to_latex(self):
        latex_cmd = [
            r'\begin{table}[H]',
            r'\renewcommand{\arraystretch}{1.2}',
            r'\centering',
            r'\begin{tabular}{|>{\centering\arraybackslash}m{14mm}|'+
                r'>{\centering\arraybackslash}m{14mm}|>{\centering\arraybackslash}m{14mm}|'+
                r'>{\centering\arraybackslash}m{14mm}|>{\centering\arraybackslash}m{14mm}|'+
                r'>{\centering\arraybackslash}m{14mm}|>{\centering\arraybackslash}m{14mm}|}',
            r'\hline',
            r'&HCP&Balanced&Singles&Voids&7+&8+\\\hline'
            ]
        recap = self.recap.copy()
        recap['HCP'] = recap['HCP'].round(2)
        for x in ['N','S','E','W']:
            tmp = recap.loc[x,'Balanced':].astype(int).tolist()
            tmp = [x, str(recap.loc[x, 'HCP'])] + [str(x) for x in tmp]
            tmp = '&'.join(tmp) + r'\\\hline'
            latex_cmd.append(tmp)
        latex_cmd+= [r'\end{tabular}', r'\end{table}']
        return '\n'.join(latex_cmd)
    
    def to_latex(self, title = None, contract = False, pdflatex = 'c:/texlive/2024/bin/windows/pdflatex.exe'):
        if title == None: title = os.path.basename(self.prefix)
        latex_cmd = [
            r'\documentclass[9pt]{article}',
            r'\usepackage{geometry}',
            r'\usepackage{float}'
            r'\usepackage{array}',
            r'\usepackage{makecell}',
            r'\geometry{a4paper, left = 1cm, right = 1cm, top = 1cm, bottom = 1cm}',
            r'\usepackage[fontsize=9pt]{fontsize}',
            r'\setlength{\tabcolsep}{0pt}',
            r'\renewcommand{\arraystretch}{0.8}',
            r'\renewcommand{\baselinestretch}{0.8}',
            r'\renewcommand{\familydefault}{\sfdefault}',
            r'\begin{document}',
            r'\begin{table}[H]',
            r'\centering',
            r'{\HUGE TITLE\vspace{0.5em}}'.replace('TITLE',title),
            r'\begin{tabular}{|m{47mm}|m{47mm}|m{47mm}|m{47mm}|}',
            r'\hline'
            ]
        idx = 1
        for b in self.boards:
            latex_cmd += [r'\vspace{0.3mm}',b.to_latex(contract)]
            
            if idx == self.n_boards: break
            
            # otherwise, 4 boards per line, 24 boards per page
            if idx % 4 != 0:
                latex_cmd.append(r'&')
            elif idx % 24 != 0:
                latex_cmd.append(r'\\\hline')
            else:
                latex_cmd += [
                    r'\\\hline',
                    r'\end{tabular}',
                    r'\end{table}',
                    r'\clearpage',
                    r'\begin{table}[H]',
                    r'\centering',
                    r'\begin{tabular}{|m{47mm}|m{47mm}|m{47mm}|m{47mm}|}',
                    r'\hline'
                    ]
            idx += 1
        # after finishing all boards, print the recap table
        latex_cmd += [r'\\\hline',r'\end{tabular}',r'\end{table}']
        latex_cmd.append(self._recap_to_latex())
        latex_cmd.append(r'\end{document}')
        latex_cmd = '\n'.join(latex_cmd)
        with open(f'{self.prefix}.tex','w') as f:
            print(latex_cmd, file = f)
            f.close()
        if os.path.isfile(pdflatex):
            os.system(f'{pdflatex} --output-directory='+os.path.dirname(self.prefix)+f' {self.prefix}.tex')
        return latex_cmd
    
    def to_pbn(self, force = False):
        if os.path.isfile(f'{self.prefix}.pbn') and not force: return
        with open(f'{self.prefix}.pbn','w') as f:
            for b in self.boards:
                print(b.to_pbn(), file = f)
                print(file = f)
    
    def generate(self, mode = 'random', n = None, epsilon = 2, bias = 0.2, contract = False):
        if mode not in ['goulash','misfit','balanced','long','slams','game','partscore']:
            mode = 'random'
            
        if n == None:
            n = self.n_boards if mode == 'random' else self.n_boards * 10
        
        def dealscore(pbn, mode, epsilon, bias = 0.2):
            '''bias = prefer single suit over two-suited hand for goulash dealer'''
            rng = np.random.Generator(np.random.PCG64()) # again reset RNG with secrets.randbits
            eps = rng.normal(loc = 0, scale = epsilon)
            
            if mode == 'random': return rng.random() + eps
            hands = pbn[2:].split()
            if mode in ['game','slams']:
                hcps = [4*x.count('A') + 3*x.count('K') + 2*x.count('Q') + x.count('J') for x in hands]
                hcpd = hcps[0]+hcps[2]-hcps[1]-hcps[3]
                if mode == 'slams': return abs(hcpd)+eps
                else: return abs(hcpd - 11) + eps
            if mode in ['goulash','balanced','long','misfit']:
                shape = [[len(y) for y in x.split('.')] for x in hands]
                if mode == 'long':
                    return sum([max(x) for x in shape]) + eps
                else:
                    fit_ns = max([shape[0][x] + shape[2][x] for x in range(4)])
                    fit_ew = max([shape[1][x] + shape[3][x] for x in range(4)])
                    fit_ns = min(fit_ns, 9); fit_ew = min(fit_ew, 9)
                    # fit_ns = -1/(fit_ns - 6.5); fit_ew = -1/(fit_ew - 6.5)
                    fit_ns = 1/(9.1-fit_ns); fit_ew = 1/(9.1-fit_ew)
                    gparams = [(1+bias)*x[-1] + (1-bias)*x[-2] - x[0] for x in shape]
                    if mode == 'goulash': return sum(gparams) + eps
                    elif mode == 'misfit': return sum(gparams) - (fit_ns+fit_ew) + eps
                    else: return -sum(gparams) + eps
        
        all_deals = []
        for _ in range(n):
            pbn = shuffle()
            all_deals.append(pd.DataFrame(dict(
                pbn = [pbn], score = dealscore(pbn, mode = mode, epsilon = epsilon, bias = bias)
                )))
        all_deals = pd.concat(all_deals).sort_values(by = 'score', ascending = False).reset_index(drop = True)
        for o, idx in zip(np.random.permutation(self.n_boards), range(self.n_boards)):
            self.boards[idx] = board(idx+1, all_deals['pbn'].iloc[o])
        all_deals['used'] = False; all_deals.iloc[:self.n_boards,-1] = True
        
        self.diag_table = all_deals
        self.recap = self._calc_recap()
        self.to_pbn(force = True)
        self.latex = self.to_latex(contract = contract)

# ses = session('test', n_boards = 24)
# ses.generate(mode = 'goulash', n = 10000, epsilon = 0)