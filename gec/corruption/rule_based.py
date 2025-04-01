import random
import warnings

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.reinflector import Reinflector

from pyarabic import araby

from typing import List, Tuple

ERROR_CLASSES = set([
    'OH', 'OT', 'OA', 'OW', 'ON', 'OS', 'OG', 'OC', 'OR', 'OD', 'OM', # orthographic errors
    'MI', 'MT', # morphological errors
    'XC', 'XF', 'XG', 'XN', 'XT', 'XM', # syntactic errors
    'SW', 'SF', # semantic errors
    'PC', 'PT', 'PM', # punctuation errors
    'MG', 'SP', # incorrectly merged/split words
    'UC' # No error
])

CAMEL_MORPHOLOGY_FEATURES = {
    'asp': ['c', 'i', 'p'],
    'cas': ['n', 'a', 'g'],
    'form_gen': ['m', 'f'],
    'form_num': ['s', 'd', 'p'],
    'gen': ['m', 'f'],
    'mod': ['i', 'j', 's'],
    'num': ['s', 'd', 'p'],
    'per': ['1st', '2nd', '3rd'],
    'rat': ['n', 'y'],
    'stt': ['c', 'd', 'i'],
    'vox': ['a', 'p'],
    'pos': ['noun', 'noun_prop', 'noun_num', 'noun_quant',
            'adj', 'adj_comp', 'adj_num',
            'adv', 'adv_interrog', 'adv_rel',
            'pron', 'pron_dem', 'pron_exclam', 'pron_interrog', 'pron_rel',
            'verb', 'verb_pseudo',
            'part', 'part_dem', 'part_det', 'part_focus', 'part_fut', 'part_interrog', 'part_neg', 'part_restrict', 'part_verb', 'part_voc',
            'prep',
            'abbrev',
            'punc',
            'conj', 'conj_sub',
            'interj',
            'digit',
            'latin'],
    'prc0': ['0', 'Aa_prondem', 'Al_det', 'AlmA_neg', 'lA_neg', 'mA_neg', 'ma_neg', 'mA_part', 'mA_rel'],
    'prc1': ['0', '<i$_interrog', 'bi_part', 'bi_prep', 'bi_prog', 'Ea_prep', 'EalaY_prep', 'fiy_prep',
             'hA_dem', 'Ha_fut', 'ka_prep', 'la_emph', 'la_prep', 'la_rc', 'libi_prep', 'laHa_emphfut', 'laHa_rcfut',
             'li_jus', 'li_sub', 'li_prep', 'min_prep', 'sa_fut', 'ta_prep', 'wa_part', 'wa_prep', 'wA_voc', 'yA_voc'],
    'prc2': ['0', 'fa_conj', 'fa_conn', 'fa_rc', 'fa_sub', 'wa_conj', 'wa_part', 'wa_sub'],
    'prc3': ['0', '>a_ques'],
    'enc0': ['0', '1s_dobj', '1s_poss', '1s_pron', '1p_dobj', '1p_poss', '1p_pron', '2d_dobj', '2d_poss', '2d_pron', '2p_dobj', '2p_poss',
             '2p_pron', '2fs_dobj', '2fs_poss', '2fs_pron', '2fp_dobj', '2fp_poss', '2fp_pron', '2ms_dobj', '2ms_poss', '2ms_pron', '2mp_dobj', '2mp_poss',
             '2mp_pron', '3d_dobj', '3d_poss', '3d_pron', '3p_dobj', '3p_poss', '3p_pron', '3fs_dobj', '3fs_poss', '3fs_pron', '3fp_dobj', '3fp_poss',
             '3fp_pron', '3ms_dobj', '3ms_poss', '3ms_pron', '3mp_dobj', '3mp_poss', '3mp_pron', 'Ah_voc', 'lA_neg', 'ma_interrog', 'mA_interrog', 'man_interrog',
             'ma_rel', 'mA_rel', 'man_rel', 'ma_sub', 'mA_sub']
}

ARABIC_PREPOSITIONS = [
    'من', 'إلى', 'عن', 'على', 'في'
]

ARABIC_CONJUNCTIONS = [
    'و', 'ثم', 'أم', 'أو', 'لا', 'لكن', 'بل', 'حتى'
]

PUNCTUATIONS = [
    '،', '؛', '؟', '!', '.', '…', '(', ')', '"'
]

class RuleBasedCorruptor:
    def __init__(self, tag_distribution=None, seed=42):
        """
        Args:
            tag_distribution (dict): A prior distribution of morphological tags given in percentages
        """
        self.disambiguator = BERTUnfactoredDisambiguator.pretrained(use_gpu=True)
        
        db = MorphologyDB.builtin_db(flags='r')
        self.reinflector = Reinflector(db)
        self.tag_distribution = tag_distribution
        random.seed(seed)
        
    def _normalize_to_ar_punctuation(self, sentence: str) -> str:
        """
        Normalizes the punctiation to the arabic punctiation
        
        Args:
            sentence (str): input sentence
        Returns:
            str: normalized sentence
        """
        return sentence.replace(',', '،').replace(';', '؛').replace('?', '؟')
        
    def _biased_coin_flip(self, p: float) -> bool:
        """
        True with probability p
        
        Args:
            p (float): probability of True
        Returns:
            bool: True with probability p else False
        """
        return random.random() < p
    
    def _check_word_err_compatibilities(self, word: str, features: dict, error: str) -> bool:
        """
        Checks if the error is possible to induce in the word
        
        Args:
            word (str): input word
            features (dict): morphological features of the word
            error (str): error type
        Returns:
            bool: True if the error is possible to induce in the word, False otherwise
        """
        if error[0] != 'P' and features['pos'] == 'punc':
            return False
        if error[0] != 'O' and error[0] != 'P' and error != 'MG' and error != 'SP':
            for f in ['gen']:
                if f not in features:
                    return False
        match error:
            case 'OH':
                word = word[2:] if features['prc0'] in ['Al_det'] else word
                return ('ا' in word or 'أ' in word or 'إ' in word or word[-1] == 'ى')
            case 'OT':
                return ('ة' == word[-1] or 'ه' == word[-1])
            case 'OA':
                return (word[-1] == 'ي' or word[-1] == 'ى')
            case 'OW':
                return ('وا' in word and len(word) > 2 and
                        features['gen'] == 'm' and features['num'] == 'p' and 
                        features['form_gen'] == 'm' and features['form_num'] == 'p' and
                        features['pos'] == 'verb')
            case 'ON':
                # either tanween with fath or tanween with kasr
                if len(word) < 2:
                    return False
                return (word[-1] == '\u064b' or word[-2] == '\u064b' or
                        word[-1] == '\u064d' or word[-2] == '\u064d')
            case 'OS':
                # either fatha or damma or kasra
                return ('\u064e' in features['diac'] or '\u064f' in features['diac'] or '\u0650' in features['diac'])
            case 'OG':
                if len(word) < 2:
                    return False
                return ('و' in word or 'ا' in word or 'ي' in word)
            case 'OC':
                return len(word) > 1
            case 'OR':
                return True
            case 'OD':
                return True
            case 'OM':
                return True
            case 'MI':
                return False
            case 'MT':
                #return (features['pos'] == 'verb')
                return False
            case 'XC':
                return not (features['cas'] in ['u', 'na'])
            case 'XF':
                return ('noun' in features['pos'])
            case 'XG':
                if 'gen' not in features:
                    return False
                return (not features['gen'] == 'na' and not features['form_gen'] == 'na' and not 'noun' in features['pos'])
            case 'XN':
                if 'num' not in features:
                    return False
                return (not features['num'] == 'na' and not features['form_num'] == 'na' and
                        not features['num'] == 'u' and not features['form_num'] == 'u' and
                        'noun' in features['pos'])
            case 'XT':
                return True
            case 'XM':
                return True
            case 'SW':
                return (features['pos'] == 'prep') or ('prep' in features['prc1'])
            case 'SF':
                return ('conj' in features['pos']) or ('conj' in features['prc2'])
            case 'PC':
                return (features['pos'] == 'punc')
            case 'PT':
                return True
            case 'PM':
                return (features['pos'] == 'punc')
            case 'MG':
                return True
            case 'SP':
                return len(word) > 3
            case 'UC':
                return True
            case _:
                raise ValueError(f'Unknown error type: {error}')
            
    def _reinflect_word(self, word: str, features: dict) -> Tuple[str, dict]:
        """
        Reinflects the word given its morphological features
        
        Args:
            word (str): input word
            features (dict): morphological features of the word
        Returns:
            Tuple[str, dict]: reinflected word and associated features
        """
        try:
            reinflections = self.reinflector.reinflect(word, features)
        except:
            # warnings.warn(f'Error reinflecting word: {word} and features: {features}')
            reinflections = []
        if len(reinflections) < 1:
            # warnings.warn(f'No reinflections found for word: {word} and features: {features}')
            return word, features
        else:
            word = random.choice(reinflections)
            features = word
            word = araby.strip_diacritics(word['diac'])
        return word, features
            
    def make_error_OH(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        # TODO: implement hamza confusion error
        is_det = features['prc0'] in ['Al_det']
        det, word = word[:2] if is_det else '', word[2:] if is_det else word
        forms = set(['ا', 'أ', 'إ'])
        if 'ا' in word:
            word = word.replace('ا', random.choice(list(forms - set(['ا']))))
        elif 'أ' in word:
            word = word.replace('أ', random.choice(list(forms - set(['أ']))))
        elif 'إ' in word:
            word = word.replace('إ', random.choice(list(forms - set(['إ']))))
        elif 'ا' in word and 'أ' in word and 'إ' in word:
            pick = random.choice(['ا', 'أ', 'إ'])
            word = word.replace(pick, random.choice(list(forms - set([pick]))))
        elif 'ا' in word and 'أ':
            pick = random.choice(['ا', 'أ'])
            word = word.replace(pick, random.choice(list(forms - set([pick]))))
        elif 'ا' in word and 'إ':
            pick = random.choice(['ا', 'إ'])
            word = word.replace(pick, random.choice(list(forms - set([pick]))))
        elif 'أ' in word and 'إ':
            pick = random.choice(['أ', 'إ'])
            word = word.replace(pick, random.choice(list(forms - set([pick]))))
        elif word[-1] == 'ى':
            word = word.replace('ى', 'ا')
        sentence[index] = det + word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return det + word, features, error, error_list, sentence, feature_list
    
    def make_error_OT(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if word[-1] == 'ة':
            word = word.replace('ة', 'ه')
        elif word[-1] == 'ه':
            word = word.replace('ه', 'ة')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OA(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if 'ي' in word:
            word = word.replace('ي', 'ى')
        elif 'ى' in word:
            word = word.replace('ى', 'ي')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OW(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        word = word.replace('وا', 'و')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_ON(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if '\u064b' in word:
            # remove tanween with fathatan
            word = word.replace('\u064b', '')
        elif '\u064d' in word:
            # remove tanween with kasratan
            word = word.replace('\u064d', '')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OS(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if '\u064e' in word:
            # remove fatha
            word = word.replace('\u064e', 'ا')
        elif '\u064f' in word:
            # remove damma
            word = word.replace('\u064f', 'و')
        elif '\u0650' in word:
            # remove kasra
            word = word.replace('\u0650', 'ي')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OG(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if 'ا' in word:
            word = word.replace('ا', '\u064e')
        elif 'و' in word:
            word = word.replace('و', '\u064f')
        elif 'ي' in word:
            word = word.replace('ي', '\u0650')
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OC(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        def swap(c, i, j):
            c = list(c)
            c[i], c[j] = c[j], c[i]
            return ''.join(c)
        swap_idx = random.choice([i for i in range(0, len(word) - 1)])
        word = swap(word, swap_idx, swap_idx + 1)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OR(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        replace_idx = random.choice([i for i in range(0, len(word))])
        word = list(word)
        word[replace_idx] = random.choice(araby.LETTERS)
        word = ''.join(word)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OD(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        add_idx = random.choice([i for i in range(0, len(word))])
        if add_idx == 0:
            word = random.choice(araby.LETTERS) + word
        elif add_idx == len(word) - 1:
            word = word + random.choice(araby.LETTERS)
        else:
            word = word[:add_idx] + random.choice(araby.LETTERS) \
                + word[add_idx + 1:]
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_OM(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        rem_idx = random.choice([i for i in range(0, len(word))])
        if rem_idx == 0:
            word = word[rem_idx+1:]
        elif rem_idx == len(word) - 1:
            word = word[:rem_idx]
        else:
            word = word[:rem_idx] + word[rem_idx + 1:]
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_MI(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        # TODO: implement incorrect word morphology error
        raise NotImplementedError
    
    def make_error_MT(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        # TODO: implement incorrect verb tense error
        raise NotImplementedError
    
    def make_error_XC(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        # TODO: implement incorrect case error
        cases = [x for x in CAMEL_MORPHOLOGY_FEATURES['cas'] if x != features['cas']]
        new_features = {'lex': features['lex'], 'cas': random.choice(cases), 'gen': features['gen'], 'form_gen': features['gen']}
        word, features = self._reinflect_word(word, new_features)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XF(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        new_features = {'lex': features['lex'], 'gen': features['gen'], 'form_gen': features['gen']}
        if features['prc0'] != 'Al_det':
            new_features['prc0'] = 'Al_det'
        else:
            new_features['prc0'] = '0'
        word, features = self._reinflect_word(word, new_features)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XG(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        new_gen = random.choice([x for x in CAMEL_MORPHOLOGY_FEATURES['gen'] if x != features['gen']])
        new_features = {'lex': features['lex'], 'gen': new_gen, 'form_gen': new_gen}
        word, features = self._reinflect_word(word, new_features)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XN(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        new_num = random.choice([x for x in CAMEL_MORPHOLOGY_FEATURES['num'] if x != features['num']])
        new_features = {'lex': features['lex'], 'num': new_num, 'form_num': new_num, 'gen': features['gen'], 'form_gen': features['form_gen']}
        word, features = self._reinflect_word(word, new_features)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XT(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        repeat_idx = random.randint(0, len(sentence)-1)
        extra = sentence[repeat_idx]
        extra_features = feature_list[repeat_idx]
        if index == 0:
            sentence = [extra] + sentence
            error_list = ['UC'] + error_list
            feature_list = [extra_features] + feature_list
        elif index == len(sentence) - 1:
            sentence = sentence + [extra]
            error_list = error_list + ['UC']
            feature_list = feature_list + [extra_features]
        else:
            sentence = sentence[:index] + [extra] + sentence[index:]
            error_list = error_list[:index + 1] + ['UC'] + error_list[index + 1:]
            feature_list = feature_list[:index + 1] + [extra_features] + feature_list[index + 1:]
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XM(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if index == 0:
            sentence = sentence[1:]
            error_list = error_list[1:]
            feature_list = feature_list[1:]
        elif index == len(sentence) - 1:
            sentence = sentence[:-1]
            error_list = error_list[:-1]
            feature_list = feature_list[:-1]
        else:
            sentence = sentence[:index] + sentence[index+1:]
            error_list = error_list[:index] + error_list[index+1:]
            feature_list = feature_list[:index] + feature_list[index + 1:]
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return "", {}, error, error_list, sentence, feature_list
    
    def make_error_SW(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if 'prep' in features['prc1']:
            prep_prc = [p for p in CAMEL_MORPHOLOGY_FEATURES['prc1'] if 'prep' in p]
            new_features = {'lex': features['lex'], 'prc1': random.choice(prep_prc), 'gen': features['gen'], 'form_gen': features['gen']}
            word, features = self._reinflect_word(word, new_features)
            sentence[index] = word
            error_list[index] = error
            feature_list[index] = features
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return word, features, error, error_list, sentence, feature_list
        else:
            prep = random.choice(ARABIC_PREPOSITIONS)
            sentence[index] = prep
            error_list[index] = error
            feature_list[index] = {}
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return prep, {}, error, error_list, sentence, feature_list
    
    def make_error_SF(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if 'conj' in features['prc2']:
            # conjunction letter removal
            new_features = {'lex': features['lex'], 'prc2': '0', 'gen': features['gen'], 'form_gen': features['gen']}
            word, features = self._reinflect_word(word, new_features)
            sentence[index] = word
            error_list[index] = error
            feature_list[index] = features
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return word, features, error, error_list, sentence, feature_list
        elif 'conj' in features['pos']:
            if self._biased_coin_flip(0.5):
                # random conjunction word replacement
                conj = random.choice(ARABIC_CONJUNCTIONS)
                sentence[index] = conj
                error_list[index] = error
                feature_list[index] = {}
                assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                    "Error: sentence, error_list, and feature_list must be of the same size"
                return conj, {}, error, error_list, sentence, feature_list
            else:
                # remove conjunction word
                if index == 0:
                    sentence = sentence[1:]
                    error_list = error_list[1:]
                    feature_list = feature_list[1:]
                elif index == len(sentence) - 1:
                    sentence = sentence[:-1]
                    error_list = error_list[:-1]
                    feature_list = feature_list[:-1]
                else:
                    sentence = sentence[:index] + sentence[index+1:]
                    error_list = error_list[:index] + error_list[index+1:]
                    feature_list = feature_list[:index] + feature_list[index + 1:]
                assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                    "Error: sentence, error_list, and feature_list must be of the same size"
                return "", {}, error, error_list, sentence, feature_list
    
    def make_error_PC(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        word = random.choice(PUNCTUATIONS)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_PT(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        extra = random.choice(PUNCTUATIONS)
        if index == 0:
            sentence = [extra] + sentence
            error_list = ['UC'] + error_list
            feature_list = [{}] + feature_list
        elif index == len(sentence) - 1:
            sentence = sentence + [extra]
            error_list = error_list + ['UC']
            feature_list = feature_list + [{}]
        else:
            sentence = sentence[:index] + [extra] + sentence[index:]
            error_list = error_list[:index + 1] + ['UC'] + error_list[index + 1:]
            feature_list = feature_list[:index + 1] + [{}] + feature_list[index + 1:]
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, {}, error, error_list, sentence, feature_list
    
    def make_error_PM(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        # remove punctiuation
        if index == 0:
            sentence = sentence[1:]
            error_list = error_list[1:]
            feature_list = feature_list[1:]
        elif index == len(sentence) - 1:
            sentence = sentence[:-1]
            error_list = error_list[:-1]
            feature_list = feature_list[:-1]
        else:
            sentence = sentence[:index] + sentence[index+1:]
            error_list = error_list[:index] + error_list[index+1:]
            feature_list = feature_list[:index] + feature_list[index + 1:]
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return "", {}, error, error_list, sentence, feature_list
    
    def make_error_MG(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        if len(sentence) < 2:
            error_list[index] = 'UC'
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return word, features, 'UC', error_list, sentence, feature_list
        if index == len(sentence) - 1:
            sentence[index] = sentence[index - 1] + sentence[index]
            word = sentence[index]
            sentence = sentence[:index-1] + sentence[index:]
            error_list = error_list[:index-1] + error_list[index:]
            feature_list = feature_list[:index-1] + feature_list[index:]
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return word, {}, 'UC', error_list, sentence, feature_list
        else:
            sentence[index] = sentence[index] + sentence[index + 1]
            word = sentence[index]
            sentence[index + 1] = ''
            error_list[index+1] = 'UC'
            feature_list[index + 1] = {}
            assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                "Error: sentence, error_list, and feature_list must be of the same size"
            return word, {}, 'UC', error_list, sentence, feature_list
    
    def make_error_SP(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        word = word[:len(word)//2] + " " + word[len(word)//2:]
        word = word.split()
        sentence[index] = word[0]
        extra = word[1]
        insert_idx = index + 1
        
        if insert_idx == len(sentence) - 1:
            sentence = sentence + [extra]
            error_list = error_list + ['UC']
            feature_list = feature_list + [{}]
        else:
            sentence = sentence[:insert_idx] + [extra] + sentence[insert_idx:]
            error_list = error_list[:insert_idx] + ['UC'] + error_list[insert_idx:]
            feature_list = feature_list[:insert_idx] + [{}] + feature_list[insert_idx:]
        word = sentence[index]
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error(self,
                    word: str,
                    features: dict,
                    error: str,
                    sentence: list[str],
                    error_list: list[str],
                    feature_list: list[dict],
                    index: int):
        """
        Induces the error in the word and returns the corrupted word, its features and the error type
        
        Args:
            word (str): input word
            features (dict): morphological features of the word
            error (str): error type
            sentence (list[str]): input sentence tokenized as a list of words
            error_list (list[str]): list of errors induced in the sentence
            index (int): index of the word in the sentence
        Returns:
            Tuple[str, dict, str, List[str], List[str]]: corrupted word, its features, error type,
                                                         updated error list and updated sentence
        """
        
        match error:
            case 'OH':
                return self.make_error_OH(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OT':
                return self.make_error_OT(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OA':
                return self.make_error_OA(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OW':
                return self.make_error_OW(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'ON':
                return self.make_error_ON(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OS':
                return self.make_error_OS(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OG':
                return self.make_error_OG(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OC':
                return self.make_error_OC(word, features, error, sentence,
                                           error_list, feature_list, index)                                           
            case 'OR':
                return self.make_error_OR(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OD':
                return self.make_error_OD(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'OM':
                return self.make_error_OM(word, features, error, sentence,
                                           error_list, feature_list, index)
            case 'MI':
                return self.make_error_MI(word, features, error, sentence,
                                             error_list, feature_list, index)
            case 'MT':
                return self.make_error_MT(word, features, error, sentence,
                                             error_list, feature_list, index)
            case 'XC':
                return self.make_error_XC(word, features, error, sentence,
                                             error_list, feature_list, index)
            case 'XF':
                return self.make_error_XF(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'XG':
                return self.make_error_XG(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'XN':
                return self.make_error_XN(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'XT':
                return self.make_error_XT(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'XM':
                return self.make_error_XM(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'SW':
                return self.make_error_SW(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'SF':
                return self.make_error_SF(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'PC':
                return self.make_error_PC(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'PT':
                return self.make_error_PT(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'PM':
                return self.make_error_PM(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'MG':
                return self.make_error_MG(word, features, error, sentence,
                                                error_list, feature_list, index)
            case 'SP':
                return self.make_error_SP(word, features, error, sentence,
                                                error_list, feature_list, index)
                
            case 'UC':
                assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
                    "Error: sentence, error_list, and feature_list must be of the same size"
                return word, features, error, error_list, sentence, feature_list
            case _:
                raise ValueError(f'Unknown error type: {error}')
    
    def _map_available_errors(self, words: List[str], features: List[dict]) -> List[List[str]]:
        """
        Some errors are not possible to induce in some words, this function maps the
        possible errors for each word in the sentence
        
        Args:
            words (str): input sentence tokenized as a list of words, 
                         must be tokenized on spaces and punctiuation
        Returns:
            errors (List[List[str]]): a list of lists, each list contains the possible errors
        """
        
        # All errors initially can be induced on all words
        initial_errors = ERROR_CLASSES.copy()
        initial_errors.remove('UC')
        errors = [initial_errors.copy() for _ in range(len(words))]
        for i, word, word_features in zip(range(len(words)), words, features):
            initial_set = errors[i].copy()
            for e in initial_set:
                if not self._check_word_err_compatibilities(word, word_features, e):
                    errors[i].remove(e)
        return [list(x) for x in errors]
        
    def _get_morphological_features(self, sentences: List[List[str]]) -> List[List[dict]]:
        """
        Gets the morphological features for each word in the sentence
        
        Args:
            sentence (List[str]): input sentence tokenized as a list of words, 
                                  must be tokenized on spaces and punctiuation
        Returns:
            features (List[dict]): a list of dictionaries, each dictionary contains the morphological features
        """
        features = self.disambiguator.tag_sentences(sentences)
        for i in range(len(features)):
            for j in range(len(features[i])):
                features[i][j] = {k: v for k, v in features[i][j].items() if k not in ['stem', 'stemgloss', 'stemcat']}
        return features
        
    
    def __call__(self, sentences):
        """
        Generates synthetically induced arabic grammatical errors in the input sentence
        Args:
            sentence List[str]: batch of input sentences, must be tokenized on spaces and punctiuation
        Returns:
            dict: a dictionary containing the original sentence, the corrupted sentence and the
                  error type for each token in the sentence (an error could span up to 2 tokens) 
        """
        # sentences = [self._normalize_to_ar_punctuation(sentence) for sentence in sentences]
        tokens = [simple_word_tokenize(sentence)[:256] for sentence in sentences] # avoid going over max length for disambiguation model
        features = self._get_morphological_features(tokens)
        outputs = []
        for sample_tokens, sample_features in zip(tokens, features):
            sample_original_tokens = sample_tokens.copy()
            available_errors_per_token = self._map_available_errors(sample_tokens, sample_features)
            assert len(sample_tokens) == len(sample_features) and len(sample_features) == len(available_errors_per_token), \
                "tokens, features, and available_errors_per_token must be of the same size"
            if not self.tag_distribution:
                for i in range(len(available_errors_per_token)):
                    if self._biased_coin_flip(0.7141): # 71.41% of the time, unchanged sentence (train distibution)
                        available_errors_per_token[i].append('UC')
                err_list = [random.choice(available_errors_per_token[i]) for i in range(len(sample_tokens))]
            else:
                # get the prior for each error type
                for i in range(len(available_errors_per_token)):
                    if self._biased_coin_flip(self.tag_distribution['UC'] / 100):
                        available_errors_per_token[i].append('UC')
                weights = [[self.tag_distribution[e] / 100 for e in errs] for errs in available_errors_per_token]
                err_list = [random.choices(available_errors_per_token[i], weights=weights[i], k=1)[0] for i in range(len(sample_tokens))]
            i = 0
            while i < len(sample_tokens):
                if err_list[i] == 'UC':
                    i += 1
                    continue
                sample_tokens[i], sample_features[i], err_list[i], err_list, sample_tokens, sample_features = \
                    self.make_error(sample_tokens[i], sample_features[i], err_list[i], sample_tokens, err_list, sample_features, i)
                i += 1

            sample_tokens = ' '.join(sample_tokens).split() # remove extra spaces in the middle of the sentence

            outputs.append({
                'correct': ' '.join(sample_original_tokens),
                'incorrect': ' '.join(sample_tokens),
                'errors': err_list
            })
        return outputs
