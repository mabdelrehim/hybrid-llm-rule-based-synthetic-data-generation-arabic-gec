import random
import warnings
import json
import logging
import torch
import os

from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.reinflector import Reinflector

from openai import AzureOpenAI 
from pydantic import BaseModel

from pyarabic import araby

from typing import List, Tuple

from gec.corruption.rule_based import (RuleBasedCorruptor, ERROR_CLASSES, CAMEL_MORPHOLOGY_FEATURES,
                                       ARABIC_PREPOSITIONS, ARABIC_CONJUNCTIONS, PUNCTUATIONS)

from transformers import pipeline, AutoTokenizer
import pprint as pp

# logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)


class GECSchema(BaseModel):
    expanation: str
    errored_word: str


class HybridGPTRuleBasedCorruptor(RuleBasedCorruptor):
    def __init__(self, tag_distribution=None, seed=42, error_description=None,
                 model_name="gpt-4o"):
        super().__init__(tag_distribution, seed)
        with open(error_description) as f:
            err_definitions = json.load(f)
    
        err_def_str = pp.pformat(err_definitions)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        endpoint = os.getenv("ENDPOINT_URL", "https://aragpt.openai.azure.com/")  
        self.deployment = os.getenv("DEPLOYMENT_NAME", model_name)  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")  
        
        # Initialize Azure OpenAI Service client with key-based authentication    
        self.client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-08-01-preview",
        )
        
        self.system_prompt = [{"role": "system", "content": "* You are an AI assistant that is an expert in the arabic language. Your Goal is to induce the specified grammatical error.\n"
                                                            "* You should provide your answer in the following json schema: {\"explanation\": string, \"errored_word\": string}\n"
                                                            "* Your output should be of correct json format and your explaination should be brief and to the point like the given examples.\n"
                                                            "* If it's not possible to induce the error, return the origianl word in the errored_word field"}]
                                                            ## "* Here are the definition of common grammatical errors in arabic and examples on how you can induce arabic grammatical errors for each error type:"
                                                            ## f"\n{err_def_str}\n"},]
        
    def _call_llm(self, prompt, original_word):
        # Generate the completion
        try:  
            completion = self.client.beta.chat.completions.parse(  
                model=self.deployment,
                messages=prompt,
                max_tokens=2048,  
                temperature=0.7,  
                top_p=0.95,  
                frequency_penalty=0,  
                presence_penalty=0,
                stop=None,
                response_format=GECSchema
            )
            output = completion.choices[0].message.parsed
            self.total_prompt_tokens += completion.usage.prompt_tokens
            self.total_completion_tokens += completion.usage.completion_tokens
            logging.debug(f"Explaination: {output.expanation}")
            logging.debug(f"Errored Word:  {output.errored_word}")
            logging.debug(f"Prompt Tokens Used Thus Far: {self.total_prompt_tokens}")
            logging.debug(f"Completion Tokens Used Thus Far: {self.total_completion_tokens}")
            return(output.errored_word)
        except Exception as e:
            logging.warning(f"Failed to generate a response for the word {original_word}\nError: {e}\n\nOriginal Prompt: {pp.pformat(prompt)}")
            return original_word
        
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
                return True
            case 'MT':
                return (features['pos'] == 'verb')
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
        
    def make_error_MI(self,
                      word: str,
                      features: dict,
                      error: str,
                      sentence: List[str],
                      error_list: List[str],
                      feature_list: List[dict],
                      index: int):
        prompt = self.system_prompt + \
            [{"role": "user", "content": f"{error} - Write a different morphological inflection of the word معروف the word should have the same root which is عرف but it should have a different meaning."}, {"role": "assistant", "content": "{\"explanation\": \"The word معروف is a passive participle (ism maf3ool) of the verb عرف which means known. The active participle (ism fa3el) of the verb عرف is عارف which means knowledgeable.\", \"errored_word\": \"عارف\"}"},
             {"role": "user", "content": f"{error} - Write a different morphological inflection of the word رفيق the word should have the same root which is رفق but it should have a different meaning."}, {"role": "assistant", "content": "{\"explanation\": \"The word رفيق is a noun that means companion. A different noun from the same root that means elbow is the word مرفق\", \"errored_word\": \"مرفق\"}"},
             {"role": "user", "content": f"{error} - Write a different morphological inflection of the word مدرسة the word should have the same root which is درس but it should have a different meaning."}, {"role": "assistant", "content": "{\"explanation\": \"The word مدرسة is a noun that means school. A different noun from the same root that means a study is the word دراسة\", \"errored_word\": \"دراسة\"}"},
             {"role": "user", "content": f"{error} - Write a different morphological inflection of the word انتباه the word should have the same root which is نبه but it should have a different meaning."}, {"role": "assistant", "content": "{\"explanation\": \"The word انتباه is a noun that means attention. A different verb from the same root that means to alert is the word نبه\", \"errored_word\": \"نبه\"}"},
             {"role": "user", "content": f"{error} - Write a different morphological inflection of the word {word} the word should have the same root which is {features['root'].replace('.', '')} but it should have a different meaning."}]
        word = self._call_llm(prompt, word)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_MT(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        prompt = self.system_prompt + \
            [{"role": "user", "content": f"_MT_ - Write the verb سيرحل in a different tense. (choose from past/present/future)"}, {"role": "assistant", "content": "{\"explanation\": \"The verb سيرحل is in the future tense. Its past tense is رحل.\", \"errored_word\": \"رحل\"}"},
             {"role": "user", "content": f"_MT_ - Write the verb اخذ in a different tense. (choose from past/present/future)"}, {"role": "assistant", "content": "{\"explanation\": \"The verb اخذ is in the past tense. Its present tense is يأخذ.\", \"errored_word\": \"يأخذ\"}"},
             {"role": "user", "content": f"_MT_ - Write the verb يهرب in a different tense. (choose from past/present/future)"}, {"role": "assistant", "content": "{\"explanation\": \"The verb يهرب is in the present tense. Its future tense is سيهرب.\", \"errored_word\": \"سيهرب\"}"},
             {"role": "user", "content": f"_MT_ - Write the verb افرحتني in a different tense. (choose from past/present/future)"}, {"role": "assistant", "content": "{\"explanation\": \"The verb افرحتني is in the past tense. Its present tense is تفرحني.\", \"errored_word\": \"تفرحني\"}"},
             {"role": "user", "content": f"_MT_ - Write the verb {word} in a different tense. (choose from past/present/future)"}]
        word = self._call_llm(prompt, word)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list
    
    def make_error_XC(self,
                       word: str,
                       features: dict,
                       error: str,
                       sentence: list[str],
                       error_list: list[str],
                       feature_list: list[dict],
                       index: int):
        prompt = self.system_prompt + \
            [{"role": "user", "content": f"_XC_ - Write the word رائع in a different case. (choose from nominative/accusative/genitive)"}, {"role": "assistant", "content": "{\"explanation\": \"The word رائع is in the nominative case. Its accusative case is رائعا.\", \"errored_word\": \"رائعا\"}"},
             {"role": "user", "content": f"_XC_ - Write the word جميلًا in a different case. (choose from nominative/accusative/genitive)"}, {"role": "assistant", "content": "{\"explanation\": \"The word جميلًا is in the accusative case. Its genative case is جميلا.\", \"errored_word\": \"جميلا\"}"},
             {"role": "user", "content": f"_XC_ - Write the word واحد in a different case. (choose from nominative/accusative/genitive)"}, {"role": "assistant", "content": "{\"explanation\": \"The word واحد is in the nominative case. Its accusative case is واحدا.\", \"errored_word\": \"واحدا\"}"},
             {"role": "user", "content": f"_XC_ - Write the word تكريما in a different case. (choose from nominative/accusative/genitive)"}, {"role": "assistant", "content": "{\"explanation\": \"The word تكريما is in the accusative case. Its nominative case is تكريم.\", \"errored_word\": \"تكريم\"}"},
             {"role": "user", "content": f"_XC_ - Write the word {word} in a different case. (choose from nominative/accusative/genitive)"}]
        word = self._call_llm(prompt, word)
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
        letter_to_word = {'m': 'male', 'f': 'female'}
        reverse = {'male': 'female', 'female': 'male'}
        gender = letter_to_word.get(features['gen'], 'N/A')
        new_gender = reverse.get(gender, 'N/A')
        prompt = self.system_prompt + \
            [{"role": "user", "content": f"_XG_ - change the gender of the word مستقل from male to female."}, {"role": "assistant", "content": "{\"explanation\": \"The word مستقل is of male gender. The female gender of it is مستقلة.\", \"errored_word\": \"مستقلة\"}"},
             {"role": "user", "content": f"_XG_ - change the gender of the word معلمات from female to male."}, {"role": "assistant", "content": "{\"explanation\": \"The word معلمات is of female gender, the male gender of it is معلميين.\", \"errored_word\": \"معلميين\"}"},
             {"role": "user", "content": f"_XG_ - change the gender of the word طالبان from male to female."}, {"role": "assistant", "content": "{\"explanation\": \"The word طالبان is of male gender, the female gender of it is طالبتان.\", \"errored_word\": \"طالبتان\"}"},
             {"role": "user", "content": f"_XG_ - change the gender of the word لاعبة from female to male."}, {"role": "assistant", "content": "{\"explanation\": \"The word لاعبة is of female gender, the male gender of it is لاعب.\", \"errored_word\": \"لاعب\"}"},
             {"role": "user", "content": f"_XG_ - change the gender of the word {word} from {gender} to {new_gender}."}]
        word = self._call_llm(prompt, word)
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
        letter_to_word = {'s': 'singular', 'd': 'two', 'p': 'plural'}
        num = letter_to_word.get(features['num'], 'N/A')
        new_num = random.choice([x for x in CAMEL_MORPHOLOGY_FEATURES['num'] if x != features['num']])
        new_num = letter_to_word.get(new_num, 'N/A')
        
        prompt = self.system_prompt + \
            [{"role": "user", "content": f"_XN_ - change the number of the word معلمان from two to plural."}, {"role": "assistant", "content": "{\"explanation\": \"The word معلمان is of dual number. The plural form of it is معلمون.\", \"errored_word\": \"معلمون'}"},
             {"role": "user", "content": f"_XN_ - change the number of the word طالبة from singular to plural."}, {"role": "assistant", "content": "{\"explanation\": \"The word طالبة is of singular number. The plural form of it is طالبات.\", \"errored_word\": \"طالبات\"}"},
             {"role": "user", "content": f"_XN_ - change the number of the word طبيب from singular to two."}, {"role": "assistant", "content": "{\"explanation\": \"The word طبيب is of singular number. The dual form of it is طبيبان.\", \"errored_word\": \"طبيبان\"}"},
             {"role": "user", "content": f"_XN_ - change the number of the word كلاب from plural to singular."}, {"role": "assistant", "content": "{\"explanation\": \"The word كلاب is of plural number. The singular form of it is كلبv., \"errored_word\": \"كلب\"}"},
             {"role": "user", "content": f"_XN_ - change the number of the word {word} from {num} to {new_num}."}]
        word = self._call_llm(prompt, word)
        sentence[index] = word
        error_list[index] = error
        feature_list[index] = features
        
        assert len(sentence) == len(error_list) and len(sentence) == len(feature_list), \
            "Error: sentence, error_list, and feature_list must be of the same size"
        return word, features, error, error_list, sentence, feature_list