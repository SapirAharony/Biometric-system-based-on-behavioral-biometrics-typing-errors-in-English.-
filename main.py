'''
chunking




# def check_word_spelling(word):
#     word = Word(word)
#     result = word.spellcheck()
#     if word == result[0][0]:
#         print(f'Spelling of "{word}" is correct!')
#     else:
#         print(f'Spelling of "{word}" is not correct!')
#         print(f'Correct spelling of "{word}": "{result[0][0]}" (with {result[0][1]} confidence).')

sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
correct_sentence = 'This are a sentence to check!'
wrong_sentence = "I loves mine family very much.!"
'''
import language_tool_python
from textblob import Word
import nltk
from spellchecker import SpellChecker
my_tool = language_tool_python.LanguageTool('en-US')
my_tool.enable_spellchecking()

# given text
my_text = """I would like to met you at 8am evening. It is really important to me, becouse I were a piece of the shit."""


from System_features_extractor import correct_spelling_autocorrect, correct_spelling_txt_blb, correct_spelling_spell_checker, candidates_to_correct_spelling_spell_checker, grammar_check
sentences = [
    'This are to checks!',
    "I loves mine family very much.!",
    "A apple is very tastefull. ",
    "He were a good guy.",
    "I really loves this plant.",
    "He is my children.",
    "My bofriends is a nic guy",
    "Our friend is a cybersecuirty specialists",
    "I just wanna test my to ols which I made. "
        ]


sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[^(\'\-)\w]', gaps=True)
i = 1
for sentence in sentences:
    tokenized_sentence = sentence_tokenizer.tokenize(sentence)
    spell_chckr_candidates = [candidates_to_correct_spelling_spell_checker(word) for word in tokenized_sentence]
    spell_chckr_correct = [correct_spelling_spell_checker(word) for word in tokenized_sentence]
    print ("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%", i, "%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    my_matches = my_tool.check(sentence)
    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()
    matches = [rule for rule in my_matches if not is_bad_rule(rule)]
    print("Sentence to correct: ", sentence)
    print("Spellcheckers:")
    print("\t- Text Blob: ", (str(correct_spelling_txt_blb(sentence.lower()))))
    print("\t- Spell checker correct: ", " ".join(spell_chckr_correct))
    print("\t- Spell checker candidates: ", spell_chckr_candidates)
    print("\t- Autocorrect: ", correct_spelling_autocorrect(sentence))
    print("Grammar checkers: ")
    print("\t- Gingerit: ", str(grammar_check(sentence.lower())))
    print("\t- Language_tool_python: ", language_tool_python.utils.correct(sentence, matches))
    print(matches)
    print()
    i +=1
my_tool.close()



