

from RealTimeListenerModule import RealTimeKeyListener
from OfflineListenerModule import OfflineListener
offline_lstnr = OfflineListener()
offline_lstnr.read_text_file('C:/Users/user/Desktop/tm.docx')
offline_lstnr.read_text_file('C:/Users/user/Desktop/tmp.txt')
offline_lstnr.read_text_file('C:/Users/user/Desktop/tmp.pdf')

import nltk

sentence = "Sh likes playing fotball."
destination_json_file_path = "C:/Users/user/Desktop/destination_file.json"
sentence_tokenizer = nltk.tokenize.RegexpTokenizer('[\W_]', gaps=True)

