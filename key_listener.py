#! python
import json
# listeners
import pynput.keyboard as keyboard
import pynput.mouse as mouse
import nltk.data
import System_features_extractor as s_f_extractor


class Combinations:
    """A class which includes key combinations or sets of keys."""
    END_COMBINATION = [keyboard.Key.esc, keyboard.Key.f4]  # key combination to finish listening
    NEXT_WORD_KEYS = [keyboard.Key.space, ':', ",", "/", '"']  # key of next words
    NEW_CONTEXT_KEYS = [keyboard.Key.down, keyboard.Key.up]  # , keyboard.Key.dot},
    SENTENCE_END = [keyboard.Key.enter, ".", ";", '?', '!']
    NUMPAD_NUMBERS = [keyboard.Key.n_zero, keyboard.Key.n_one, keyboard.Key.n_two,
                      keyboard.Key.n_three, keyboard.Key.n_four,
                      keyboard.Key.n_five, keyboard.Key.n_six, keyboard.Key.n_seven,
                      keyboard.Key.n_eight, keyboard.Key.n_nine]


class RealTimeKeyListener:
    __left_button_mouse_is_pressed = False
    __position = 0
    __previous_key = None
    __sentence = ""
    __list_of_words = None
    __keys_counter = {}
    __key_string_list = list()
    keyboard_listener = None
    mouse_listener = None

    def __init__(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.__on_press)
        self.mouse_listener = mouse.Listener(on_click=self.__on_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()
        self.keyboard_listener.join()
        self.mouse_listener.join()
        
    def __on_press(self, key):
        print("\n\n\nSentence: ", self.__sentence)
        print('position: ', self.__position)
        self.__insert_key(key)

    def __on_click(self, x, y, button, pressed):
        self.__count_clicks(button)
        if pressed and button == mouse.Button.left:
            # print("Clicked: ", button)
            self.__left_button_mouse_is_pressed = True

    def __count_clicks(self, button):
        """A method that adds each mouse button click to dictionary collecting each click event."""
        if str(button) not in self.__keys_counter.keys():
            self.__keys_counter[str(button)] = 1
        else:
            self.__keys_counter[str(button)] += 1

    def __insert_key(self, key):
        char = ''
        if hasattr(key, 'char') and key.char is not None and len(key.char) < 2:
            char = key.char
        if char is not None:
            if self.__position == 0:
                self.__sentence += char
            else:
                self.__sentence = self.__sentence[:self.__position] + char + self.__sentence[self.__position:]


# def add_dict_list_to_json_file(path_to_file, key, obj):
#     # check if is empty
#     if os.path.isfile(path_to_file) and os.path.getsize(path_to_file) > 0:
#         data = s_f_extractor.read_json_file(path_to_file)
#         open(path_to_file, 'w').close()
#         file = open(path_to_file, 'a+')
#         if isinstance(obj, dict) and obj.keys() and key in data.keys():
#             for k in obj.keys():
#                 if k not in data[key].keys():
#                     data[key][k] = obj[k]
#                 elif k in data[key].keys() and (isinstance(data[key][k], int) or isinstance(data[key][k], float)):
#                     data[key][k] += obj[k]
#
#         elif key in data.keys() and isinstance(obj, list) and obj and isinstance(data[key], list):
#             data[key].extend(obj)
#         else:
#             data[key] = obj
#             file.seek(0)
#         json.dump(data, file, indent=4)
#         file.close()