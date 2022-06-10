import tkinter as tk
import tkinter.messagebox
import pynput.keyboard as keyboard
from RealTimeListenerModule import RealTimeKeyListener
from OfflineListenerModule import OfflineListener
import tkinter.font as font
from tkinter import filedialog

title = "Listener by BM"

def on_online_start():
    global online_listener, folder_selected, title
    print(folder_selected)
    if online_listener:
        tkinter.messagebox.showwarning(title=title, message="Listener IS running!")
    else:
        online_listener = RealTimeKeyListener()
        if folder_selected is not None:
            online_listener.destination_json_file_path = folder_selected
        tkinter.messagebox.showwarning(title=title, message="Listener IS running! Your destination file is: " +
                                                            str(online_listener.destination_json_file_path))

def press_end_combination():
    controler = keyboard.Controller()
    controler.press(keyboard.Key.esc)
    controler.press(keyboard.Key.f4)


def online_stop():
    global online_listener
    if isinstance(online_listener, RealTimeKeyListener):
        tkinter.messagebox.showwarning(title=title,
                                       message="Listener has just been stopped. \nYou can start it again, "
                                               "quit or change destination file path")
        press_end_combination()
        online_listener.keyboard_listener.join()
        online_listener.mouse_listener.join()
        online_listener = None
    else:
        tkinter.messagebox.showwarning(title=title, message="Listener IS NOT running!")


def set_path():
    global online_listener, folder_selected
    folder_selected = filedialog.askdirectory()
    if isinstance(online_listener, RealTimeKeyListener):
        online_listener. destination_json_file_path = folder_selected

def offline_start():
    offline_lstnr = OfflineListener()
    file_selected = filedialog.askopenfilename(filetypes=[('PDFs', '*pdf'), ('DOCXs', '*docx'), ('TXTs', '*.txt')])
    tkinter.messagebox.showinfo(title=title,
                                   message="You have just chosen" + str(file_selected) + "Destination_file is: " + offline_lstnr.destination_json_file_path)
    offline_lstnr.read_text_file(file_selected)

if __name__ == '__main__':
    online_listener, folder_selected = None, None
    root = tk.Tk()
    root.geometry('300x500')
    root.resizable(True, True)
    root.title('Listener Demo')
    myFont = font.Font(weight="bold", family='Courier', size=12)

    # start_button
    start_button = tk.Button(root, command=on_online_start, text="Start Online Listener", bg="#7E85F8", width=20, height=1)
    start_button['font'] = myFont
    start_button.pack(ipadx=5, ipady=5, expand=True)

    # stop_button
    stop_button = tk.Button(root, command=online_stop, text="Stop Online Listener", bg="#7E85F8", width=20, height=1)
    stop_button.pack(ipadx=5, ipady=5, expand=True)
    stop_button['font'] = myFont

    offline_btn = tk.Button(root, command=offline_start, text="Offline Listener", bg="#7E85F8", width=20, height=1)
    offline_btn.pack(ipadx=5, ipady=5, expand=True)
    offline_btn['font'] = myFont


    set_path_button = tk.Button(root, command=set_path, text="Set path for end files", bg="#FF8767", width=20, height=1)
    set_path_button.pack(ipadx=5, ipady=5, expand=True)
    set_path_button['font'] = myFont

    # exit_button
    exit_button = tk.Button(root, text='Exit', command=root.destroy, bg="#FF8767", width=20, height=1)
    exit_button.pack(ipadx=5, ipady=5, expand=True)
    exit_button['font'] = myFont

    root.mainloop()
