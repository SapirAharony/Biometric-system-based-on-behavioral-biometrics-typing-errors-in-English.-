import tkinter as tk
import tkinter.messagebox
import pynput.keyboard as keyboard
from RealTimeListenerModule import RealTimeKeyListener
import tkinter.font as font
from tkinter import filedialog

title = "Listener by BM"

def on_start():
    global listener, folder_selected, title
    print(folder_selected)
    if listener:
        tkinter.messagebox.showwarning(title=title, message="Listener IS running!")
    else:
        listener = RealTimeKeyListener()
        if folder_selected is not None:
            listener.destination_json_file_path = folder_selected
        tkinter.messagebox.showwarning(title=title, message="Listener IS running! Your destination file is: " +
                                                            str(listener.destination_json_file_path))

def press_end_combination():
    controler = keyboard.Controller()
    controler.press(keyboard.Key.esc)
    controler.press(keyboard.Key.f4)


def on_stop():
    global listener
    if isinstance(listener, RealTimeKeyListener):
        tkinter.messagebox.showwarning(title=title,
                                       message="Listener has just been stopped. \nYou can start it again, "
                                               "quit or change destination file path")
        press_end_combination()
        listener.keyboard_listener.join()
        listener.mouse_listener.join()
        listener = None
    else:
        tkinter.messagebox.showwarning(title=title, message="Listener IS NOT running!")

def set_path():
    global listener, folder_selected
    folder_selected = filedialog.askdirectory()
    if isinstance(listener, RealTimeKeyListener):
        listener. destination_json_file_path = folder_selected
    print(folder_selected)


if __name__ == '__main__':
    listener, folder_selected = None, None
    root = tk.Tk()
    root.geometry('300x500')
    root.resizable(True, True)
    root.title('Listener Demo')
    myFont = font.Font(weight="bold", family='Courier', size=12)

    # start_button
    start_button = tk.Button(root, command=on_start, text="Start Listener", bg="#7E85F8", width=20, height=1)
    start_button['font'] = myFont
    start_button.pack(ipadx=5, ipady=5, expand=True)

    # stop_button
    stop_button = tk.Button(root, command=on_stop, text="Stop Listener", bg="#7E85F8", width=20, height=1)
    stop_button.pack(ipadx=5, ipady=5, expand=True)
    stop_button['font'] = myFont

    set_path_button = tk.Button(root, command=set_path, text="Set path for end files", bg="#FF8767", width=20, height=1)
    set_path_button.pack(ipadx=5, ipady=5, expand=True)
    set_path_button['font'] = myFont

    # exit_button
    exit_button = tk.Button(root, text='Exit', command=root.destroy, bg="#FF8767", width=20, height=1)
    exit_button.pack(ipadx=5, ipady=5, expand=True)
    exit_button['font'] = myFont

    root.mainloop()
