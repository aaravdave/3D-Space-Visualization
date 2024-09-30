# SILVER
# Simulated Interactions for Launch Visualization and Earth Rotation

from tkinter import *
import tkintermapview
# from PIL import ImageTk, Image
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from interplanetary import init as interplanetary
from surface import init as surface


def information():
    info_window = Toplevel(window)
    info_window.geometry('800x800')
    info_window.title('Information')

    info_content = Frame(info_window)
    info_content.place(relx=0.5, rely=0.5, anchor=CENTER)

    map = tkintermapview.TkinterMapView(info_content, width=800, height=600, corner_radius=0)
    map.pack()
    map.set_address('Space Commerce Way, Merritt Island, FL 32953', marker=True)

    info_location = Label(info_content, text='Artemis II will use the SLS system paired with the Orion spacecraft. This will take off from the Kennedy Space Center in Cape Canaveral, Florida.', font=small, width=100, wraplength=600)
    info_location.pack()

    # info_logo = ImageTk.PhotoImage(Image.open('assets/system/youngwonks.png'))
    # info_logo = Label(info_content, image=info_logo)
    # info_logo.pack()
    #
    # info_team = Label(info_content, text='Created for the NASA App Development Challenge. Team YoungWonks.', font=small, width=100, wraplength=600)
    # info_team.pack()


window = Tk()
window.geometry('800x600')
window.title('SILVER')
window.resizable(False, False)
large = ('JetBrains Mono', 30)
small = ('JetBrains Mono', 15)

content = Frame(window)
content.place(relx=0.5, rely=0.5, anchor=CENTER)
title = Label(content, text='SILVER', font=large)
title.pack()
subtitle = Label(content, text='Simulated Interactions for Launch Visualization and Earth Rotation', font=small)
subtitle.pack()
start_interplanetary = Button(content, text='Interplanetary Simulation', command=interplanetary, font=large, width=32)
start_interplanetary.pack()
start_surface = Button(content, text='Surface Simulation', command=surface, font=large, width=32)
start_surface.pack()
about = Button(content, text='Additional Information', command=information, font=large, width=32)
about.pack()

window.mainloop()
