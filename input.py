# imports
import json
from types import SimpleNamespace
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo


def read_invoice(invoice):
    return json.loads(invoice, object_hook=lambda d: SimpleNamespace(**d))


def read_invoices(filename):
    fdata = open(filename, encoding="utf-8")
    invoices = json.load(fdata, object_hook=lambda d: SimpleNamespace(**d))
    showinfo("Invoice import", "Invoice import completed successfully!!")
    return invoices


def open_invoice_file():
    Tk().withdraw()
    return askopenfilename()
