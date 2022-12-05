# imports
import json
from types import SimpleNamespace


def read_invoice(invoice):
    return json.loads(invoice, object_hook=lambda d: SimpleNamespace(**d))
