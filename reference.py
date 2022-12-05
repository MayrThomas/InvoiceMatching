import json

from product import Product


def __prepare_unimarkt_data():
    file = open("unimarktData.json", "r", encoding="utf-8")
    json_string = file.read()

    unimarkt_products = json.loads(json_string)

    products = list()
    for val in unimarkt_products:
        products.append(Product(val['productName'], val['actualPrice'], 'Unimarkt'))

    return products


def prepare_reference_data():
    reference_products = list()
    reference_products.extend(__prepare_unimarkt_data())
    return reference_products
