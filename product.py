class Product(object):
    def __init__(self, name, price, market):
        self.name = name
        self.price = price
        self.market = market

    def __str__(self):
        return '{name: ' + self.name + ', price: ' + self.price + ', market: ' + self.market + '}'

    def __repr__(self):
        return '{name: ' + self.name + ', price: ' + self.price + ', market: ' + self.market + '}'