# imports
from input import *
from reference import *
import jellyfish

x = '{"name": "Avocados 500g", "quantity": null, "unitPrice": null, "totalPrice": {"value": 1.46, "currency": "EUR", ' \
    '"convertedValue": null, "convertedCurrency": null, "convertTime": null}} '


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reference_products = prepare_reference_data()

    result = read_invoice(x)
    print("extracted product name: -- ", result)

    print()
    print("Jaro-Winkler match for --", result.name)
    print("---------------------------------------------")
    best_match = -1
    match_product = ''

    for product in reference_products:
        match = jellyfish.jaro_winkler_similarity(product.name, result.name)

        if match > best_match:
            best_match = match
            match_product = product

    print("product: --", match_product.name)
    print("value: --", best_match)
    print()
    print("Levenshtein match for --", result.name)

    shortest_distance = float("inf")

    for product in reference_products:
        distance = jellyfish.levenshtein_distance(product.name, result.name)

        if distance < shortest_distance:
            shortest_distance = distance
            match_product = product

    print("product: --", match_product.name)
    print("distance: --", shortest_distance)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
